import os
from typing import Union
import hydra
import yaml
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
import time
import datetime
import logging

import torch
from torch import Tensor, nn

import MinkowskiEngine as ME

import lightning as L
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from lightning.fabric.loggers import TensorBoardLogger, CSVLogger
from torchmetrics.aggregation import RunningMean

from transformers import get_cosine_schedule_with_warmup

from emsim.networks import (
    EMModel,
    EMCriterion,
)
from emsim.geant.dataset import (
    make_test_train_datasets,
    electron_collate_fn,
    worker_init_fn,
)
from emsim.geant.io import convert_electron_pixel_file_to_hdf5
from emsim.preprocessing import NSigmaSparsifyTransform


_logger = logging.getLogger(__name__)

torch._dynamo.config.capture_scalar_outputs = True
torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    _logger.setLevel(cfg.log_level)
    _logger.info("Starting...")
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    if cfg.log_tensorboard:
        tb_logger = TensorBoardLogger(
            output_dir,
            name=(
                cfg.tensorboard_name
                if cfg.tensorboard_name is not None
                else "lightning_logs"
            ),
        )
        tb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    fabric = Fabric(
        strategy=DDPStrategy(find_unused_parameters=cfg.ddp.find_unused_parameters),
        # strategy="ddp_find_unused_parameters_true",
        accelerator="gpu",
        num_nodes=cfg.ddp.nodes,
        devices=cfg.ddp.devices,
        loggers=(
            [
                tb_logger,
                # CSVLogger(output_dir + "/csv_logs"),
            ]
            if cfg.log_tensorboard
            else None
        ),
    )
    if fabric.is_global_zero:
        _logger.info("Setting up...")
        _logger.info(print(yaml.dump(OmegaConf.to_container(cfg, resolve=True))))
    fabric.seed_everything(cfg.seed + fabric.global_rank)
    fabric.launch()
    model = EMModel.from_config(cfg)
    if cfg.unet.convert_sync_batch_norm and fabric.world_size > 1:
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    if cfg.compile:
        if fabric.is_global_zero:
            _logger.info("torch.compile-ing model...")
        model = torch.compile(model, dynamic=True)

    electron_hdf_file = os.path.join(
        cfg.dataset.directory, os.path.splitext(cfg.dataset.pixels_file)[0] + ".hdf5"
    )
    if not os.path.exists(electron_hdf_file):
        if fabric.is_global_zero:
            pixels_file = os.path.join(cfg.dataset.directory, cfg.dataset.pixels_file)
            _logger.info(f"Converting {pixels_file} to {electron_hdf_file}...")
            convert_electron_pixel_file_to_hdf5(pixels_file, electron_hdf_file)
            _logger.info("Done converting.")
    fabric.barrier()
    train_dataset, eval_dataset = make_test_train_datasets(
        electron_hdf_file=electron_hdf_file,
        events_per_image_range=cfg.dataset.events_per_image_range,
        pixel_patch_size=cfg.dataset.pixel_patch_size,
        hybrid_sparse_tensors=False,
        train_percentage=cfg.dataset.train_percentage,
        noise_std=cfg.dataset.noise_std,
        transform=NSigmaSparsifyTransform(
            cfg.dataset.n_sigma_sparsify,
            cfg.dataset.pixel_patch_size,
            max_pixels_to_keep=cfg.dataset.max_pixels_to_keep,
        ),
        shared_shuffle_seed=cfg.seed,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        cfg.batch_size,
        collate_fn=electron_collate_fn,
        pin_memory=True,
        num_workers=cfg.dataset.num_workers,
        worker_init_fn=worker_init_fn,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        cfg.batch_size,
        collate_fn=electron_collate_fn,
        pin_memory=True,
        num_workers=cfg.dataset.num_workers,
        worker_init_fn=worker_init_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(cfg.num_steps * cfg.warmup_percentage),
        cfg.num_steps,
    )

    ## prepare fabric
    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader, eval_dataloader = fabric.setup_dataloaders(
        train_dataloader, eval_dataloader
    )

    if cfg.resume_file is not None:
        start_iter = load(cfg.resume_file, model, optimizer, fabric)
    else:
        start_iter = 0

    train(
        cfg,
        fabric,
        model,
        optimizer,
        lr_scheduler,
        train_dataloader,
        eval_dataloader,
        os.path.join(output_dir, "checkpoints"),
        start_iter,
    )


def train(
    cfg: DictConfig,
    fabric: Fabric,
    model,
    optimizer,
    lr_scheduler,
    train_dataloader,
    eval_dataloader,
    save_dir,
    start_iter: int = 0,
):
    if cfg.ddp.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    ## main training loop
    iter_timer = RunningMean(cfg.print_interval).to(fabric.device)
    model.train()
    iter_loader = iter(train_dataloader)
    if fabric.is_global_zero:
        _logger.info("Begin training.")
    epoch = 0
    criterion: EMCriterion = model.criterion
    training_start_time = time.time()
    for i in range(start_iter, cfg.num_steps):
        t0 = time.time()
        try:
            batch = next(iter_loader)
        except StopIteration:
            eval(epoch, i, t0, model, eval_dataloader, fabric)
            model.train()
            epoch += 1
            iter_loader = iter(train_dataloader)
            batch = next(iter_loader)

        loss_dict, model_output = model(batch)
        total_loss = loss_dict["loss"]
        fabric.backward(total_loss)
        fabric.clip_gradients(model, optimizer, cfg.max_grad_norm)
        if cfg.ddp.find_unused_parameters:
            __debug_find_unused_parameters(model)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        with torch.no_grad():
            log_dict = fabric.all_reduce(loss_dict, reduce_op="mean")
        log_dict["lr"] = lr_scheduler.get_last_lr()[0]

        if i > 0 and i % cfg.print_interval == 0:
            criterion.update_detection_metrics(criterion.train_metrics["detection"], model_output, batch)
            metric_log_dict = criterion.get_train_logs()
            log_str = criterion.make_log_str(metric_log_dict)
            elapsed_time = time.time() - training_start_time
            elapsed_time_str = _elapsed_time_str(elapsed_time)
            iter_time = iter_timer.compute()

            log_str = f"Iter {i} (Epoch {epoch}) -- ({elapsed_time_str}) -- " + log_str
            log_str = log_str + f" iter_time: {iter_time}\n"
            if fabric.is_global_zero:
                _logger.info(log_str)
            metric_log_dict["iter_time"] = iter_time
            log_dict.update(metric_log_dict)

        log_dict = criterion.format_log_keys(log_dict)
        fabric.log_dict(log_dict, step=i)

        if i > 0 and i % cfg.eval_steps == 0:
            save(save_dir, f"step_{i}", model, optimizer, fabric, i)
            eval(epoch, i, t0, model, eval_dataloader, fabric)
            model.train()

        iter_timer.update(time.time() - t0)
        # MinkowskiEngine says to clear the cache periodically
        if i > 0 and i % cfg.clear_cache_interval == 0:
            torch.cuda.empty_cache()

    save(save_dir, "final", model, optimizer, fabric, i)
    if fabric.is_global_zero:
        elapsed_time_str = _elapsed_time_str(time.time() - training_start_time)
        _logger.info(f"Training complete in {elapsed_time_str}.")


@torch.no_grad()
def eval(
    epoch: int,
    step: int,
    start_time: float,
    model: nn.Module,
    eval_loader: torch.utils.data.DataLoader,
    fabric: Fabric,
):
    start_eval = time.time()
    model.eval()
    criterion: EMCriterion = model.criterion
    for batch in eval_loader:
        output = model(batch)
        criterion.eval_batch(output, batch)
    metric_log_dict = criterion.get_eval_logs()
    metric_log_dict = criterion.format_log_keys(metric_log_dict)
    log_str = criterion.make_log_str(metric_log_dict)
    elapsed_time = time.time() - start_time
    elapsed_time_str = _elapsed_time_str(elapsed_time)
    eval_time_str = _elapsed_time_str(time.time() - start_eval)
    log_str = (
        f"Evaluation: Iter {step} (Epoch {epoch}) -- ({elapsed_time_str}) -- " + log_str
    )
    log_str = log_str + f"Eval time: {eval_time_str}"
    if fabric.is_global_zero:
        _logger.info(log_str)
    fabric.log_dict(metric_log_dict, step)


def save(
    save_dir: str, save_name: str, model, optimizer, fabric: Fabric, iteration: int
):
    state = {"model": model, "optimizer": optimizer, "iteration": iteration}
    save_file = os.path.join(save_dir, save_name + ".ckpt")
    fabric.save(save_file, state)
    if fabric.is_global_zero:
        _logger.info(f"Saved to {save_file}\n")


def load(checkpoint_file: str, model, optimizer, fabric: Fabric):
    state = {"model": model, "optimizer": optimizer}
    remainder = fabric.load(checkpoint_file, state)
    iteration = remainder["iteration"]
    return iteration


def _elapsed_time_str(elapsed_time):
    return str(datetime.timedelta(seconds=int(elapsed_time)))


class CudaUsageMonitor(nn.Module):
    MB = 1024**2

    def __init__(self, sample_window: int):
        self.utilization = RunningMean(sample_window)
        self.memory = RunningMean(sample_window)
        self.max_memory = RunningMean(sample_window)

    def update(self):
        self.utilization.update(torch.cuda.utilization(device=self.utilization.device))
        self.memory.update(torch.cuda.memory_allocated(device=self.memory.device))
        self.max_memory.update(
            torch.cuda.max_memory_allocated(device=self.max_memory.device)
        )

    def compute(self):
        utilization = self.utilization.compute()
        memory = self.memory.compute() // self.MB
        max_memory = self.max_memory.compute() // self.max_memory
        return utilization, memory, max_memory


def __debug_find_unused_parameters(model):
    unused_parameters = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused_parameters.append(name)
    if len(unused_parameters) > 0:
        _logger.debug(f"Unused parameters: {unused_parameters}")


# Doesn't work because can't get each worker's dataset's state
# def verify_data_integrity(dataloader: torch.utils.data.DataLoader, fabric: Fabric):
#     if fabric.world_size <= 1:
#         return
#     dataset: GeantElectronDataset = dataloader.dataset
#     electron_order = torch.tensor(dataset._shuffled_elec_indices, device=fabric.device)
#     rank0_electron_order = fabric.broadcast(electron_order, 0)
#     assert torch.equal(electron_order, rank0_electron_order)

#     thisrank_indices = torch.cat([
#         torch.tensor(chunk, device=fabric.device)
#         for chunk in dataset._chunks_this_loader
#     ])
#     rank0_indices = fabric.broadcast(thisrank_indices, 0)
#     if not fabric.is_global_zero:
#         combined = torch.cat(rank0_indices, thisrank_indices)
#         assert torch.unique(combined).shape[0] == combined.shape[0]
#     fabric.barrier()


if __name__ == "__main__":
    main()
