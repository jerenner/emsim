import os
from typing import Union
import hydra
import yaml
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import trange
import time
import datetime
import logging

import torch
from torch import Tensor, nn

import lightning as L
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from lightning.fabric.loggers import TensorBoardLogger, CSVLogger
from torchmetrics.aggregation import RunningMean

from transformers import get_cosine_schedule_with_warmup

from emsim.networks import (
    EMModel,
    SparseResnetUnet,
    EMTransformer,
    ValueEncoder,
    ElectronSalienceCriterion,
    EMCriterion,
)
from emsim.utils.misc_utils import _get_layer
from emsim.utils.sparse_utils import unpack_sparse_tensors
from emsim.geant.dataset import (
    GeantElectronDataset,
    make_test_train_datasets,
    electron_collate_fn,
)
from emsim.preprocessing import NSigmaSparsifyTransform


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):
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

    train_dataset, eval_dataset = make_test_train_datasets(
        pixels_file=os.path.join(cfg.dataset.directory, cfg.dataset.pixels_file),
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
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        cfg.batch_size,
        collate_fn=electron_collate_fn,
        pin_memory=True,
        num_workers=1,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        cfg.batch_size,
        collate_fn=electron_collate_fn,
        pin_memory=True,
        num_workers=1,
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
    ## main training loop
    iter_time = RunningMean(cfg.print_interval).to(fabric.device)
    model.train()
    iter_loader = iter(train_dataloader)
    if fabric.is_global_zero:
        _logger.info("Begin training")
    epoch = 0
    training_start_time = time.time()
    for i in range(start_iter, cfg.num_steps):
        t0 = time.time()
        try:
            batch = next(iter_loader)
        except StopIteration:
            eval(epoch, i, model, eval_dataloader, fabric)
            model.train()
            epoch += 1
            iter_loader = iter(train_dataloader)
            batch = next(iter_loader)

        batch = unpack_sparse_tensors(batch)
        loss_dict, _ = model(batch)
        total_loss = loss_dict["loss"]
        fabric.backward(total_loss)
        fabric.clip_gradients(model, optimizer, cfg.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        with torch.no_grad():
            log_dict = fabric.all_reduce(loss_dict, reduce_op="mean")
        log_dict["lr"] = lr_scheduler.get_last_lr()[0]

        if i > 0 and i % cfg.print_interval == 0:
            metric_log_dict = model.criterion.get_logs()
            log_str = model.criterion.make_log_str(metric_log_dict)
            elapsed_time = time.time() - training_start_time
            elapsed_time_str = _elapsed_time_str(elapsed_time)

            log_str = f"Iter {i} (Epoch {epoch}) -- ({elapsed_time_str}) -- " + log_str
            log_str = log_str + f" iter_time: {iter_time.compute()}\n"
            if fabric.is_global_zero:
                _logger.info(log_str)
            log_dict.update(metric_log_dict)

        log_dict = model.criterion.tensorboardify_keys(log_dict)
        fabric.log_dict(log_dict, step=i)

        if i > 0 and i % cfg.eval_steps == 0:
            save(save_dir, f"step_{i}", model, optimizer, fabric, i)
            eval(epoch, i, model, eval_dataloader, fabric)
            model.train()

        iter_time.update(time.time() - t0)
    if fabric.is_global_zero:
        elapsed_time_str = _elapsed_time_str(elapsed_time)
        _logger.info(f"Training complete in {elapsed_time_str}.")


@torch.no_grad()
def eval(
    epoch: int,
    step: int,
    model: nn.Module,
    eval_loader: torch.utils.data.DataLoader,
    fabric: Fabric,
):
    return
    logger = logging.getLogger(__name__)
    model.eval()
    incidence_error = []
    query_classification = []
    dice_score = []
    for batch in eval_loader:
        batch = unpack_sparse_tensors(batch)
        output = model(batch)


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


def __debug_find_unused_parameters(model):
    unused_parameters = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused_parameters.append(name)
    if len(unused_parameters) > 0:
        _logger.debug(f"Unused parameters: {unused_parameters}")


if __name__ == "__main__":
    main()
