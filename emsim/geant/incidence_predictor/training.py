import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

from ..dataset import GeantElectronDataset, electron_collate_fn


def fit_gaussian_patch_predictor(
    model: nn.Module, dataset: GeantElectronDataset, n_steps=1000, device=None
):
    if device is None:
        if hasattr(model, "device"):
            device = model.device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

    def make_loader():
        return iter(
            DataLoader(dataset, collate_fn=electron_collate_fn)
        )

    losses = []
    rmse_metrics = []
    optim = torch.optim.AdamW(model.parameters())
    iter_loader = make_loader()
    mse = torch.nn.MSELoss(reduction="none")
    with trange(n_steps) as pbar:
        for i in pbar:
            try:
                batch = next(iter_loader)
            except StopIteration:
                iter_loader = make_loader()
                batch = next(iter_loader)
            patches = batch["pixel_patches"].to(device)
            incidence_points = batch["local_incidence_points_pixels"].to(device)

            optim.zero_grad()
            predicted_distribution = model(patches)
            log_prob = predicted_distribution.log_prob(incidence_points)
            loss = -log_prob.mean()
            with torch.no_grad():
                rmse = (
                    mse(incidence_points, predicted_distribution.mean)
                    .sum(-1)
                    .sqrt()
                    .mean()
                )
            losses.append(loss.item())
            rmse_metrics.append(rmse.item())
            loss.backward()
            optim.step()

            pbar.set_postfix(loss=losses[-1], rmse=rmse_metrics[-1])

    return np.stack(losses), np.stack(rmse_metrics)


def fit_pointwise_patch_predictor(model: nn.Module, dataset: GeantElectronDataset, n_steps=1000, device=None):
    if device is None:
        if hasattr(model, "device"):
            device = model.device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

    def make_loader():
        return iter(DataLoader(dataset, collate_fn=electron_collate_fn))

    losses = []
    optim = torch.optim.AdamW(model.parameters())
    iter_loader = make_loader()
    criterion = torch.nn.HuberLoss()
    with trange(n_steps) as pbar:
        for i in pbar:
            try:
                batch = next(iter_loader)
            except StopIteration:
                iter_loader = make_loader()
                batch = next(iter_loader)
            patches = batch["pixel_patches"].to(device)
            incidence_points = batch["local_incidence_points_pixels"].to(device)

            optim.zero_grad()
            predicted_incidences = model(patches)
            loss = criterion(predicted_incidences, incidence_points)
            loss.backward()
            optim.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=losses[-1])
