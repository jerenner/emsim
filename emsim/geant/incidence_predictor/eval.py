import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from emsim.geant.dataset import GeantElectronDataset, electron_collate_fn


def eval_pointwise_model(model: nn.Module, dataset: GeantElectronDataset, device=None):
    if device is None:
        if hasattr(model, "device"):
            device = model.device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

    nn_errors = []
    com_errors = []
    loader = DataLoader(dataset, collate_fn=electron_collate_fn)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            patches = batch["pixel_patches"].to(device)
            true_incidence_points = batch["local_incidence_points_pixels"].to(device)
            com_estimates = batch["local_centers_of_mass_pixels"].to(device)
            nn_estimates = model(patches)
            batch_nn_error = nn_estimates - true_incidence_points
            batch_com_error = com_estimates - true_incidence_points

            nn_errors.append(batch_nn_error.cpu().numpy())
            com_errors.append(batch_com_error.cpu().numpy())

    nn_errors = np.concatenate(nn_errors) * np.array(dataset.grid.pixel_size_um)
    com_errors = np.concatenate(com_errors) * np.array(dataset.grid.pixel_size_um)

    nn_distances = np.sqrt(np.sum(np.square(nn_errors), -1))
    com_distances = np.sqrt(np.sum(np.square(com_errors), -1))

    return nn_errors, com_errors, nn_distances, com_distances


def eval_gaussian_model(model: nn.Module, dataset: GeantElectronDataset, device=None):
    if device is None:
        if hasattr(model, "device"):
            device = model.device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

    nn_errors = []
    com_errors = []
    loader = DataLoader(dataset, collate_fn=electron_collate_fn)

    with torch.no_grad():
        for batch in loader:
            patches = batch["pixel_patches"].to(device)
            true_incidence_points = batch["local_incidence_points_pixels"].to(device)
            com_estimates = batch["local_centers_of_mass_pixels"].to(device)
            nn_estimates = model(patches)
            batch_nn_error = nn_estimates.mean - true_incidence_points
            batch_com_error = com_estimates - true_incidence_points

            nn_errors.append(batch_nn_error.cpu().numpy())
            com_errors.append(batch_com_error.cpu().numpy())

    nn_errors = np.concatenate(nn_errors) * np.array(dataset.grid.pixel_size_um)
    com_errors = np.concatenate(com_errors) * np.array(dataset.grid.pixel_size_um)

    nn_distances = np.sqrt(np.sum(np.square(nn_errors), -1))
    com_distances = np.sqrt(np.sum(np.square(com_errors), -1))

    return nn_errors, com_errors, nn_distances, com_distances
