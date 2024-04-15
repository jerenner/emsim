import sys
sys.path.insert(0,'..')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sys
import timeit
import timm
import torch
import os
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from emsim.geant.dataset import (
    GeantElectronDataset,
    electron_collate_fn,
    plot_pixel_patch_and_points,
)
from emsim.geant.incidence_predictor import (
    GaussianIncidencePointPredictor,
    IncidencePointPredictor,
    eval_gaussian_model,
    eval_pointwise_model,
    fit_gaussian_patch_predictor,
    fit_pointwise_patch_predictor,
)

# To run, enter commands in the format:
#     python3 geant_localization.py -m backbone_1 ... backbone_n -n noise_std_1 ... noise_std_m

# taken from StackOverflow
def is_num(source):
    try:
        float(source)
        return True
    except ValueError:
        return False


def main(backbone_types, noise_levels):
    if torch.cuda.is_available():
        print("device is cuda")
        device = torch.device("cuda")
    else:
        print("device is cpu")
        device = "cpu"
	
    device = torch.device("cpu")

    pixels_file = "../segmentation/pixelated_1pt25um_tracks_thinned_4um_back_20k_300keV.txt"
    trajectory_file = "../B1-build/test_20k/e300keV_thinned_4um_back_20k.txt"
    for backbone_type in backbone_types:
        for noise_std in noise_levels:

            dataset = GeantElectronDataset(pixels_file, [128, 129], 7, noise_std=float(noise_std), trajectory_file=trajectory_file)
            test_dataset = GeantElectronDataset(pixels_file, [128, 129], 7, noise_std=float(noise_std), trajectory_file=trajectory_file, split="test")

            test_loader = DataLoader(test_dataset, 1, collate_fn=electron_collate_fn)
            test_batch = next(iter(test_loader))

            backbone = timm.create_model(backbone_type, in_chans=1, num_classes=0)
            model = IncidencePointPredictor(backbone).to(device)

            fit_pointwise_patch_predictor(model, dataset, n_steps=1000)

            nn_errors_pointwise, com_errors_pointwise, nn_distances_pointwise, com_distances_pointwise = eval_pointwise_model(model, test_dataset)

            # Print results
            print(f"{nn_distances_pointwise.mean()=}")
            print(f"{com_distances_pointwise.mean()=}")

            test_batch_input = test_batch["pixel_patches"].to(device)

            timing_pointwise = timeit.timeit(lambda: model(test_batch["pixel_patches"].to(device)), number=1, globals=globals())
            print(timing_pointwise)

            backbone = timm.create_model("resnet18", in_chans=1, num_classes=0)
            model2 = GaussianIncidencePointPredictor(backbone).to(device)

            nll_losses, rmse = fit_gaussian_patch_predictor(model2, dataset, n_steps=1000)

            nn_errors_gaussian, com_errors_gaussian, nn_distances_gaussian, com_distances_gaussian = eval_gaussian_model(model2, test_dataset)

            print(f"{nn_distances_gaussian.mean()=}")
            print(f"{com_distances_gaussian.mean()=}")

            timing_gaussian = timeit.timeit(lambda: model2(test_batch["pixel_patches"].to(device)), number=1, globals=globals())
            print(timing_gaussian)

            with open(f"test_output/{backbone_type}_noise_{noise_std}", "w") as f:
                f.write(f"Pointwise NN mean distances: {nn_distances_pointwise.mean()}\n")
                f.write(f"Pointwise COM mean distances: {com_distances_pointwise.mean()}\n")
                f.write(f"Pointwise Model Time: {timing_pointwise}\n")

                f.write(f"Gaussian NN mean distances: {nn_distances_gaussian.mean()}\n")
                f.write(f"Gaussian COM mean distances: {com_distances_gaussian.mean()}\n")
                f.write(f"Gaussian Model Time: {timing_gaussian}\n")


if __name__ == "__main__":

    arguments = sys.argv
    noise_levels = []
    backbone_types = []

    os.system("mkdir /global/homes/b/basch/emsim/notebooks/test_output")

    i = 2
    if arguments[1] != "-m":
        print("Model backbone name is required.")
        sys.exit(0)
    else:
        while i < len(arguments) and arguments[i] != "-n":
            backbone_types.append(arguments[2])
            i += 1

    if len(backbone_types) == 0:
        print("Must specify a backbone after flag.")
        sys.exit(0)

    if i >= len(arguments) or arguments[i] != "-n":
        print("Noise flag is required.")
        sys.exit(0)
    else:
        i += 1
        while i < len(arguments) and is_num(arguments[i]):
            noise_levels.append(arguments[i])
            i += 1

    if len(noise_levels) == 0:
        print("Specify at least one noise level after the noise flag.")
        sys.exit(0)
    elif i < len(arguments) and not is_num(arguments[i]):
        print("All noise arguments must be numbers.")
        sys.exit(0)

    main(backbone_types, noise_levels)
