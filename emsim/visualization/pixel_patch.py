from typing import Optional

import numpy as np

from .utils import plot_ellipse


def plot_pixel_patch_and_points(
    pixel_patch: np.ndarray,
    points: list[np.ndarray],
    point_labels: Optional[list[str]] = None,
    ellipse_mean: Optional[np.ndarray] = None,
    ellipse_cov: Optional[np.ndarray] = None,
    ellipse_n_stds: list[int] = [1.0, 2.0, 3.0],
    ellipse_facecolor: str = "none",
    ellipse_colors: list[str] = ["blue", "purple", "red"],
    ellipse_kwargs: Optional[dict] = None,
):
    import matplotlib.pyplot as plt

    ellipse_kwargs = ellipse_kwargs or {}

    pixel_patch = pixel_patch.squeeze()
    fig, ax = plt.subplots()
    map = ax.imshow(
        pixel_patch.T,
        extent=(0, pixel_patch.shape[0], pixel_patch.shape[1], 0),
        interpolation=None,
    )
    fig.colorbar(map, ax=ax, label="Charge")
    ax.set_xlabel("Pixel")
    ax.set_ylabel("Pixel")
    for point in points:
        ax.scatter(*point)
    if point_labels is not None:
        ax.legend(point_labels)
    if ellipse_mean is not None and ellipse_cov is not None:
        for n_std, color in zip(ellipse_n_stds, ellipse_colors):
            plot_ellipse(
                ax,
                ellipse_mean,
                ellipse_cov,
                n_std,
                ellipse_facecolor,
                ellipse_color=color,
                **ellipse_kwargs,
            )
    return fig, ax
