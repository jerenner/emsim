import numpy as np


def plot_ellipse(
    ax, mean, cov, n_std=1.0, facecolor="none", ellipse_color="red", **kwargs
):
    # https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
    import matplotlib.transforms as transforms
    from matplotlib.patches import Ellipse

    evals, evecs = eigsorted(cov)
    evals_sqrt = np.sqrt(evals)
    first_evec = evecs[:, 0]
    angle = np.degrees(np.arctan2(*first_evec[::-1]))
    # angle = 0
    ellipse = Ellipse(
        mean,
        width=evals_sqrt[0] * 2 * n_std,
        height=evals_sqrt[1] * 2 * n_std,
        angle=angle,
        edgecolor=ellipse_color,
        linestyle="--",
        facecolor=facecolor,
        **kwargs,
    )
    ax.add_patch(ellipse)


def eigsorted(cov):
    if cov.ndim == 2:
        squeeze_cov = True
        cov = np.expand_dims(cov, 0)
    else:
        squeeze_cov = False
    evals, evecs = np.linalg.eigh(cov)
    order = evals.argsort(-1)[:, ::-1]
    sorted_evals = np.stack([val[ord] for val, ord in zip(evals, order)])
    sorted_evecs = np.stack([vec[:, ord] for vec, ord in zip(evecs, order)])
    if squeeze_cov:
        sorted_evals, sorted_evecs = sorted_evals.squeeze(0), sorted_evecs.squeeze(0)
    return sorted_evals, sorted_evecs
