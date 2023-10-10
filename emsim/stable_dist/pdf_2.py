from typing import Optional

import torch
from torch import Tensor
from emsim.stable_dist.zeta import _zeta

from emsim.stable_dist.pdf import _c2, _g, _theta_0, gamma
from emsim.stable_dist.integrator import Batch1DIntegrator


def standard_density_2(x, alpha, beta, N=501) -> Tensor:
    zeta = _zeta(alpha, beta)
    theta_0 = _theta_0(alpha, beta)

    def density_x_eq_zeta():
        return (
            gamma(1 + 1 / alpha)
            * theta_0.cos()
            / torch.pi
            / ((1 + zeta**2) ** (1 / 2 / alpha))
        )

    def density_x_neq_zeta():
        return torch.where(
            x < zeta,
            density_x_gt_zeta(-x, alpha, -beta, -zeta, N),
            density_x_gt_zeta(x, alpha, beta, zeta, N),
        )

    return torch.where(x == zeta, density_x_eq_zeta(), density_x_neq_zeta())


def density_x_gt_zeta(
    x: Tensor, alpha: Tensor, beta: Tensor, zeta: Tensor, N: Optional[int] = 501
) -> Tensor:
    integrator = Batch1DIntegrator()
    c2 = _c2(x, alpha, beta, zeta)

    def integrand(theta: Tensor):
        g = _g(theta, x, alpha, beta, zeta)
        g[g.isinf().logical_or(g < 0.0)] = 0.0
        return g * torch.exp(-g)

    theta_0 = _theta_0(alpha, beta)
    integration_domain = torch.stack(
        [-theta_0, theta_0.new_full(theta_0.shape, torch.pi / 2)], -1
    )

    integral = integrator.integrate(
        integrand, 1, N=N, integration_domain=integration_domain
    )
    return c2 * integral
