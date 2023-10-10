from typing import Optional

import torch
from torch import Tensor
from .zeta import _zeta

from emsim.stable_dist.integrator import Batch1DIntegrator

from .pdf import gamma, _theta_0


def _V(
    theta: Tensor,
    alpha: Tensor,
    beta: Tensor,
    theta_0: Optional[Tensor] = None,
) -> Tensor:
    # Eq. (3.24)
    def alpha_not_1(theta: Tensor, alpha: Tensor, theta_0: Tensor, cos_alpha_) -> Tensor:

        term1 = torch.cos(alpha * theta_0) ** (1 / (alpha - 1))

        term2_base = torch.cos(theta) / torch.sin(alpha * theta_0 + alpha * theta)
        term2_base[term2_base < 0.0] = 0.0 # stop nans from appearing
        term2 = term2_base ** (alpha / (alpha - 1))

        term3 = torch.cos(alpha * theta_0 + (alpha - 1) * theta) / torch.cos(theta)

        out = term1 * term2
        out = out * term3
        # out = term1 * term2 * term3
        if out.isnan().any():
            raise ValueError
        return out

    def alpha_is_1(theta: Tensor, beta: Tensor) -> Tensor:
        term1 = 2 / torch.pi

        term2 = (torch.pi / 2 + beta * theta) / torch.cos(theta)

        term3_exponent = 1 / beta * (torch.pi / 2 + beta * theta) * torch.tan(theta)
        term3_exponent = term3_exponent.clamp_max(100)
        term3 = torch.exp(term3_exponent)

        out = term1 * term2
        out = out * term3
        # out = term1 * term2 * term3
        if out.isnan().any():
            raise ValueError
        return out

    return torch.where(
        alpha == 1.0, alpha_is_1(theta, beta), alpha_not_1(theta, alpha, theta_0)
    )


def stable_standard_density_old(
    x: Tensor,
    alpha: Tensor,
    beta: Tensor,
    integrator: Optional[Batch1DIntegrator] = None,
    integrator_N_gridpoints: Optional[int] = 201,
) -> Tensor:
    # Thm 3.3

    if x.ndim == 0:
        x = x.unsqueeze(0)
    if alpha.ndim == 0:
        alpha = alpha.unsqueeze(0)
    if beta.ndim == 0:
        beta = beta.unsqueeze(0)

    if integrator is None:
        integrator = Batch1DIntegrator()

    def alpha_not_1(x: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
        def alpha_not_1_x_is_0(alpha: Tensor, beta: Tensor) -> Tensor:
            theta_0 = _theta_0(alpha, beta)
            zeta = _zeta(alpha, beta)
            return (
                gamma(1 + 1 / alpha)
                * torch.cos(theta_0)
                / ((1 + zeta**2) ** (1 / alpha / 2))  # Eq (3.23)
                / torch.pi
            )

        def alpha_not_1_x_not_0(x: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
            def alpha_not_1_x_gt_0(x: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
                theta_0 = _theta_0(alpha, beta)
                term1_num = alpha * x ** (1 / (alpha - 1))
                term1_den = torch.pi * torch.abs(alpha - 1)
                term1 = term1_num / term1_den

                def integrand(theta: Tensor) -> Tensor:
                    V = _V(theta, alpha, beta, theta_0)
                    return V * torch.exp(-(x ** (alpha / (alpha - 1))) * V)

                integration_domain = torch.stack(
                    [-theta_0, theta_0.new_full(theta_0.shape, torch.pi / 2)], -1
                )

                integral = integrator.integrate(
                    integrand,
                    1,
                    integrator_N_gridpoints,
                    integration_domain=integration_domain,
                )
                return term1 * integral

            return torch.where(
                x < 0.0,
                alpha_not_1_x_gt_0(-x, alpha, -beta),
                alpha_not_1_x_gt_0(x, alpha, beta),
            )

        return torch.where(
            x == 0.0,
            alpha_not_1_x_is_0(alpha, beta),
            alpha_not_1_x_not_0(x, alpha, beta),
        )

    def alpha_is_1(x: Tensor, beta: Tensor) -> Tensor:
        def beta_is_0(x: Tensor) -> Tensor:
            return 1 / (torch.pi * (1 + x**2))

        def beta_not_0(x: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
            pi_over_2 = x.new_full(x.shape, torch.pi / 2)
            term1 = torch.exp(-(torch.pi * x) / (2 * beta)) / (2 * torch.abs(beta))

            def integrand(theta: Tensor) -> Tensor:
                V = _V(theta, alpha, beta)
                first_exp = torch.exp(-torch.pi * x / (2 * beta))
                second_exp = torch.exp(-first_exp * V)
                out = V * second_exp
                out[torch.logical_and(V.isinf(), second_exp == 0)] = 0.0
                # assert out.isnan().logical_not().all()
                out[out.isnan()] = 0.0
                return out

            integration_domain = torch.stack([-pi_over_2, pi_over_2], -1)

            integral = integrator.integrate(
                integrand,
                1,
                integrator_N_gridpoints,
                integration_domain=integration_domain,
            )

            return term1 * integral

        return torch.where(beta == 0.0, beta_is_0(x), beta_not_0(x, alpha, beta))

    return torch.where(alpha == 1.0, alpha_is_1(x, beta), alpha_not_1(x, alpha, beta))
