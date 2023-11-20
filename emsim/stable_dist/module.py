import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchstable import Batch1DIntegrator, StableWithLogProb, quantile_loss
from torchstable.mcculloch_quantile_estimate import cosine_decay
from tqdm import trange

from emsim.io.ncemhub_dataset import collate


class StableParameters(nn.Module):
    def __init__(
        self, alpha=None, beta=None, scale=None, loc=None, integration_N_gridpoints=501
    ):
        super().__init__()
        raw_alpha, raw_beta, raw_scale, loc = self._get_initial_params(
            alpha, beta, scale, loc
        )
        self._raw_alpha = nn.Parameter(raw_alpha)
        self._raw_beta = nn.Parameter(raw_beta)
        self._raw_scale = nn.Parameter(raw_scale)
        self.loc = nn.Parameter(loc)

        self.integrator = Batch1DIntegrator()
        self.integration_N_gridpoints = integration_N_gridpoints

    @property
    def alpha(self):
        return torch.sigmoid(self._raw_alpha) * 2

    @property
    def stability(self):
        return self.alpha

    @property
    def beta(self):
        return torch.sigmoid(self._raw_beta) * 2 - 1

    @property
    def skew(self):
        return self.beta

    @property
    def scale(self):
        return torch.exp(self._raw_scale)

    def _get_initial_params(self, alpha=None, beta=None, scale=None, loc=None):
        if alpha is None:
            raw_alpha = torch.randn([])
        else:
            assert 0 <= alpha <= 2
            alpha = torch.tensor(alpha)
            raw_alpha = torch.logit(alpha / 2)
        if beta is None:
            raw_beta = torch.randn([])
        else:
            assert -1 <= beta <= 1
            beta = torch.tensor(beta)
            raw_beta = torch.logit((beta + 1) / 2)
        if scale is None:
            raw_scale = torch.randn([])
        else:
            assert scale > 0.0
            scale = torch.tensor(scale)
            raw_scale = scale.log()
        if loc is None:
            loc = torch.randn([])
        else:
            loc = torch.tensor(loc)

        return raw_alpha, raw_beta, raw_scale, loc

    def params(self):
        return torch.stack([self.alpha, self.beta, self.scale, self.loc])

    @staticmethod
    def _reduce(loss, reduction):
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss
        else:
            raise ValueError(
                f"Invalid reduction {reduction}, expected one of ('sum', 'mean', 'none')"
            )

    def nll_loss(self, data, reduction="mean"):
        dist = StableWithLogProb(
            self.alpha,
            self.beta,
            self.scale,
            self.loc,
            integrator=self.integrator,
            integration_N_gridpoints=self.integration_N_gridpoints,
        )
        nll = -dist.log_prob(data)
        return self._reduce(nll, reduction)

    def quantile_loss(self, data, reduction="mean"):
        return self._reduce(quantile_loss(data, self.params()), reduction)

    def combined_loss(self, data, quantile_weight, reduction="mean"):
        nll_loss = self.nll_loss(data, reduction)
        quantile_loss = self.quantile_loss(data, "none")
        return nll_loss + quantile_weight * quantile_loss

    @property
    def device(self):
        return self._raw_alpha.device


def fit_from_true_dist(
    true_dist,
    module,
    num_steps,
    batch_size,
    initial_quantile_weight=10,
    quantile_decay_steps=2000,
):
    optim = torch.optim.AdamW(module.parameters())
    losses = []
    gradients = []
    param_values = []
    param_errors = []
    torch.manual_seed(123)

    with trange(num_steps) as pbar:
        true_params = np.array(
            [
                true_dist.stability.item(),
                true_dist.skew.item(),
                true_dist.scale.item(),
                true_dist.loc.item(),
            ]
        )
        for i in pbar:
            data = true_dist.sample((batch_size,)).to(module.device)
            optim.zero_grad()
            loss = module.combined_loss(
                data, cosine_decay(initial_quantile_weight, quantile_decay_steps, i)
            )

            loss.backward()
            with torch.no_grad():
                grad_t = [param.grad.item() for param in module.parameters()]

            gradients.append(grad_t)

            optim.step()
            losses.append(loss.item())
            with torch.no_grad():
                params_t = np.array([param.item() for param in module.params()])
            param_values.append(params_t)

            param_error = np.linalg.norm(params_t - true_params, 1)
            param_errors.append(param_error)

            pbar.set_postfix(
                loss=losses[-1],
                param_error=param_error,
                alpha=param_values[-1][0].item(),
                beta=param_values[-1][1].item(),
            )
    per_param_errors = np.abs(np.array(param_values) - true_params)

    return param_values, per_param_errors, param_errors, losses


def fit_from_dataset(
    dataset,
    param_module,
    num_steps=6000,
    batch_size=4,
    initial_quantile_weight=10,
    quantile_decay_steps=2000,
    manual_seed=123,
):
    optim = torch.optim.SGD(param_module.parameters(), lr=1e-3)
    losses = []
    param_values = []
    torch.manual_seed(manual_seed)

    def make_loader():
        return iter(DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, pin_memory=True))

    loader = make_loader()
    with trange(num_steps) as pbar:
        for i in pbar:
            try:
                batch = next(loader)
            except StopIteration:
                loader = make_loader()
                batch = next(loader)

            energies = batch["energies"].to(param_module.device, torch.float)

            optim.zero_grad()
            if i <= quantile_decay_steps:
                loss = param_module.combined_loss(
                    energies, cosine_decay(initial_quantile_weight, quantile_decay_steps, i)
                )
            else:
                loss = param_module.nll_loss(energies)
            loss.backward()

            optim.step()
            losses.append(loss.item())
            with torch.no_grad():
                params_t = np.array([param.item() for param in param_module.params()])
            param_values.append(params_t)

            pbar.set_postfix(
                loss=losses[-1],
                alpha=param_values[-1][0].item(),
                beta=param_values[-1][1].item(),
                scale=param_values[-1][2].item(),
                loc=param_values[-1][3].item(),
            )

    return param_values, losses
