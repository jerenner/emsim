import torch
import torch.nn.functional as F
from torch import Tensor, nn


from contextlib import ExitStack


class RelativePositionalEncodingTableInterpolate2D(nn.Module):
    def __init__(
        self,
        features: int,
        table_rows: int,
        table_columns: int,
    ):
        super().__init__()
        self.features = features
        self.table_rows = table_rows
        self.table_columns = table_columns

        param_shape = [features, table_rows, table_columns]
        self.rpe_table = nn.Parameter(torch.zeros(param_shape, dtype=torch.float))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.rpe_table)

    def forward(self, query_subpixel_positions: Tensor, key_relative_indices: Tensor):
        assert query_subpixel_positions.ndim == 2
        assert key_relative_indices.ndim == 3
        n_queries = query_subpixel_positions.shape[0]
        qk_fractional_offsets = (
            query_subpixel_positions.view(n_queries, 1, 1, 2)
            - 0.5
            + key_relative_indices.unsqueeze(0)
        )
        qk_fractional_offsets /= qk_fractional_offsets.new_tensor(
            [self.table_rows / 2, self.table_columns / 2]
        )
        with ExitStack() as stack:
            # https://github.com/pytorch/pytorch/issues/88380
            if qk_fractional_offsets.shape[0] >= 65536:
                stack.enter_context(torch.backends.cudnn.flags(enabled=False))
            out = F.grid_sample(
                self.rpe_table.unsqueeze(0).expand(n_queries, -1, -1, -1),
                qk_fractional_offsets,
                "bilinear",
                align_corners=True,
            )
        return out
