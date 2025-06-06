import torch
from torch import Tensor, nn

from emsim.utils.sparse_utils.batching import split_batch_concatted_tensor
from emsim.config.denoising import DenoisingConfig


class DenoisingGenerator(nn.Module):
    def __init__(self, config: DenoisingConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.max_denoising_group_size = config.max_electrons_per_image
        self.max_total_denoising_queries = config.max_total_denoising_queries
        self.position_noise_scale = config.position_noise_variance
        self.dn_query_embedding = nn.Embedding(
            (
                config.max_electrons_per_image
                if config.pos_neg_queries_share_embedding
                else config.max_electrons_per_image * 2
            ),
            config.embed_dim,
        )
        self.pos_neg_queries_share_embedding = config.pos_neg_queries_share_embedding

    def forward(self, batch_dict: dict[str, Tensor]):
        true_positions = batch_dict["incidence_points_pixels_rc"]
        image_size = batch_dict["image_size_pixels_rc"]
        electrons_per_image = batch_dict["batch_size"]
        batch_offsets = batch_dict["electron_batch_offsets"]

        max_electrons_per_image = electrons_per_image.max()
        assert max_electrons_per_image > 0, "Zero case unsupported"

        n_denoising_groups = max(
            self.max_total_denoising_queries // sum(electrons_per_image) // 2, 1
        )

        noised_positions = self.make_pos_neg_noised_positions(
            true_positions, n_denoising_groups, image_size, electrons_per_image
        )
        assert noised_positions.ndim == 4

        ext_batch_offsets = torch.cat(
            [batch_offsets, batch_offsets.new_tensor([len(true_positions)])]
        )
        denoising_queries = self.dn_query_embedding.weight.new_zeros(
            [*noised_positions.shape[:3], self.dn_query_embedding.weight.shape[1]]
        )
        for i in range(len(electrons_per_image)):
            n_electrons = electrons_per_image[i]
            posneg_mult = 1 if self.pos_neg_queries_share_embedding else 2
            assert (
                n_electrons * posneg_mult <= self.dn_query_embedding.num_embeddings
            ), "Not enough denoising embeddings"
            query_indices = torch.randperm(
                n_electrons * posneg_mult, device=denoising_queries.device
            )
            dn_queries_this_image: Tensor = self.dn_query_embedding(query_indices)
            dn_queries_this_image = dn_queries_this_image.reshape(
                n_electrons, 1, posneg_mult, -1
            )
            dn_queries_this_image = dn_queries_this_image.expand(
                -1, n_denoising_groups, 2, -1
            )

            start = ext_batch_offsets[i]
            end = ext_batch_offsets[i + 1]
            denoising_queries[start:end] = dn_queries_this_image

        return denoising_queries, noised_positions

    def make_pos_neg_noised_positions(
        self,
        true_positions_pixels_ij: Tensor,
        denoising_groups: int,
        image_size_rc: Tensor,
        electrons_per_image: Tensor,
    ):
        true_positions_pixels_ij = true_positions_pixels_ij.unsqueeze(1).expand(
            -1, denoising_groups, -1
        )
        position_noise = (
            torch.randn_like(true_positions_pixels_ij) * self.position_noise_scale
        )

        pos_points = true_positions_pixels_ij + position_noise
        neg_points = true_positions_pixels_ij + position_noise * (
            torch.rand_like(true_positions_pixels_ij) + 1.0
        )

        image_size_per_point = torch.cat(
            [
                size.expand(n_elecs, -1)
                for size, n_elecs in zip(image_size_rc, electrons_per_image)
            ],
            0,
        ).flip(-1)
        image_size_per_point = image_size_per_point.unsqueeze(1)

        # (electron x denoising group x 2)
        pos_points_normalized_xy = pos_points.flip(-1).double() / image_size_per_point
        neg_points_normalized_xy = neg_points.flip(-1).double() / image_size_per_point

        # (electron x denoising group x pos/neg x 2)
        positions = torch.stack([pos_points_normalized_xy, neg_points_normalized_xy], 2)
        positions = positions.clamp(0.0, 1.0)
        return positions

    @staticmethod
    def stack_main_and_denoising_queries(
        main_queries: Tensor,
        main_reference_points: Tensor,
        query_batch_offsets: Tensor,
        denoising_queries: Tensor,
        denoising_reference_points: Tensor,
        electron_batch_offsets: Tensor,
        n_attn_heads: int,
        mask_main_queries_from_denoising: bool = False,
    ):
        device = main_queries.device
        assert denoising_queries.ndim == 4  # (electron, dn group, pos/neg, feature)
        assert denoising_reference_points.ndim == 4
        n_electrons = torch.tensor(denoising_queries.shape[0], device=device)
        n_dn_groups = torch.tensor(denoising_queries.shape[1], device=device)
        main_queries_per_image = split_batch_concatted_tensor(
            main_queries, query_batch_offsets
        )
        n_main_queries_per_image = torch.tensor(
            [q.shape[0] for q in main_queries_per_image],
            device=device,
        )
        denoising_queries_per_image = split_batch_concatted_tensor(
            denoising_queries, electron_batch_offsets
        )
        main_reference_points_per_image = split_batch_concatted_tensor(
            main_reference_points, query_batch_offsets
        )
        denoising_reference_points_per_image = split_batch_concatted_tensor(
            denoising_reference_points, electron_batch_offsets
        )
        n_electrons_per_image = [q.shape[0] for q in denoising_queries_per_image]

        denoising_positive_mask = torch.zeros(
            denoising_queries.shape[:-1],
            dtype=torch.bool,
            device=device,
        )
        denoising_positive_mask[..., 0] = True

        # flatten denoising group dim and pos/neg dim
        denoising_queries_per_image = [
            queries.view(-1, queries.shape[-1])
            for queries in denoising_queries_per_image
        ]
        denoising_reference_points_per_image = [
            refpoints.view(-1, refpoints.shape[-1])
            for refpoints in denoising_reference_points_per_image
        ]

        # concatenate the main and denoising queries
        stacked_queries_per_image = [
            torch.cat([main, denoising], 0)
            for main, denoising in zip(
                main_queries_per_image, denoising_queries_per_image
            )
        ]
        stacked_refpoints_per_image = [
            torch.cat([main, denoising], 0)
            for main, denoising in zip(
                main_reference_points_per_image, denoising_reference_points_per_image
            )
        ]
        main_mask_per_image = [
            torch.cat(
                [
                    torch.ones(main.shape[:-1], device=device, dtype=torch.bool),
                    torch.zeros(denoising.shape[:-1], device=device, dtype=torch.bool),
                ],
                0,
            )
            for main, denoising in zip(
                main_queries_per_image, denoising_queries_per_image
            )
        ]

        stacked_queries = torch.cat(stacked_queries_per_image)
        stacked_refpoints = torch.cat(stacked_refpoints_per_image)
        stacked_batch_offsets = torch.cumsum(
            torch.tensor(
                [0, *[queries.shape[0] for queries in stacked_queries_per_image[:-1]]],
                device=device,
            ),
            0,
        )

        assert stacked_queries.shape[:-1] == stacked_refpoints.shape[:-1]
        total_main_queries = sum(n_main_queries_per_image)
        total_denoising_queries = sum(
            [q * n_dn_groups * 2 for q in n_electrons_per_image]
        )
        assert stacked_queries.shape[0] == total_main_queries + total_denoising_queries

        attn_mask = DenoisingGenerator.make_attn_mask(
            main_queries,
            query_batch_offsets,
            denoising_queries,
            electron_batch_offsets,
            n_dn_groups,
            n_attn_heads,
            mask_main_queries_from_denoising,
        )

        # make matched indices for loss calculation
        denoising_matched_indices = []
        dn_pos_mask_by_image = split_batch_concatted_tensor(
            denoising_positive_mask, electron_batch_offsets
        )
        for n_elecs, dn_positives in zip(n_electrons_per_image, dn_pos_mask_by_image):
            electron_indices = torch.arange(n_elecs, device=device).unsqueeze(1)
            electron_indices = electron_indices.expand(-1, n_dn_groups).flatten()

            query_indices = torch.arange(n_elecs * n_dn_groups * 2, device=device)
            query_indices = query_indices.view(n_elecs, n_dn_groups, 2)
            query_indices = query_indices[dn_positives]

            denoising_matched_indices.append(
                torch.stack([query_indices, electron_indices])
            )

        dn_batch_mask_dict = {
            "main_query_masks": main_mask_per_image,
            "denoising_positive_mask": denoising_positive_mask,
            "stacked_batch_offsets": stacked_batch_offsets,
            "n_main_queries_per_image": n_main_queries_per_image,
            "n_electrons_per_image": n_electrons_per_image,
            "n_denoising_groups": n_dn_groups,
            "n_total_gt_electrons": n_electrons,
            "electron_batch_offsets": electron_batch_offsets,
            "denoising_matched_indices": denoising_matched_indices,
        }
        return stacked_queries, stacked_refpoints, attn_mask, dn_batch_mask_dict

    @staticmethod
    def unstack_main_and_denoising_tensor(
        stacked_tensor: Tensor,
        dn_batch_mask_dict: dict,
    ):
        main_query_masks = dn_batch_mask_dict["main_query_masks"]
        stacked_batch_offsets = dn_batch_mask_dict["stacked_batch_offsets"]
        n_denoising_groups = dn_batch_mask_dict["n_denoising_groups"]
        n_total_gt_electrons = dn_batch_mask_dict["n_total_gt_electrons"]

        if stacked_tensor.ndim == 2:  # query x feature
            dummy_layer_dim = True
            stacked_tensor = stacked_tensor.unsqueeze(0)
        else:
            assert stacked_tensor.ndim in [3, 4]  # decoder layer x query x feature
            dummy_layer_dim = False

        batch_split_tensor = torch.tensor_split(
            stacked_tensor,
            stacked_batch_offsets[1:].cpu(),
            dim=1,
        )

        main_parts = [
            queries[:, mask]
            for queries, mask in zip(batch_split_tensor, main_query_masks)
        ]
        denoising_parts = [
            queries[:, mask.logical_not()]
            for queries, mask in zip(batch_split_tensor, main_query_masks)
        ]

        catted_main_parts = torch.cat(main_parts, dim=1)
        catted_denoising_parts = torch.cat(denoising_parts, dim=1)

        reshaped_denoising_parts = catted_denoising_parts.view(
            catted_denoising_parts.shape[0],
            n_total_gt_electrons,
            n_denoising_groups,
            2,
            *catted_denoising_parts.shape[2:],
        )
        if dummy_layer_dim:
            catted_main_parts = catted_main_parts.squeeze(0)
            reshaped_denoising_parts = reshaped_denoising_parts.squeeze(0)

        return catted_main_parts, reshaped_denoising_parts

    @torch.no_grad()
    @staticmethod
    def make_attn_mask(
        main_queries: Tensor,
        query_batch_offsets: Tensor,
        denoising_queries: Tensor,
        electron_batch_offsets: Tensor,
        n_denoising_groups: int,
        n_attn_heads: int,
        mask_main_queries_from_denoising: bool = False,
    ):
        assert denoising_queries.ndim == 4  # electron, dn group, pos/neg, 2
        batch_size = len(query_batch_offsets)

        def _queries_per_image(batch_offsets: Tensor, total_queries: int):
            ext_batch_offsets = torch.cat(
                [batch_offsets, batch_offsets.new_tensor([total_queries])]
            )
            return ext_batch_offsets[1:] - ext_batch_offsets[:-1]

        n_main_queries_per_image = _queries_per_image(
            query_batch_offsets, main_queries.shape[0]
        )
        electron_batch_offsets = electron_batch_offsets.to(query_batch_offsets)
        n_denoising_queries_per_image = (
            _queries_per_image(electron_batch_offsets, denoising_queries.shape[0])
            * n_denoising_groups
            * 2
        )
        n_total_queries_per_image = (
            n_main_queries_per_image + n_denoising_queries_per_image
        )

        padded_seq_length = max(n_total_queries_per_image)
        attn_masks = []
        for i, (n_main, n_denoising) in enumerate(
            zip(n_main_queries_per_image, n_denoising_queries_per_image)
        ):
            mask = torch.zeros(
                [padded_seq_length, padded_seq_length],
                dtype=torch.bool,
                device=main_queries.device,
            )
            # mask out denoising queries from main queries
            mask[:n_main, n_main:] = True

            if mask_main_queries_from_denoising:
                mask[n_main:, :n_main] = True

            assert n_denoising % n_denoising_groups == 0
            dn_group_size = n_denoising // n_denoising_groups
            for i in range(n_denoising_groups):
                dn_start_row = dn_start_col = n_main + dn_group_size * i
                dn_end_row = dn_end_col = n_main + dn_group_size * (i + 1)
                assert dn_end_col <= padded_seq_length
                # Mask this group out from the other dn groups
                mask[dn_start_row:dn_end_row, n_main:dn_start_col] = True
                mask[dn_start_row:dn_end_row, dn_end_col:] = True

            # mask out the padding queries just for completeness
            mask[n_main + n_denoising :] = True

            attn_masks.append(mask)

        attn_mask = torch.stack(attn_masks)
        attn_mask = attn_mask.unsqueeze(1).expand(-1, n_attn_heads, -1, -1)
        attn_mask = attn_mask.reshape(
            batch_size * n_attn_heads, padded_seq_length, padded_seq_length
        )

        return attn_mask
