import torch
from torch import Tensor, nn

from emsim.utils.sparse_utils.batching.batch_utils import (
    split_batch_concatted_tensor,
    normalize_batch_offsets,
    batch_offsets_to_seq_lengths,
    seq_lengths_to_batch_offsets,
)
from emsim.config.denoising import DenoisingConfig


class DenoisingGenerator(nn.Module):
    def __init__(self, config: DenoisingConfig):
        super().__init__()
        self.config = config
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

    def forward(
        self, batch_dict: dict[str, Tensor]
    ) -> tuple[list[Tensor], list[Tensor]]:
        true_positions = batch_dict["incidence_points_pixels_rc"]
        image_size = batch_dict["image_size_pixels_rc"]
        objects_per_image = batch_dict["batch_size"]
        batch_offsets = batch_dict["electron_batch_offsets"]
        n_images = objects_per_image.size(0)
        spatial_dim = image_size.size(-1)
        device = true_positions.device

        max_electrons_per_image = int(objects_per_image.max())
        n_total_objects = int(objects_per_image.sum())
        assert max_electrons_per_image > 0, "Zero case unsupported"

        # Select 2x as many queries if positive and negative use different embeddings
        posneg_mult = 1 if self.pos_neg_queries_share_embedding else 2
        n_dn_embeddings = self.dn_query_embedding.weight.size(0)
        needed_queries = posneg_mult * max_electrons_per_image
        assert (
            needed_queries <= n_dn_embeddings
        ), f"Not enough denoising queries ({n_dn_embeddings}, need {needed_queries})"

        n_denoising_groups = max(
            int(self.max_total_denoising_queries // n_total_objects // 2), 1
        )

        true_positions_split = split_batch_concatted_tensor(
            true_positions, batch_offsets
        )

        per_image_noised_positions = self.make_pos_neg_noised_positions(
            true_positions_split, n_denoising_groups, image_size
        )
        # (dn group, object, pos/neg, spatial dim)
        assert all(
            pos.shape
            == (n_denoising_groups, objects_per_image[b].item(), 2, spatial_dim)
            for b, pos in enumerate(per_image_noised_positions)
        )

        batch_offsets = normalize_batch_offsets(batch_offsets, n_total_objects)

        # Extract denoising queries
        # Random query permutation to ensure all get trained
        query_permutation = torch.randperm(
            max_electrons_per_image * posneg_mult, device=device
        )
        selected_queries = self.dn_query_embedding.weight.index_select(
            0, query_permutation
        )
        per_image_queries = []
        for b in range(n_images):
            n_objects_b = int(objects_per_image[b])
            queries_b = selected_queries[: (n_objects_b * posneg_mult)]
            queries_b = (
                queries_b.view(1, n_objects_b, posneg_mult, self.embed_dim)
                .expand(n_denoising_groups, -1, 2, -1)
                .contiguous()
            )
            per_image_queries.append(queries_b)

        return per_image_queries, per_image_noised_positions

    def make_pos_neg_noised_positions(
        self,
        true_positions_pixels_ij: list[Tensor],
        denoising_groups: int,
        image_size_rc: Tensor,
    ) -> list[Tensor]:
        per_image_noised_positions = []
        n_images = len(true_positions_pixels_ij)
        for b in range(n_images):
            true_pts_b = true_positions_pixels_ij[b]
            expanded_pts = true_pts_b.unsqueeze(0).expand(denoising_groups, -1, -1)

            position_noise = torch.randn_like(expanded_pts) * self.position_noise_scale

            pos_points = expanded_pts + position_noise
            neg_1to2x = torch.rand_like(position_noise) + 1.0
            neg_points = expanded_pts + position_noise * neg_1to2x

            noised_positions_b = torch.stack([pos_points, neg_points], 2)
            noised_positions_b.clamp_(
                noised_positions_b.new_zeros([]),
                image_size_rc[b].expand_as(noised_positions_b),
            )

            per_image_noised_positions.append(noised_positions_b)

        return per_image_noised_positions

    @staticmethod
    def _per_image_to_per_object(tensor: Tensor, objects_per_image: Tensor):
        assert tensor.size(0) == objects_per_image.size(0)
        # figure out how many images and objects
        n_images = objects_per_image.size(0)
        batch_offsets = seq_lengths_to_batch_offsets(objects_per_image)

        out_shape = list(tensor.shape)
        out_shape[0] = int(batch_offsets[-1].item())
        out = tensor.new_empty(out_shape)

        for i in range(n_images):
            start = int(batch_offsets[i])
            end = int(batch_offsets[i + 1])
            out[start:end] = tensor[i]

        return out

    @staticmethod
    def stack_main_and_denoising_queries(
        main_queries: Tensor,
        main_reference_points: Tensor,
        query_batch_offsets: Tensor,
        denoising_queries: list[Tensor],
        denoising_reference_points: list[Tensor],
        object_batch_offsets: Tensor,
        n_attn_heads: int,
        mask_main_queries_from_denoising: bool = False,
    ):
        device = main_queries.device
        n_images = len(denoising_queries)

        # (dn group, object, pos/neg, feature)
        assert all(q.ndim == 4 for q in denoising_queries)
        assert all(refpt.ndim == 4 for refpt in denoising_reference_points)
        assert query_batch_offsets.shape == object_batch_offsets.shape

        # index bookkeeping
        n_objects_per_image = [dn.size(1) for dn in denoising_queries]
        n_dn_groups = denoising_queries[0].size(0)
        n_dn_queries_per_image = [
            n_obj * n_dn_groups * 2 for n_obj in n_objects_per_image
        ]

        n_main_queries_per_image = batch_offsets_to_seq_lengths(
            query_batch_offsets
        ).tolist()

        n_total_main_queries, embed_dim = main_queries.shape
        assert n_total_main_queries == sum(n_main_queries_per_image)
        position_dim = main_reference_points.size(1)

        n_stacked_queries_per_image = [
            main + dn
            for main, dn in zip(n_main_queries_per_image, n_dn_queries_per_image)
        ]

        n_total_dn_queries = sum(n_dn_queries_per_image)
        n_total_stacked_queries = sum(n_stacked_queries_per_image)
        assert n_total_stacked_queries == n_total_main_queries + n_total_dn_queries

        # Create tensors that will be filled in
        output_stacked_queries = main_queries.new_empty(
            n_total_stacked_queries, embed_dim
        )
        output_stacked_refpoints = main_reference_points.new_empty(
            n_total_stacked_queries, position_dim
        )
        stacked_batch_offsets = torch.zeros_like(query_batch_offsets)
        # Denoising queries are in the order (dn group, object, pos/neg).flatten()

        # Create attn mask tensor in stacked and padded instead of batch-concat format
        max_stacked_batch = max(n_stacked_queries_per_image)
        stacked_attn_mask = torch.zeros(
            (n_images, max_stacked_batch, max_stacked_batch),
            device=device,
            dtype=torch.bool,
        )
        dn_matched_indices = []  # for loss calculation

        # fill in output tensor image by image
        output_cursor = 0
        for b in range(n_images):
            ### bookkeeping ###
            n_main_queries_b = int(n_main_queries_per_image[b])
            n_objects_b = int(n_objects_per_image[b])
            n_dn_b = n_objects_b * n_dn_groups * 2

            ### main queries and reference points ###
            # start and end position of this image in input tensor
            main_in_start = int(query_batch_offsets[b])
            main_in_end = int(query_batch_offsets[b + 1])

            # extract main queries and refpoints
            main_queries_b = main_queries[main_in_start:main_in_end]
            main_refpoints_b = main_reference_points[main_in_start:main_in_end]

            # start and end position of this image's main queries in new output tensor
            main_out_start = output_cursor
            main_out_end = output_cursor + n_main_queries_b

            # copy over the embeddings and refpoints
            output_stacked_queries[main_out_start:main_out_end] = main_queries_b
            output_stacked_refpoints[main_out_start:main_out_end] = main_refpoints_b

            ### denoising queries and reference points ###
            # extract denoising embedding and refpoints
            dn_queries_b = denoising_queries[b]
            dn_refpoints_b = denoising_reference_points[b]

            assert dn_queries_b.shape == (n_dn_groups, n_objects_b, 2, embed_dim)
            assert dn_refpoints_b.shape == (n_dn_groups, n_objects_b, 2, position_dim)

            # get start and end position in output tensor
            dn_out_start = main_out_end
            dn_out_end = dn_out_start + n_dn_b

            # copy over denoising values, flattening dn group, object, pos/neg dims
            output_stacked_queries[dn_out_start:dn_out_end] = dn_queries_b.view(
                n_dn_b, embed_dim
            )
            output_stacked_refpoints[dn_out_start:dn_out_end] = dn_refpoints_b.view(
                n_dn_b, position_dim
            )

            ### attention mask to separate denoising groups ###
            submask_b = stacked_attn_mask[b]
            # mask out denoising queries from main queries
            submask_b[:n_main_queries_b, n_main_queries_b:] = True

            if mask_main_queries_from_denoising:
                submask_b[n_main_queries_b:, :n_main_queries_b] = True

            dn_group_size = n_objects_b * 2
            assert dn_group_size == n_objects_per_image[b] * 2
            for i in range(n_dn_groups):
                dn_start_row = dn_start_col = n_main_queries_b + dn_group_size * i
                dn_end_row = dn_end_col = n_main_queries_b + dn_group_size * (i + 1)
                assert dn_end_col <= submask_b.size(1)
                # Mask this group out from the other dn groups
                submask_b[dn_start_row:dn_end_row, n_main_queries_b:dn_start_col] = True
                submask_b[dn_start_row:dn_end_row, dn_end_col:] = True

            # mask out the padding queries just for completeness
            submask_b[n_main_queries_b + n_dn_b :] = True

            assert torch.equal(stacked_attn_mask[b], submask_b)  # sanity check

            ### matching indices for loss calculation ###
            # prepare matching indices
            object_indices = (
                torch.arange(n_objects_b, device=device)
                .unsqueeze(0)
                .expand(n_dn_groups, n_objects_b)
                .flatten()
            )
            query_indices = torch.arange(n_dn_b, device=device)
            query_indices = query_indices.view(n_objects_b, n_dn_groups, 2)
            query_indices = query_indices[:, :, 0].flatten()  # select only positives

            dn_matched_indices.append(torch.stack([query_indices, object_indices]))

            ### update bookkeeping ###
            output_cursor = dn_out_end
            stacked_batch_offsets[b + 1] = dn_out_end

        assert dn_out_end == output_stacked_queries.size(0) == n_total_stacked_queries

        dn_batch_mask_dict = {
            "stacked_batch_offsets": stacked_batch_offsets,
            "n_main_queries_per_image": n_main_queries_per_image,
            "n_objects_per_image": n_objects_per_image,
            "n_denoising_groups": n_dn_groups,
            "electron_batch_offsets": object_batch_offsets,
            "denoising_matched_indices": dn_matched_indices,
        }
        return (
            output_stacked_queries,
            output_stacked_refpoints,
            stacked_attn_mask,
            dn_batch_mask_dict,
        )

    @staticmethod
    def unstack_main_and_denoising_tensor(
        stacked_tensor: Tensor,
        dn_batch_mask_dict: dict,
    ):
        stacked_batch_offsets = dn_batch_mask_dict["stacked_batch_offsets"]
        n_denoising_groups = dn_batch_mask_dict["n_denoising_groups"]
        n_objects_per_image = dn_batch_mask_dict["n_objects_per_image"]
        n_main_queries_per_image = dn_batch_mask_dict["n_main_queries_per_image"]

        if stacked_tensor.ndim == 2:  # query x feature
            dummy_layer_dim = True
            stacked_tensor = stacked_tensor.unsqueeze(0)
        else:
            assert stacked_tensor.ndim in [3, 4]  # decoder layer x query x feature
            dummy_layer_dim = False

        batch_split_tensor = torch.tensor_split(
            stacked_tensor,
            stacked_batch_offsets[1:-1].cpu(),
            dim=1,
        )
        batch_split_tensor = [t for t in batch_split_tensor if t.numel() > 0]

        main_tensors = []
        denoising_tensors = []
        for b, tensor_b in enumerate(batch_split_tensor):
            main_b, dn_b = torch.tensor_split(
                tensor_b, [n_main_queries_per_image[b]], dim=1
            )
            assert dn_b.size(1) == n_objects_per_image[b] * n_denoising_groups * 2
            main_tensors.append(main_b)
            denoising_tensors.append(dn_b)

        main_tensor = torch.cat(main_tensors, dim=1)
        dn_tensor = torch.cat(denoising_tensors, dim=1)

        # reshape dn tensor to add back dn group, object, and pos/neg dims
        dn_tensor = dn_tensor.view(
            (dn_tensor.size(0), n_denoising_groups, sum(n_objects_per_image), 2)
            + dn_tensor.shape[2:]
        )

        if dummy_layer_dim:
            main_tensor = main_tensor.squeeze(0)
            dn_tensor = dn_tensor.squeeze(0)

        return main_tensor, dn_tensor
