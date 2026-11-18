# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for dual-projector backbone joiner routing and per-level mask construction."""

from __future__ import annotations

from argparse import Namespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from rfdetr.models.backbone import Joiner
from rfdetr.models.backbone.backbone import Backbone
from rfdetr.utilities.tensors import NestedTensor, nested_tensor_from_tensor_list


class _FakeBackbone(nn.Module):
    """Backbone shim used to validate Joiner contract changes.

    Examples:
        >>> features = [_feature((8, 4, 4), batch_size=1)]
        >>> backbone = _FakeBackbone(features, None)
        >>> len(backbone(torch.ones(1, 3, 8, 8))[0])
        1
    """

    def __init__(
        self,
        features: list[NestedTensor],
        cross_attention_features: list[object] | None,
    ) -> None:
        super().__init__()
        self._features = features
        self._cross_attention_features = cross_attention_features

    def forward(self, tensor: torch.Tensor | NestedTensor):
        if isinstance(tensor, torch.Tensor):
            feats = [f.tensors for f in self._features]
            masks = [f.mask for f in self._features]
            return feats, masks, self._cross_attention_features
        return self._features, self._cross_attention_features


class _FakePositionEncoding(nn.Module):
    """Tiny callable that behaves like a position encoder."""

    def forward(self, nested_tensor: NestedTensor | torch.Tensor, align_dim_orders: bool = False) -> torch.Tensor:
        if isinstance(nested_tensor, NestedTensor):
            base = nested_tensor.tensors
        else:
            base = nested_tensor
        if base.dim() == 3:
            base = base[:, None]
        return torch.zeros((base.shape[0], 1, base.shape[-2], base.shape[-1]), dtype=base.dtype, device=base.device)


def _feature(shape: tuple[int, ...], batch_size: int = 2) -> NestedTensor:
    """Build a NestedTensor feature map with an all-false mask.

    Examples:
        >>> feat = _feature((8, 4, 4), batch_size=1)
        >>> feat.tensors.shape
        torch.Size([1, 8, 4, 4])
    """
    channels, height, width = shape
    return NestedTensor(
        tensors=torch.ones((batch_size, channels, height, width), dtype=torch.float32),
        mask=torch.zeros((batch_size, height, width), dtype=torch.bool),
    )


def _input_tensor(batch_size: int = 2) -> tuple[NestedTensor, torch.Tensor]:
    """Return matching NestedTensor and raw image inputs.

    Examples:
        >>> nested, image = _input_tensor(batch_size=1)
        >>> nested.tensors.shape, image.shape
        (torch.Size([1, 3, 16, 16]), torch.Size([1, 3, 16, 16]))
    """
    return (
        NestedTensor(
            tensors=torch.ones((batch_size, 3, 16, 16), dtype=torch.float32),
            mask=torch.zeros((batch_size, 16, 16), dtype=torch.bool),
        ),
        torch.ones((batch_size, 3, 16, 16), dtype=torch.float32),
    )


def test_joiner_dual_projector_disabled_contract() -> None:
    """Joiner should forward one feature stream and a ``None`` cross-attention stream when disabled."""
    features = [_feature((256, 16, 16))]
    joiner = Joiner(_FakeBackbone(features, None), _FakePositionEncoding())

    input_tensor, image = _input_tensor()

    _, _, cross_attention = joiner(input_tensor)
    assert cross_attention is None
    assert len(joiner(input_tensor)[0]) == 1

    exported = joiner.forward_export(image)
    assert exported[3] is None
    assert len(exported[0]) == 1
    assert exported[2][0].shape == (2, 16, 16)


def test_joiner_dual_projector_enabled_contract() -> None:
    """Joiner should forward cross-attention features in parallel with feature features when enabled."""
    features = [_feature((256, 16, 16)), _feature((256, 8, 8))]
    cross_attention_features = [_feature((256, 16, 16)), _feature((256, 8, 8))]
    joiner = Joiner(_FakeBackbone(features, cross_attention_features), _FakePositionEncoding())

    input_tensor, _ = _input_tensor()

    feature_tensors, _, cross_attention = joiner(input_tensor)
    assert len(feature_tensors) == len(cross_attention)
    assert all(f.tensors.shape == c.tensors.shape for f, c in zip(feature_tensors, cross_attention))
    assert all(f.mask is not None for f in cross_attention)


def test_joiner_forward_export_contract() -> None:
    """Exported joiner contracts should remain 4-tuples and preserve cross-attention stream arity."""
    exported_features = [torch.ones(2, 256, 16, 16), torch.ones(2, 256, 8, 8)]
    exported_masks = [torch.zeros(2, 16, 16, dtype=torch.bool), torch.zeros(2, 8, 8, dtype=torch.bool)]
    export_backbone = _FakeBackbone(
        [NestedTensor(t, mask) for t, mask in zip(exported_features, exported_masks)],
        [torch.ones(2, 256, 16, 16), torch.ones(2, 256, 8, 8)],
    )
    joiner = Joiner(export_backbone, _FakePositionEncoding())

    outputs = joiner.forward_export(torch.ones(2, 3, 16, 16))
    feats_out, masks_out, poss, cross_attention = outputs

    assert len(feats_out) == len(exported_features)
    assert len(masks_out) == len(exported_masks)
    assert feats_out[0].shape == exported_features[0].shape
    assert masks_out[0].shape == exported_masks[0].shape
    assert len(outputs) == 4
    assert poss[0].shape == exported_features[0][:, :1, :, :].shape
    assert isinstance(cross_attention, list)
    assert all(isinstance(feature, torch.Tensor) for feature in cross_attention)


class TestBackboneLevelMask:
    """``Backbone._level_mask`` substitutes zeros only where that is exactly the interpolation.

    Nearest-neighbour resampling of an all-False mask is all-False at every output size, so the substitution must agree
    with ``F.interpolate`` for unpadded batches and must not be taken when the batch carries real padding.
    """

    @staticmethod
    def _interpolated(mask: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """Reference result: the interpolation the unflagged path performs.

        Examples:
            >>> mask = torch.zeros(1, 8, 8, dtype=torch.bool)
            >>> TestBackboneLevelMask._interpolated(mask, torch.rand(1, 4, 2, 2)).shape
            torch.Size([1, 2, 2])
        """
        return F.interpolate(mask[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]

    @pytest.mark.parametrize("no_padding", [True, False])
    def test_all_false_mask_matches_interpolation(self, no_padding: bool) -> None:
        """An all-False mask gives the interpolation result whether or not the no-padding shortcut flag is set.

        Flagged: the zeros shortcut equals what interpolating the all-False mask produces. Unflagged: the mask is still
        read and the result is unchanged.
        """
        mask = torch.zeros(2, 64, 64, dtype=torch.bool)
        feat = torch.rand(2, 8, 16, 16)
        nested = NestedTensor(torch.rand(2, 3, 64, 64), mask, no_padding)
        assert torch.equal(Backbone._level_mask(nested, feat), self._interpolated(mask, feat))

    def test_padded_mask_is_preserved(self) -> None:
        """Real padding must survive downsampling rather than be zeroed away."""
        mask = torch.zeros(1, 64, 64, dtype=torch.bool)
        mask[:, 32:, :] = True
        feat = torch.rand(1, 8, 16, 16)
        unflagged = NestedTensor(torch.rand(1, 3, 64, 64), mask, False)
        level = Backbone._level_mask(unflagged, feat)
        assert torch.equal(level, self._interpolated(mask, feat))
        assert level[:, 8:, :].all().item() is True
        assert level[:, :8, :].any().item() is False

    def test_output_shape_and_dtype_follow_the_feature_map(self) -> None:
        """The shortcut must produce the same shape/dtype/device contract as the interpolation."""
        mask = torch.zeros(3, 40, 40, dtype=torch.bool)
        feat = torch.rand(3, 8, 10, 20)
        flagged = NestedTensor(torch.rand(3, 3, 40, 40), mask, True)
        level = Backbone._level_mask(flagged, feat)
        assert level.shape == (3, 10, 20)
        assert level.dtype == torch.bool
        assert level.device == feat.device

    @pytest.mark.parametrize("no_padding", [True, False])
    def test_forward_propagates_no_padding_to_feature_batches(self, no_padding: bool) -> None:
        """The backbone must preserve the collate-time claim for position encoding consumers."""
        nested = NestedTensor(
            torch.rand(2, 3, 32, 32),
            torch.zeros(2, 32, 32, dtype=torch.bool),
            no_padding,
        )
        feature = torch.rand(2, 8, 8, 8)
        backbone = MagicMock()
        backbone.encoder.return_value = [feature]
        backbone.projector.return_value = [feature]
        backbone.cross_attn_projector = None
        backbone._level_mask = Backbone._level_mask

        outputs, cross_attention_outputs = Backbone.forward(backbone, nested)

        assert cross_attention_outputs is None
        assert len(outputs) == 1
        assert outputs[0].no_padding is no_padding
        assert outputs[0].mask is not None

    def test_matches_unflagged_after_default_config_batch_uniform_resize(self) -> None:
        """The shortcut still agrees with the interpolation after the real default-training mutation.

        The default training config (``square_resize_div_64=True``, ``multi_scale="per-batch"``) resizes every sample to
        one fixed square scale before collate, so the batch is flagged. ``RFDETRLightningModule.on_train_batch_start``
        then resizes the whole batch uniformly to a randomly chosen scale in place, without touching ``no_padding`` (see
        ``tests/utilities/test_tensors.py::TestNestedTensorNoPadding::test_flag_survives_inplace_batch_uniform_resize``
        for the flag/mask invariant this relies on). The two consumers of the mask must still agree afterwards.
        """
        images = [torch.rand(3, 512, 512) for _ in range(2)]
        nested = nested_tensor_from_tensor_list(images, block_size=64)
        with torch.no_grad():
            nested.tensors = F.interpolate(nested.tensors, size=(640, 640), mode="bilinear", align_corners=False)
            nested.mask = (
                F.interpolate(nested.mask.unsqueeze(1).float(), size=(640, 640), mode="nearest").squeeze(1).bool()
            )

        unflagged_twin = NestedTensor(nested.tensors, nested.mask, False)
        feat = torch.rand(2, 8, 20, 20)
        assert torch.equal(Backbone._level_mask(nested, feat), Backbone._level_mask(unflagged_twin, feat))


def _toy_backbone() -> Backbone:
    """Build a ``Backbone`` whose encoder and projector are tiny modules named like the DINOv2 layout.

    ``Backbone.__init__`` builds a real DINOv2 encoder, so the instance is created without it and given modules whose
    parameter names exercise the embedding, per-layer and norm branches of the learning-rate and weight-decay rules.

    Examples:
        >>> sorted(name for name, _ in _toy_backbone().named_parameters())[:2]
        ['encoder.embeddings.weight', 'encoder.encoder.layer.0.bias']
    """
    backbone = object.__new__(Backbone)
    nn.Module.__init__(backbone)
    backbone.encoder = nn.ModuleDict(
        {
            "embeddings": nn.Linear(2, 2, bias=False),
            "encoder": nn.ModuleDict({"layer": nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])}),
            "layernorm": nn.LayerNorm(2),
        }
    )
    backbone.projector = nn.Linear(2, 2)
    return backbone


def _lr_args() -> Namespace:
    """Learning-rate settings with every factor different from 1, so each rule changes the result.

    Examples:
        >>> _lr_args().out_feature_indexes
        [1]
    """
    return Namespace(
        out_feature_indexes=[1],
        lr_encoder=1e-4,
        lr_vit_layer_decay=0.8,
        lr_component_decay=0.7,
        weight_decay=1e-4,
    )


def _lr_settings(pairs: dict[str, dict[str, Any]], prefix: str) -> dict[str, tuple[float, float]]:
    """Map each parameter name, without *prefix*, to its learning rate and weight decay.

    Examples:
        >>> _lr_settings({"backbone.1.encoder.w": {"params": None, "lr": 0.1, "weight_decay": 0.0}}, "backbone.1")
        {'.encoder.w': (0.1, 0.0)}
    """
    return {name.removeprefix(prefix): (pair["lr"], pair["weight_decay"]) for name, pair in pairs.items()}


class TestGetNamedParamLrPairs:
    """``Backbone.get_named_param_lr_pairs`` covers the encoder under whatever ``prefix`` the caller names.

    The prefix is the backbone's own path in the model (``backbone.0`` for the single backbone of ``LWDETR``); a model
    that holds a backbone under another path must still get the encoder learning rate and decay for it.
    """

    @pytest.mark.parametrize("prefix", ["backbone.0", "backbone.1"])
    def test_returns_every_trainable_encoder_parameter_under_the_prefix(self, prefix: str) -> None:
        backbone = _toy_backbone()

        pairs = backbone.get_named_param_lr_pairs(_lr_args(), prefix=prefix)

        expected = {f"{prefix}.{name}" for name, _ in backbone.named_parameters() if name.startswith("encoder.")}
        assert set(pairs) == expected

    def test_learning_rate_and_weight_decay_do_not_depend_on_the_prefix(self) -> None:
        backbone = _toy_backbone()
        args = _lr_args()

        default = backbone.get_named_param_lr_pairs(args, prefix="backbone.0")
        other = backbone.get_named_param_lr_pairs(args, prefix="backbone.1")

        assert _lr_settings(other, "backbone.1") == _lr_settings(default, "backbone.0")
