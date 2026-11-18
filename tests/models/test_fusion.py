# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the bi-directional cross-attention fusion of two sensors' feature maps."""

from __future__ import annotations

import pytest
import torch

from rfdetr.models.fusion import BiCrossAttentionFusion

_D_MODEL = 16


def _maps(seed: int, batch_size: int = 2, height: int = 3, width: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a seeded ``(src, pos)`` pair of feature maps shaped ``(batch_size, _D_MODEL, height, width)``.

    Examples:
        >>> src, pos = _maps(0)
        >>> tuple(src.shape), tuple(pos.shape)
        ((2, 16, 3, 4), (2, 16, 3, 4))
    """
    generator = torch.Generator().manual_seed(seed)
    shape = (batch_size, _D_MODEL, height, width)
    return torch.randn(shape, generator=generator), torch.randn(shape, generator=generator)


def _fusion(seed: int = 0, num_layers: int = 2) -> BiCrossAttentionFusion:
    """Return a seeded fusion block in eval mode.

    Examples:
        >>> _fusion().training
        False
    """
    torch.manual_seed(seed)
    return BiCrossAttentionFusion(d_model=_D_MODEL, num_heads=4, num_layers=num_layers, dim_feedforward=32).eval()


def _open_gates(fusion: BiCrossAttentionFusion) -> BiCrossAttentionFusion:
    """Set every gate of *fusion* to one, so each residual branch contributes.

    Examples:
        >>> fusion = _open_gates(_fusion())
        >>> all(bool((gate == 1).all()) for name, gate in fusion.named_parameters() if "gate" in name)
        True
    """
    with torch.no_grad():
        for name, parameter in fusion.named_parameters():
            if "gate" in name:
                parameter.fill_(1.0)
    return fusion


class TestBiCrossAttentionFusion:
    """Each sensor's map is updated from the other's, starting as a no-op so pretrained branches are kept."""

    def test_outputs_have_the_input_shapes(self) -> None:
        src_a, pos_a = _maps(0)
        src_b, pos_b = _maps(1)

        out_a, out_b = _open_gates(_fusion())(src_a, src_b, pos_a, pos_b)

        assert (out_a.shape, out_b.shape) == (src_a.shape, src_b.shape)

    def test_is_the_identity_at_initialisation(self) -> None:
        src_a, pos_a = _maps(0)
        src_b, pos_b = _maps(1)

        out_a, out_b = _fusion()(src_a, src_b, pos_a, pos_b)

        assert torch.equal(out_a, src_a) and torch.equal(out_b, src_b)

    def test_gates_receive_gradient_at_initialisation(self) -> None:
        fusion = _fusion().train()
        src_a, pos_a = _maps(0)
        src_b, pos_b = _maps(1)

        out_a, out_b = fusion(src_a, src_b, pos_a, pos_b)
        (out_a.square().sum() + out_b.square().sum()).backward()

        gate_grads = [p.grad for name, p in fusion.named_parameters() if "gate" in name]
        assert gate_grads and all(grad is not None and grad.abs().sum() > 0 for grad in gate_grads)

    @pytest.mark.parametrize("changed", ["a", "b"])
    def test_each_output_depends_on_the_other_sensor(self, changed: str) -> None:
        fusion = _open_gates(_fusion())
        src_a, pos_a = _maps(0)
        src_b, pos_b = _maps(1)
        other, _ = _maps(2)

        out_a, out_b = fusion(src_a, src_b, pos_a, pos_b)
        if changed == "b":
            new_out, _ = fusion(src_a, other, pos_a, pos_b)
            assert not torch.allclose(new_out, out_a)
        else:
            _, new_out = fusion(other, src_b, pos_a, pos_b)
            assert not torch.allclose(new_out, out_b)

    def test_padded_positions_of_the_other_sensor_are_ignored(self) -> None:
        fusion = _open_gates(_fusion(num_layers=1))
        src_a, pos_a = _maps(0)
        src_b, pos_b = _maps(1)
        mask_b = torch.zeros(2, 3, 4, dtype=torch.bool)
        mask_b[:, :, -1] = True  # last column of sensor b is padding
        changed_b = src_b.clone()
        changed_b[..., -1] = 100.0

        out_a, _ = fusion(src_a, src_b, pos_a, pos_b, mask_b=mask_b)
        out_a_changed, _ = fusion(src_a, changed_b, pos_a, pos_b, mask_b=mask_b)

        # Same computation on the unmasked keys; only floating-point summation order may differ.
        torch.testing.assert_close(out_a_changed, out_a, rtol=1e-5, atol=1e-6)
