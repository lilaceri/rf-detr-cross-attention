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


def _missing(sensor: str, batch_size: int = 2) -> dict[str, torch.Tensor]:
    """Return ``present_a``/``present_b`` keyword arguments where *sensor* is missing for sample 0 only.

    Examples:
        >>> _missing("b")["present_b"].tolist()
        [False, True]
    """
    present = torch.ones(batch_size, dtype=torch.bool)
    absent = present.clone()
    absent[0] = False
    return {"present_a": absent if sensor == "a" else present, "present_b": absent if sensor == "b" else present}


def _pick(outputs: tuple[torch.Tensor, torch.Tensor], sensor: str) -> torch.Tensor:
    """Return the output map of *sensor* (``"a"`` or ``"b"``) from a fusion result.

    Examples:
        >>> _pick((torch.zeros(1), torch.ones(1)), "b").item()
        1.0
    """
    return outputs[0] if sensor == "a" else outputs[1]


class TestMissingSensor:
    """A sample may lack one sensor; the other is then fused from nothing and the missing map is a learned embedding."""

    @pytest.mark.parametrize("missing", ["a", "b"])
    def test_present_sensor_ignores_the_missing_sensors_content(self, missing: str) -> None:
        fusion = _open_gates(_fusion())
        src_a, pos_a = _maps(0)
        src_b, pos_b = _maps(1)
        present = "b" if missing == "a" else "a"
        garbage_a = src_a.clone()
        garbage_b = src_b.clone()
        (garbage_a if missing == "a" else garbage_b)[0] = float("nan")

        reference = _pick(fusion(src_a, src_b, pos_a, pos_b, **_missing(missing)), present)
        result = _pick(fusion(garbage_a, garbage_b, pos_a, pos_b, **_missing(missing)), present)

        assert torch.equal(result[0], reference[0])

    @pytest.mark.parametrize("missing", ["a", "b"])
    def test_missing_sensor_output_is_its_learned_embedding(self, missing: str) -> None:
        fusion = _open_gates(_fusion())
        with torch.no_grad():
            embedding = getattr(fusion, f"missing_embed_{missing}")
            embedding.copy_(torch.arange(_D_MODEL, dtype=torch.float32))
        src_a, pos_a = _maps(0)
        src_b, pos_b = _maps(1)

        result = _pick(fusion(src_a, src_b, pos_a, pos_b, **_missing(missing)), missing)

        assert torch.equal(result[0], embedding[:, None, None].expand(_D_MODEL, 3, 4))

    @pytest.mark.parametrize("missing", ["a", "b"])
    def test_fully_padded_missing_sensor_gives_finite_outputs_and_gradients(self, missing: str) -> None:
        fusion = _open_gates(_fusion()).train()
        src_a, pos_a = _maps(0)
        src_b, pos_b = _maps(1)
        mask = torch.zeros(2, 3, 4, dtype=torch.bool)
        mask[0] = True  # the missing sensor's map is all padding for sample 0
        masks = {"mask_a": mask} if missing == "a" else {"mask_b": mask}

        out_a, out_b = fusion(src_a, src_b, pos_a, pos_b, **masks, **_missing(missing))
        (out_a.sum() + out_b.sum()).backward()

        grads = [p.grad for p in fusion.parameters() if p.grad is not None]
        assert torch.isfinite(out_a).all() and torch.isfinite(out_b).all()
        assert grads and all(torch.isfinite(grad).all() for grad in grads)

    @pytest.mark.parametrize("missing", ["a", "b"])
    def test_missing_sensors_content_does_not_reach_the_gradients(self, missing: str) -> None:
        fusion = _open_gates(_fusion()).train()
        src_a, pos_a = _maps(0)
        src_b, pos_b = _maps(1)
        (src_a if missing == "a" else src_b)[0] = float("nan")

        out_a, out_b = fusion(src_a, src_b, pos_a, pos_b, **_missing(missing))
        (out_a.sum() + out_b.sum()).backward()

        grads = [p.grad for p in fusion.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(grad).all() for grad in grads)

    @pytest.mark.parametrize("missing", ["a", "b"])
    def test_samples_with_both_sensors_are_unaffected_by_another_samples_missing_sensor(self, missing: str) -> None:
        fusion = _open_gates(_fusion())
        src_a, pos_a = _maps(0)
        src_b, pos_b = _maps(1)

        all_present = fusion(src_a, src_b, pos_a, pos_b)
        one_missing = fusion(src_a, src_b, pos_a, pos_b, **_missing(missing))

        assert torch.equal(one_missing[0][1], all_present[0][1]) and torch.equal(one_missing[1][1], all_present[1][1])

    def test_missing_embedding_receives_gradient(self) -> None:
        fusion = _fusion().train()
        src_a, pos_a = _maps(0)
        src_b, pos_b = _maps(1)

        _, out_b = fusion(src_a, src_b, pos_a, pos_b, **_missing("b"))
        out_b.sum().backward()

        assert fusion.missing_embed_b.grad is not None and fusion.missing_embed_b.grad.abs().sum() > 0

    def test_rejects_a_sample_with_neither_sensor(self) -> None:
        src_a, pos_a = _maps(0)
        src_b, pos_b = _maps(1)
        absent = torch.tensor([False, True])

        with pytest.raises(ValueError, match="neither sensor"):
            _fusion()(src_a, src_b, pos_a, pos_b, present_a=absent, present_b=absent)
