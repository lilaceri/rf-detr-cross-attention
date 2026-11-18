# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Fusion of two sensors' feature maps by bi-directional cross-attention."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class _BiCrossAttentionLayer(nn.Module):
    """One fusion layer: each sensor's tokens attend to the other sensor's tokens, then a feed-forward block.

    Both directions read the layer's input, so neither direction sees the other's update from the same layer. Every
    residual branch is pre-normalized and scaled by a per-channel gate that starts at zero, so a new layer leaves its
    input unchanged until training opens the gates.

    Args:
        d_model: Channels per token.
        num_heads: Attention heads per direction.
        dim_feedforward: Hidden width of each feed-forward block.
        dropout: Dropout in attention and feed-forward blocks.
    """

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_b = nn.LayerNorm(d_model)
        self.attn_a_from_b = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_b_from_a = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_gate_a = nn.Parameter(torch.zeros(d_model))
        self.attn_gate_b = nn.Parameter(torch.zeros(d_model))
        self.ffn_norm_a = nn.LayerNorm(d_model)
        self.ffn_norm_b = nn.LayerNorm(d_model)
        self.ffn_a = _feed_forward(d_model, dim_feedforward, dropout)
        self.ffn_b = _feed_forward(d_model, dim_feedforward, dropout)
        self.ffn_gate_a = nn.Parameter(torch.zeros(d_model))
        self.ffn_gate_b = nn.Parameter(torch.zeros(d_model))

    def forward(
        self,
        tokens_a: Tensor,
        tokens_b: Tensor,
        pos_a: Tensor,
        pos_b: Tensor,
        mask_a: Tensor | None,
        mask_b: Tensor | None,
        present_a: Tensor,
        present_b: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Update both token sequences from each other.

        Args:
            tokens_a: Sensor A tokens, ``(batch, tokens_a, d_model)``.
            tokens_b: Sensor B tokens, ``(batch, tokens_b, d_model)``.
            pos_a: Positional embeddings for ``tokens_a``.
            pos_b: Positional embeddings for ``tokens_b``.
            mask_a: Padding mask for sensor A, ``(batch, tokens_a)``, ``True`` where padded, or ``None``.
            mask_b: Padding mask for sensor B, ``(batch, tokens_b)``, ``True`` where padded, or ``None``.
            present_a: ``(batch,)`` booleans, ``False`` where sensor A is missing; sensor B gets no update from it.
            present_b: ``(batch,)`` booleans, ``False`` where sensor B is missing; sensor A gets no update from it.

        Returns:
            The updated ``(tokens_a, tokens_b)``.
        """
        normed_a = self.norm_a(tokens_a)
        normed_b = self.norm_b(tokens_b)
        from_b, _ = self.attn_a_from_b(
            normed_a + pos_a, normed_b + pos_b, normed_b, key_padding_mask=mask_b, need_weights=False
        )
        from_a, _ = self.attn_b_from_a(
            normed_b + pos_b, normed_a + pos_a, normed_a, key_padding_mask=mask_a, need_weights=False
        )
        # A sample gets no update from a sensor it lacks.
        from_b = torch.where(present_b[:, None, None], from_b, 0.0)
        from_a = torch.where(present_a[:, None, None], from_a, 0.0)
        tokens_a = tokens_a + self.attn_gate_a * from_b
        tokens_b = tokens_b + self.attn_gate_b * from_a
        tokens_a = tokens_a + self.ffn_gate_a * self.ffn_a(self.ffn_norm_a(tokens_a))
        tokens_b = tokens_b + self.ffn_gate_b * self.ffn_b(self.ffn_norm_b(tokens_b))
        return tokens_a, tokens_b


def _feed_forward(d_model: int, dim_feedforward: int, dropout: float) -> nn.Sequential:
    """Return the two-layer ReLU feed-forward block used after each attention.

    Args:
        d_model: Input and output channels.
        dim_feedforward: Hidden channels.
        dropout: Dropout after the activation.

    Returns:
        The feed-forward block.
    """
    return nn.Sequential(
        nn.Linear(d_model, dim_feedforward),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(dim_feedforward, d_model),
    )


class BiCrossAttentionFusion(nn.Module):
    """Bi-directional cross-attention between two sensors' feature maps.

    Each layer lets every location of sensor A attend to all locations of sensor B, and the reverse, so the two maps
    need not be pixel-aligned. Gates start at zero, so at initialization the block returns its inputs unchanged and
    two pretrained single-sensor branches behave as they did before fusion was added.

    A sample may lack one of the sensors (``present_a`` / ``present_b``). The present sensor then gets no update from
    the missing one, and the missing sensor's output map is a learned per-channel embedding, so code downstream always
    receives two maps. The missing sensor's input values and padding are never read, so they may be anything.

    Args:
        d_model: Channels of both feature maps.
        num_heads: Attention heads per direction.
        num_layers: Number of fusion layers.
        dim_feedforward: Hidden width of the feed-forward blocks.
        dropout: Dropout in attention and feed-forward blocks.
    """

    def __init__(
        self, d_model: int, num_heads: int, num_layers: int, dim_feedforward: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_BiCrossAttentionLayer(d_model, num_heads, dim_feedforward, dropout) for _ in range(num_layers)]
        )
        self.missing_embed_a = nn.Parameter(torch.zeros(d_model))
        self.missing_embed_b = nn.Parameter(torch.zeros(d_model))

    def forward(
        self,
        src_a: Tensor,
        src_b: Tensor,
        pos_a: Tensor,
        pos_b: Tensor,
        mask_a: Tensor | None = None,
        mask_b: Tensor | None = None,
        present_a: Tensor | None = None,
        present_b: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Fuse two feature maps.

        Args:
            src_a: Sensor A feature map, ``(batch, d_model, height_a, width_a)``.
            src_b: Sensor B feature map, ``(batch, d_model, height_b, width_b)``.
            pos_a: Positional embedding for ``src_a``, same shape.
            pos_b: Positional embedding for ``src_b``, same shape.
            mask_a: Padding mask for sensor A, ``(batch, height_a, width_a)``, ``True`` where padded, or ``None``.
            mask_b: Padding mask for sensor B, ``(batch, height_b, width_b)``, ``True`` where padded, or ``None``.
            present_a: ``(batch,)`` booleans, ``False`` where the sample has no sensor A image. ``None`` means every
                sample has one.
            present_b: ``(batch,)`` booleans, ``False`` where the sample has no sensor B image. ``None`` means every
                sample has one.

        Returns:
            The fused ``(src_a, src_b)`` maps, each with its input's shape. Where a sensor is missing, its map is that
            sensor's missing embedding at every location.

        Raises:
            ValueError: If a sample has neither sensor.
        """
        present_a = _presence(present_a, src_a)
        present_b = _presence(present_b, src_b)
        if not bool((present_a | present_b).all()):
            raise ValueError("Every sample needs at least one sensor; a sample has neither sensor present.")
        # A missing sensor's tokens and padding are replaced before attention, so its input is never read. Non-finite
        # values there would otherwise reach the attention weights' gradients, and older PyTorch versions return NaN
        # for a sample whose keys are all padding.
        tokens_a = torch.where(present_a[:, None, None], _tokens(src_a), self.missing_embed_a)
        tokens_b = torch.where(present_b[:, None, None], _tokens(src_b), self.missing_embed_b)
        flat_pos_a, flat_pos_b = _tokens(pos_a), _tokens(pos_b)
        flat_mask_a = mask_a.flatten(1) & present_a[:, None] if mask_a is not None else None
        flat_mask_b = mask_b.flatten(1) & present_b[:, None] if mask_b is not None else None
        for layer in self.layers:
            tokens_a, tokens_b = layer(
                tokens_a, tokens_b, flat_pos_a, flat_pos_b, flat_mask_a, flat_mask_b, present_a, present_b
            )
        tokens_a = torch.where(present_a[:, None, None], tokens_a, self.missing_embed_a)
        tokens_b = torch.where(present_b[:, None, None], tokens_b, self.missing_embed_b)
        return tokens_a.transpose(1, 2).reshape(src_a.shape), tokens_b.transpose(1, 2).reshape(src_b.shape)


def _presence(present: Tensor | None, feature_map: Tensor) -> Tensor:
    """Return the per-sample presence flags, all ``True`` when *present* is ``None``.

    Args:
        present: ``(batch,)`` booleans, or ``None``.
        feature_map: The sensor's ``(batch, ...)`` map, which gives the batch size and device.

    Returns:
        ``(batch,)`` booleans on the map's device.

    Examples:
        >>> _presence(None, torch.zeros(2, 8, 3, 4)).tolist()
        [True, True]
    """
    if present is None:
        return torch.ones(feature_map.shape[0], dtype=torch.bool, device=feature_map.device)
    return present.to(device=feature_map.device, dtype=torch.bool)


def _tokens(feature_map: Tensor) -> Tensor:
    """Flatten a ``(batch, channels, height, width)`` map to ``(batch, height * width, channels)`` tokens.

    Args:
        feature_map: The map to flatten.

    Returns:
        One token per location.

    Examples:
        >>> _tokens(torch.zeros(2, 8, 3, 4)).shape
        torch.Size([2, 12, 8])
    """
    return feature_map.flatten(2).transpose(1, 2)
