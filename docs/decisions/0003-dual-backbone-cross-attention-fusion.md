# 0003: Dual backbone with bi-directional cross-attention fusion

- **Date:** 2026-11-04
- **Status:** Proposed (no experiment yet)
- **Links:** `plans/dual-sensor-fusion.md` (sections 2–4), [0002](0002-paired-data-and-thermal-reference-frame.md)

## Context

RF-DETR Nano runs one windowed DINOv2-Small backbone whose projector produces a single P4 map (256 channels, 24×24 at 384×384 input). `LWDETR.forward` passes that map to a transformer with no encoder: two-stage query selection, then a 2-layer deformable decoder. We need one prediction set that uses both thermal and RGB features.

## Options considered

1. **Early fusion:** stack thermal and RGB as extra input channels into one backbone. Cheapest, but loses the pretrained RGB patch embedding and cannot handle a missing sensor.
2. **Late fusion:** run two full detectors and merge their boxes. Simple, but the sensors never inform each other's features.
3. **Feature fusion with bi-directional cross-attention between two backbones**, then one decoder.

For the merge before the decoder:

- **`concat`:** concatenate the fused maps and project back to 256 channels with a 1×1 convolution. Keeps one feature level, so the pretrained decoder loads unchanged. Assumes the maps are spatially aligned.
- **`levels`:** pass the two fused maps as two decoder levels. Works without alignment, but the decoder's sampling layers change shape and must be partly re-initialised.

## Decision

Option 3 with the `concat` merge as the default and `levels` as a fallback:

- Backbone A (thermal) and backbone B (RGB), each a Nano backbone with its own weights.
- `BiCrossAttentionFusion` with 2 layers: each sensor attends to the other, both directions from the same layer input, with a separate `nn.MultiheadAttention` (8 heads) per direction and a feed-forward block per sensor. No sensor embeddings (see the update below).
- Every residual branch (cross-attention and feed-forward) is pre-normalized and scaled by a per-channel gate that starts at 0, and the merge convolution starts as `[I | 0]`, so the initial model equals the thermal detector.
- A per-sample sensor-present flag and a learned missing-sensor embedding; the update from a missing sensor is dropped, and its input tokens and padding are replaced before attention (see the second update below).
- The fusion sits in a `DualJoiner` that returns the same outputs as `Joiner`, so the transformer, heads, loss and matcher are unchanged.

## Rationale and trade-offs

- Global cross-attention lets each location find its match anywhere in the other map, so small residual misalignment is tolerated, and offsets of a few pixels fall inside one 16-pixel feature cell.
- Full attention over 576 tokens per sensor is cheap compared with the backbones.
- Zero-initialised gates keep the pretrained detectors intact at the start of training.
- Two backbones cost roughly 1.8–2× Nano's compute; sharing weights would save memory, not compute.

## Consequences

- To validate in the ablations of plan phase 7: `fusion_layers=0`, one- vs bi-directional attention, `concat` vs `levels`, shared vs separate backbones, and each sensor blanked. Record the results in `docs/experiments/` and update this record's status.
- `Backbone.get_named_param_lr_pairs` hard-codes the `backbone.0.encoder` prefix; it must take the prefix as an argument so the second encoder gets the encoder learning rate. That fix is a candidate for upstream.

## Update 2026-11-18: no sensor embeddings

The first version of this record added a learned sensor embedding to each sensor's tokens so attention could tell the sensors apart. The implementation in `src/rfdetr/models/fusion.py` (PR adding `BiCrossAttentionFusion`) uses a separate attention module for each direction instead: thermal-from-RGB and RGB-from-thermal have their own query, key and value projections, so each already knows which sensor it reads. A sensor embedding would add parameters without adding information, so it was left out. If one shared attention module is ever used for both directions, sensor embeddings become necessary again.

The gates are per-channel vectors (LayerScale-style) on both the cross-attention and the feed-forward branches, rather than one scalar per direction, so that the whole layer, not only the attention, starts as the identity.

## Update 2026-11-18: how a missing sensor is handled

Implemented in `BiCrossAttentionFusion` with `present_a` / `present_b` (one boolean per sample):

- A missing sensor's input tokens are replaced by its learned embedding, and its padding mask by "no padding", before any attention runs, so its input is never read.
- The present sensor's update from the missing one is dropped, so the present sensor is fused from nothing.
- The missing sensor's output map is its embedding at every location, so the merge always receives two maps.
- A sample with neither sensor is rejected.

Why the input is replaced rather than only the output dropped: with PyTorch 2.15, a NaN in the missing sensor's input still produced NaN gradients in the attention weights even when the output was dropped (a test covers this). The padding replacement guards against older PyTorch versions, which return NaN for a sample whose keys are all padding; PyTorch 2.15 no longer does, so no test here can show it, but `rfdetr` supports PyTorch from 2.2.
