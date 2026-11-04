# Dual-sensor RF-DETR Nano: thermal + RGB fusion with bi-directional cross-attention

Implementation plan for extending RF-DETR Nano to two backbones (thermal and RGB) whose features are fused by a bi-directional cross-attention block and decoded into a single set of predictions.

## 1. Data we have

| Source                  | Content                                                                                       |
| ----------------------- | --------------------------------------------------------------------------------------------- |
| Thermal-only (unpaired) | Many thermal images, classes 1 and 2, bounding boxes                                          |
| RGB-only (unpaired)     | Many RGB images, same classes and box format, captured at different times and places          |
| Paired (simultaneous)   | Limited set of thermal + RGB pairs, split from the camera's split-screen view and **aligned** |

Rules that follow from this:

- **Never build synthetic pairs** from unrelated thermal and RGB images. The model would learn that the other sensor is noise and switch the fusion off.
- The paired images are aligned, so both sensors share one image frame. Use **thermal as the reference frame** for labels and outputs (it is the sensor that still works at night and in smoke).

## 2. How Nano works today

| Stage       | Code                                                                                           | Nano                                                                                                                               |
| ----------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Backbone    | `models/backbone/backbone.py` `Backbone`, wrapped by `Joiner` in `models/backbone/__init__.py` | Windowed DINOv2-Small, 384×384 input, 16-px patches; layers [3, 6, 9, 12] projected to one **P4 map: 256 ch × 24×24 (576 tokens)** |
| Position    | `Joiner` (second element)                                                                      | Sine position embedding per map                                                                                                    |
| Transformer | `models/transformer.py` `Transformer.forward(srcs, masks, poss, …)`                            | **No transformer encoder**: flattened map is the memory for two-stage query selection, then a 2-layer deformable decoder           |
| Heads       | `models/lwdetr.py` `LWDETR.forward`                                                            | Box and class heads; 13 query groups in training                                                                                   |

The seam is `LWDETR.forward`: `features, poss, _ = self.backbone(samples)` hands lists of maps to the transformer, and everything after that only sees those lists. The fusion goes there, so the decoder, heads, loss and matcher stay unchanged.

The existing `dual_projector` / `cross_attn_srcs` feature (one backbone, two projectors, used by keypoint models) is unrelated. Do not build on it, but do not break it.

## 3. Target architecture

```
thermal ─► Backbone A (DINOv2-S + projector) ─► F1 (B,256,24,24) + pos1 ─┐
                                                                         ├─► BiCrossAttentionFusion ─► concat merge ─► existing Transformer/decoder ─► one prediction set
RGB     ─► Backbone B (DINOv2-S + projector) ─► F2 (B,256,24,24) + pos2 ─┘
```

### Fusion block (`models/fusion.py`, `BiCrossAttentionFusion`)

`N` layers (start with `N = 2`). Each layer:

- Tokens `T1, T2` of shape `(B, 576, 256)`. Each sensor's tokens get its position embedding plus a learned **sensor embedding** so attention can tell the sensors apart.
- Both directions are computed from the same layer input:
    - `T1' = T1 + g1 · CrossAttn(q=T1, k=T2, v=T2)`
    - `T2' = T2 + g2 · CrossAttn(q=T2, k=T1, v=T1)`
- Feed-forward layer and LayerNorm, as in the existing decoder layers.
- Plain `nn.MultiheadAttention`, 8 heads. Full attention over 576 × 576 tokens is cheap, so deformable attention is not needed. Padding masks go in as `key_padding_mask`.
- **Gates `g1`, `g2` start at 0** (or zero-initialised output projections), so at the start of training each branch behaves like its pretrained single-sensor detector.

Global cross-attention lets each location find its match anywhere in the other map, so small residual misalignment between the sensors is tolerated. At 16 px per feature cell, offsets of a few pixels fall inside one cell anyway.

### Missing sensor

A per-sample "sensor present" flag. When a sensor is missing:

- its tokens are replaced by a learned **missing-sensor embedding**;
- cross-attention *from* that sensor is switched off by zeroing that attention output. Do **not** use an all-`True` `key_padding_mask`: `nn.MultiheadAttention` returns NaNs when every key is masked.

### Merge

- **`merge="concat"` (default).** Concatenate the two fused maps (512 ch) and apply a 1×1 conv back to 256 ch. One level, so decoder shapes are unchanged and decoder and head weights load as they are. Initialise the conv as `[I | 0]` so that at initialisation the dual model equals the thermal detector.
- **`merge="levels"` (fallback).** Pass both fused maps to the decoder as 2 levels (`n_levels = 2`); deformable attention samples each separately. Use only if residual misalignment turns out to hurt. The decoder's `sampling_offsets` and `attention_weights` change shape and must be re-initialised for the new level.

## 4. Code changes

Implementation approach: a new `DualJoiner` in `models/backbone/__init__.py` holds `backbone_a`, `backbone_b`, the fusion block and the merge, takes `(NestedTensor_thermal, NestedTensor_rgb, sensor_present)` and **returns the same `(features, poss, None)` tuple as `Joiner`**. `LWDETR.forward` then needs only a small change to accept paired input (today it runs `nested_tensor_from_tensor_list(samples)` on any list or tensor).

| File                                                            | Change                                                                                                                                                                                                                                                                            |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.py`                                                     | `RFDETRNanoDualConfig(RFDETRNanoConfig)` with `fusion_layers=2`, `fusion_heads=8`, `fusion_merge="concat"`, `share_backbone_weights=False`, `sensor2_in_chans=3`. Register new fields in `_namespace.py` and `models/_types.py` like `dual_projector`.                            |
| `models/fusion.py` (new)                                        | `BiCrossAttentionFusion`, missing-sensor embedding, merge module.                                                                                                                                                                                                                 |
| `models/backbone/__init__.py`                                   | `DualJoiner`; `build_backbone` builds it for the dual config.                                                                                                                                                                                                                     |
| `models/lwdetr.py`                                              | Accept paired input in `forward`; pass dual settings through `build_model`. Fix `_get_backbone_encoder_layers` / `update_drop_path` (they assume `self.backbone[0]`) to cover both encoders.                                                                                      |
| `training/param_groups.py`, `Backbone.get_named_param_lr_pairs` | **Trap:** `backbone_key = "backbone.0.encoder"` is hard-coded (`backbone.py:277`), so a second backbone under another prefix would silently miss the encoder learning rate and layer decay. Pass the prefix through; give the fusion block its own group (decoder learning rate). |
| `models/weights.py`, `utilities/state_dict.py`                  | Load backbone A from the thermal detector checkpoint and backbone B from the RGB detector checkpoint (`backbone.0.*` → each branch). Transformer and heads from the thermal detector. List the expected missing keys (fusion, merge, embeddings) so warnings stay meaningful.     |
| `detr.py`                                                       | Public `RFDETRNanoDual` class modelled on `RFDETRNano`.                                                                                                                                                                                                                           |

## 5. Data pipeline

- **Thermal format.** Convert every thermal image (standalone and paired) to one representation: single-channel grayscale with the same white-hot/black-hot polarity, repeated to 3 channels. Paired thermal images came from a display view and may use a colour palette or lower resolution than the standalone set. Per-sensor normalisation mean and std.
- **Paired dataset.** COCO JSON whose image entries carry two paths (for example `file_name` for thermal and `file_name_rgb`), boxes in the thermal frame. Extend `datasets/coco.py` or add `datasets/paired_coco.py`, returning `(thermal, rgb, sensor_present, target)`.
- **Mixed dataset.** Weighted sampling over paired, thermal-only and RGB-only sources (start at 50 / 25 / 25). Single-sensor samples get `sensor_present` with the other sensor off.
- **Matched augmentations** for paired samples: geometric transforms (flip, crop, resize, multi-scale) use **the same random values on both images**; photometric transforms are drawn **separately per sensor**. Albumentations `additional_targets` covers the CPU path. Start with `augmentation_backend="torchvision"` or `"cpu"`; leave the Kornia GPU path out.
- **Collate** (`utilities/tensors.py` `collate_fn` / `make_collate_fn`): two `NestedTensor`s padded to the same size, plus the `sensor_present` tensor.
- **Multi-scale training** (`training/module_model.py`, around lines 851–860) resizes `samples.tensors` and `samples.mask` in place; it must resize both sensors identically.
- **Splits by session or location**, not by random frame, so near-duplicate frames never cross train/val/test. Keep a held-out **paired** test set.

## 6. Phases

### Phase 0: data preparation

1. Convert all thermal images to one format.
2. Build the paired COCO JSON and the session-based train/val/test splits.
3. Draw the boxes on a sample of pairs to confirm they line up on both images.

### Phase 1: single-sensor detectors (no code changes)

1. Set up the environment (`uv venv` and the install step in `AGENTS.md`) and run the CPU tests on `develop`.
2. Fine-tune Nano on all thermal images **plus the thermal images from the pairs** → thermal detector.
3. Fine-tune Nano on all RGB images **plus the RGB images from the pairs** → RGB detector.
4. Evaluate both on the paired test set. These are the baselines and fallback detectors.

### Phase 2: dual model (no real data yet)

Config, `BiCrossAttentionFusion`, `DualJoiner`, `LWDETR` input change. Unit tests (CPU, Nano, per `AGENTS.md`):

- output shapes for random pairs;
- **at initialisation the dual model's output equals the thermal-only model** (gates and merge start as no-ops);
- gradients reach both backbones and the fusion block once gates are non-zero;
- padding masks handled;
- missing sensor: no NaNs, output equals the single-sensor path.

### Phase 3: weights

Load backbone A from the thermal checkpoint and backbone B from the RGB checkpoint; check missing and unexpected keys; test the learning-rate groups (both encoders get `lr_encoder` and layer decay, fusion has its own group).

### Phase 4: data pipeline

Paired and mixed datasets, matched augmentations, collate, multi-scale resize for both sensors. Test that a flip or crop moves both images and the boxes the same way.

### Phase 5: fusion training (paired data)

1. Overfit about 20 pairs to confirm the loss goes to near zero.
2. **Freeze both backbones**; train fusion, merge and decoder on the pairs.
3. With enough pairs (roughly a few thousand or more), unfreeze the backbones at a low learning rate or use LoRA (`backbone_lora`).
4. Keep `compile`, CUDA graphs and FP8 off until training is stable.

### Phase 6: mixed training (optional, recommended)

Train on paired + thermal-only + RGB-only together, with random sensor dropout on about 15% of paired samples. Keeps the decoder from overfitting the small paired set and gives one model that works when a sensor is missing.

### Phase 7: evaluation and ablations

All on the held-out paired test set (cross-validate if it is small):

- fused vs thermal-only vs RGB-only detectors;
- `fusion_layers=0` (merge only, no cross-attention);
- one-directional vs bi-directional attention;
- `concat` vs `levels` merge;
- shared vs separate backbone weights;
- robustness with each sensor blanked.

### Phase 8: prediction

A small `predict_pair(thermal, rgb)` helper. `RFDETRInference` and export are image-only; leave them for later.

## 7. Out of scope for the first version

ONNX/TensorRT export, `RFDETRInference`, segmentation and keypoint heads, GPU (Kornia) augmentations, CUDA-graph training.

## 8. Cost

Each pair runs the backbone twice. The backbone dominates Nano's compute, so expect roughly 1.8–2× training and inference time. Sharing backbone weights saves memory, not compute.

## 9. Open questions

- How many paired images are there? Decides whether backbones stay frozen in phase 5.
- What format is the data in (COCO JSON, YOLO, other)?
- Are boxes on both images of each pair, or thermal only?
