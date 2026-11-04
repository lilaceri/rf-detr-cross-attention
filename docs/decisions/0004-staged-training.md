# 0004: Staged training with single-sensor detectors first

- **Date:** 2026-11-04
- **Status:** Proposed (no experiment yet)
- **Links:** `plans/dual-sensor-fusion.md` (section 6), [0002](0002-paired-data-and-thermal-reference-frame.md), [0003](0003-dual-backbone-cross-attention-fusion.md)

## Context

There are many unpaired thermal and RGB images but only a limited number of real pairs. The fusion block can learn only from pairs, while the backbones and decoder can learn from any labelled image.

## Options considered

1. Train the dual model from Nano's COCO weights on the pairs only.
2. Train on pairs plus synthetic pairs built from unpaired images (rejected in [0002](0002-paired-data-and-thermal-reference-frame.md)).
3. **Stages:** single-sensor detectors on all data, then fusion on pairs, then optional mixed training.

## Decision

Option 3:

1. Fine-tune Nano on all thermal images plus the thermal images of the pairs, and separately on all RGB images plus the RGB images of the pairs. These are the baselines and fallback detectors.
2. Build the dual model with backbone A from the thermal detector, backbone B from the RGB detector, and the decoder and heads from the thermal detector. Freeze both backbones and train the fusion, merge and decoder on the pairs. Unfreeze at a low learning rate, or use LoRA, only with enough pairs (roughly a few thousand or more).
3. Optionally, train on paired, thermal-only and RGB-only data together (starting at 50 / 25 / 25), with the missing sensor marked absent and about 15% random sensor dropout on pairs.

## Rationale and trade-offs

- The unpaired data adapts each backbone to its sensor, which matters most for thermal because DINOv2 was pretrained on RGB photographs.
- Freezing the backbones keeps the number of weights trained on the small paired set low, reducing overfitting.
- Mixed training keeps the decoder seeing varied data and produces one model that works when a sensor is missing.
- Three stages take longer to run and need checkpoints from two prior trainings.

## Consequences

- Weight loading must take backbone A and backbone B from two different checkpoints.
- The data pipeline needs a weighted mix of three sources and a sensor-present flag.
- Each stage's runs are recorded in `docs/experiments/`.
