# 0002: Paired data, thermal reference frame and dataset splits

- **Date:** 2026-11-04
- **Status:** Accepted
- **Links:** `plans/dual-sensor-fusion.md` (sections 1 and 5), [0003](0003-dual-backbone-cross-attention-fusion.md), [0004](0004-staged-training.md)

## Context

Three data sources are available, all labelled with classes 1 and 2 and bounding boxes:

- many thermal-only images;
- many RGB-only images, captured at different times and places from the thermal set;
- a limited set of simultaneous thermal + RGB captures, taken from the camera's split-screen view. The camera software aligns the two views for display, and the captures have already been split into two aligned images per pair.

## Options considered

- **Reference frame for labels and outputs:** thermal or RGB. Because the pairs are aligned, either works geometrically.
- **Extra pairs:** build synthetic pairs from unrelated thermal and RGB images, or use only real simultaneous pairs.
- **Splits:** random by frame, or grouped by capture session or location.

## Decision

- Thermal is the reference frame for labels and model outputs.
- Only real simultaneous captures are used as pairs. Unpaired images are used as single-sensor data (see [0004](0004-staged-training.md)).
- All thermal images, standalone and paired, are converted to one representation: single-channel grayscale with one polarity (white-hot or black-hot), repeated to 3 channels, with thermal-specific normalisation.
- Train, validation and test splits are grouped by capture session or location. A held-out paired test set is kept for all model comparisons.

## Rationale and trade-offs

- Thermal still works at night and in smoke, so its frame is the one a deployed system can always rely on.
- Synthetic pairs would teach the model that the other sensor is unrelated noise, so it would learn to ignore the fusion.
- Paired thermal images come from a display view and may use a colour palette and a lower resolution than standalone thermal images; one representation removes that domain gap.
- Frames from one session are near-duplicates; random splits would inflate test scores.
- The camera's alignment is a display alignment, probably calibrated for one distance, so small offsets may remain at other distances. [0003](0003-dual-backbone-cross-attention-fusion.md) is designed to tolerate them.

## Consequences

- Phase 0 of the plan: thermal conversion, a paired COCO JSON with boxes in the thermal frame, session-grouped splits, and a visual check of boxes on a sample of pairs.
- Tests for the data pipeline must check for split leakage and image–annotation alignment on both sensors.
