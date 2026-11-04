# Decision log

Dated records of material choices about architecture, data preparation, dataset splits, model design, evaluation and deployment in this fork. The rules for when to write one are in `AGENTS.fork.md`.

| ID                                                      | Date       | Decision                                                 | Status   |
| ------------------------------------------------------- | ---------- | -------------------------------------------------------- | -------- |
| [0001](0001-fork-branching-and-upstream-sync.md)        | 2026-11-04 | Fork branch layout and keeping upstream files untouched  | Accepted |
| [0002](0002-paired-data-and-thermal-reference-frame.md) | 2026-11-04 | Paired data, thermal reference frame and dataset splits  | Accepted |
| [0003](0003-dual-backbone-cross-attention-fusion.md)    | 2026-11-04 | Dual backbone with bi-directional cross-attention fusion | Proposed |
| [0004](0004-staged-training.md)                         | 2026-11-04 | Staged training with single-sensor detectors first       | Proposed |

**Status values:** Proposed (chosen, not yet backed by an experiment), Accepted, Superseded (link to the replacement), Rejected.

## Template

```markdown
# NNNN: Title

- **Date:** YYYY-MM-DD
- **Status:** Proposed | Accepted | Superseded by [NNNN](NNNN-....md) | Rejected
- **Links:** PRs, experiments, related decisions

## Context

## Options considered

## Decision

## Rationale and trade-offs

## Consequences
```
