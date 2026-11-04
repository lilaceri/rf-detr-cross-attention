# 0001: Fork branch layout and keeping upstream files untouched

- **Date:** 2026-11-04
- **Status:** Accepted
- **Links:** [#1](https://github.com/lilaceri/rf-detr-cross-attention/pull/1) (restore upstream workflows), the PR that adds `AGENTS.fork.md`

## Context

This repository is a private copy of `roboflow/rf-detr`, created to build a dual-sensor model. We want to keep taking upstream updates and to be able to contribute general fixes back. Upstream accepts pull requests only from a GitHub fork of `roboflow/rf-detr`, so contributions must be separable from fork-only work. Upstream also changes `AGENTS.md`, `CHANGELOG.md` and its workflows often, so fork edits to those files would conflict on most merges.

## Options considered

1. **Work directly on `develop`.** Simplest, but mixes fork work with upstream history and makes clean upstream contributions hard.
2. **`develop` as an exact upstream mirror, a separate integration branch for fork work.** Contributions branch off the mirror; fork work branches off the integration branch.
3. **Install rf-detr as a package and keep only fork code.** No merges at all, but the fusion change needs edits inside the model, which a package dependency cannot make cleanly.

## Decision

Option 2:

- `develop` mirrors upstream `develop` and is only ever synced.
- `fusion` is the default branch and receives upstream by merging `develop`.
- Fork work goes through `feat/`, `fix/` and `docs/` branches and pull requests into `fusion`.
- Changes that could go upstream branch off `develop` and are merged into `fusion` as well.
- GitHub Actions is disabled in the repository settings rather than by moving `.github/workflows/`.
- Fork-only guidance and records live in new files: `AGENTS.fork.md` (loaded for Claude Code by `.claude/CLAUDE.md`), `CHANGELOG.fork.md`, `plans/`, `docs/decisions/` and `docs/experiments/`. `AGENTS.md`, `CLAUDE.md`, `CHANGELOG.md`, `.pre-commit-config.yaml`, `mkdocs.yaml` and `.github/workflows/` stay identical to upstream.

## Rationale and trade-offs

- Upstream merges conflict only where fork code genuinely changes upstream modules.
- Contributions start from pure upstream code, so they can be pushed to a public fork unchanged.
- `.claude/` is ignored by upstream's `.gitignore`, so `.claude/CLAUDE.md` is committed with `git add -f`. Files added to `.claude/` later are not tracked unless they are added the same way.
- Agents that read only `AGENTS.md` (for example Copilot) do not see the fork rules unless pointed to `AGENTS.fork.md`.
- `docs/decisions/` and `docs/experiments/` sit inside the mkdocs source but are not in its navigation. A docs build includes them as unlisted pages; excluding them would mean editing `mkdocs.yaml`, which this decision keeps identical to upstream.

## Consequences

- Before each PR, `git diff --stat develop -- AGENTS.md CLAUDE.md CHANGELOG.md .pre-commit-config.yaml mkdocs.yaml .github/workflows/` must be empty.
- Fork changelog entries go in `CHANGELOG.fork.md`.
