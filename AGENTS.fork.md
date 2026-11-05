# RF-DETR Cross-Attention Fork - Agent Instructions

This repository is a private fork of [roboflow/rf-detr](https://github.com/roboflow/rf-detr) for dual-sensor (thermal + RGB) fusion. Read [AGENTS.md](AGENTS.md) first: everything there applies here. The rules below add to it, and where they are stricter, they take precedence.

`AGENTS.md`, `CLAUDE.md` and `CHANGELOG.md` are kept identical to upstream so that upstream merges never conflict in them. Put fork-specific guidance in this file, never in those three.

## Agent Responsibilities

1. **Following test-driven development practices**

    - Follow the strict RED → GREEN → REFACTOR workflow below for every new feature and bug fix
    - Ensure final PR commit has all tests passing

2. **Keeping the fork mergeable with upstream** (see [Working with Upstream](#working-with-upstream))

## Testing

### Commands

In addition to the commands in `AGENTS.md`:

```bash
# Run a single test by pytest node ID (substitute the actual path and name)
uv run --no-sync pytest 'tests/path/test_module.py::test_case'

# Type-check with the same mypy hook used by the project
pre-commit run mypy --all-files
```

### Strict Test-Driven Development

1. **RED.** Before writing implementation code for any feature or bug fix, write a focused pytest test that specifies the expected behaviour or reproduces the bug. Run the test by node ID and confirm it fails for the expected reason. A collection error or an unrelated dependency or device failure does not satisfy RED.
2. **GREEN.** Make the smallest implementation change that satisfies the test. Rerun the focused test and the affected tests.
3. **REFACTOR.** Clean up the implementation while preserving behaviour. Rerun the affected tests after each refactor, then run the CPU suite and any applicable GPU tests before opening the PR.

**Python data science tests**

- Use small, deterministic fixtures such as synthetic images, annotations, or tensors. Prefer CPU tests when the behaviour is device-independent. Control randomness locally and use explicit, justified tolerances for floating-point comparisons.
- Assert the behaviour that matters to callers. For data loading and transforms, check image–annotation alignment, shapes, dtypes, and relevant edge cases such as empty annotations. For model changes, use tiny forward or backward passes to check output contracts, finite losses, or gradient flow as appropriate.
- When changing data splitting or evaluation, test for leakage and verify metrics on inputs with known expected results. Keep full training runs and benchmarks separate from the fast test that proves the change.

**Forbidden actions**

- Never write implementation code before running the new test and confirming it fails for the expected reason.
- Never weaken or rewrite existing assertions just to make broken code pass.
- Never skip, mark as expected to fail, or delete a failing test to bypass a failure.

## Documentation Required for Each Change

Keep documentation in the same PR as the change. A reviewer should be able to understand what changed, why it changed, how to use it, and how to reproduce any reported result from the repository records.

| Record                  | When to update it                                                                                               | What to include                                                                                                                                                                                                                                                                                                                                                                                 |
| ----------------------- | --------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Changelog               | Any user-visible change to behaviour, APIs, configuration, dependencies, or compatibility                       | Add an entry under `[Unreleased]` in `CHANGELOG.fork.md` (not upstream's `CHANGELOG.md`) using the same categories as `CHANGELOG.md`. Describe the user impact and any migration or compatibility implications.                                                                                                                                                                                 |
| Decision log            | A material choice about architecture, data preparation, dataset splits, model design, evaluation, or deployment | Add or update a dated record under `docs/decisions/`. Include the context, options considered, rationale, trade-offs, consequences, and status. Link to the PR and supporting experiments. Link superseded decisions to their replacements.                                                                                                                                                     |
| Experiment log          | A run used to evaluate a hypothesis or support a claim about model quality, data processing, or performance     | Add or update a dated record under `docs/experiments/`. Include the hypothesis, baseline, code revision, dataset version and split, preprocessing, configuration and seeds, environment and hardware, exact run and evaluation commands, metrics and results, and conclusions with limitations. Record failed and inconclusive runs that affect the decision. Link to any supporting artifacts. |
| User and technical docs | A change to how someone configures, runs, extends, or maintains the project                                     | Update the relevant README, guide, examples, or API docs. Cover prerequisites, inputs and outputs, defaults, and migration steps where applicable. Keep docstrings and type hints aligned with the implementation.                                                                                                                                                                              |

Use versioned dataset and artifact references for reproducibility. Keep credentials, sensitive samples, and large run artifacts out of Git. Before opening a PR, check that commands and links are current and list the documentation records updated in the PR description. If a record does not apply, briefly explain why.

## Working with Upstream

### Branches

| Branch                                   | Purpose                                                                                                                      |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `develop`                                | Exact mirror of upstream `develop`. Only updated by syncing; never commit to it.                                             |
| `fusion` (default)                       | Fork integration branch. Receives upstream by merging `develop`.                                                             |
| `feat/…`, `fix/…`, `docs/…` off `fusion` | Fork work, merged into `fusion` through pull requests.                                                                       |
| `fix/…`, `feat/…` off `develop`          | Changes that could be contributed upstream. Based on upstream code only; also merged into `fusion` so the fork can use them. |

Upstream accepts pull requests only from a GitHub fork of `roboflow/rf-detr`. To contribute, push the branch to such a fork and open the pull request there, following `.github/CONTRIBUTING.md` (CLA, PR template, issue first for new models or significant features). Fork-only files (this file, `.claude/CLAUDE.md`, `CHANGELOG.fork.md`, `plans/`, `docs/decisions/`, `docs/experiments/`) must never appear in a branch meant for upstream.

### Syncing

```bash
git fetch upstream develop
git push origin upstream/develop:develop   # fast-forward the mirror
git checkout fusion && git merge develop
```

### Avoiding conflicts

- Put fork-only content in new files. Do not edit upstream files unless the change itself requires it.

- `AGENTS.md`, `CLAUDE.md`, `CHANGELOG.md`, `.pre-commit-config.yaml`, `mkdocs.yaml` and `.github/workflows/` stay identical to upstream. Check before opening a PR:

    ```bash
    git diff --stat develop -- AGENTS.md CLAUDE.md CHANGELOG.md .pre-commit-config.yaml mkdocs.yaml .github/workflows/
    ```

    The output must be empty.

- Where fork code must change upstream modules (for example `models/lwdetr.py`), keep the edit small and put the new behaviour in new modules, behind a config option, so merges stay small.

- GitHub Actions is disabled in the repository settings, so upstream workflows do not run here.

## Development Environment

The cloud sessions used for this fork cannot reach `download.pytorch.org` or Hugging Face. Install PyTorch from PyPI instead of with `--torch-backend=cpu`. This pulls the CUDA build (it runs on CPU too) and its NVIDIA libraries, about 4 GB; the install took about 35 minutes in a cloud session.

```bash
uv venv
uv pip install -e ".[train,augment,cli,visual,data]" --group tests
```

The `mypy` pre-commit hook runs `uv run --group typing`, whose PyTorch source is `download.pytorch.org`, so it cannot run in such sessions. Run the same check from the project environment instead, and run the other hooks with the hook skipped:

```bash
uv pip install "mypy==1.19.1" "numpy<2.4" types-PyYAML types-requests types-tqdm
.venv/bin/mypy src/rfdetr/ scripts/ --no-error-summary
SKIP=mypy pre-commit run --all-files
```

The `typing` group pins `torch==2.14.0`; the PyPI install may be newer. If mypy reports errors in code you did not change, check whether they come from that version difference before fixing them.

Tests that download pretrained weights or the DINOv2 snapshot fail in such sessions; this is an environment limit, not a test failure to fix in code.

**Known baseline in cloud sessions.** On upstream `487a9bb`, with PyTorch 2.15 from PyPI and no GPU, the CPU suite (`-n 4 --timeout=240`) gave 10485 passed, 469 skipped, 4 failed and 51 errors in 27.5 minutes. None of the 55 are code faults:

- 53 timeouts in `tests/export/tensorrt/test_tensorrt_int8.py`, `tests/training/test_lightning_checkpoint_reload.py` and `tests/training/test_resume_optimizer_settings.py`: slow CPU. With `--timeout=1800` all 723 tests in those files pass (6.5 minutes at `-n 2`).
- `tests/training/test_fused_adamw_ema_routing.py::TestFusedAdamWEMARouting::test_combined_checkpoint_keeps_fp32_ddp_destination_unfused` and `tests/export/tensorrt/test_tensorrt_export.py::TestPortableEngines::test_the_lean_runtime_is_not_probed_unless_version_compatible_is_asked_for[ampere_plus]`: the PyPI CUDA build of PyTorch on a machine without an NVIDIA driver.

Use `--timeout=1800` for those three files. Any failure outside this list needs investigating.
