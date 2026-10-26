# Source study: testing the delivered artifact

Research date: 2026-10-15, Asia/Kuala_Lumpur. This task addresses delivery reliability in stage 1 of the AI portfolio, without changing model/data/evaluation protocols.

## pytest

- Project: [pytest](https://github.com/pytest-dev/pytest), maintained by pytest-dev and contributors; created by Holger Krekel.
- Problem: reproducible discovery, execution and reporting of Python tests.
- Structure: `src/_pytest/` for implementation, `testing/` for its own tests, `doc/en/` for guidance.
- Official guidance read: [Good Integration Practices](https://docs.pytest.org/en/stable/explanation/goodpractices.html).
- Versioned implementation read: [9.1.1 `src/_pytest/pathlib.py`](https://github.com/pytest-dev/pytest/blob/9.1.1/src/_pytest/pathlib.py), especially `ImportMode`, `import_path` and `ImportPathMismatchError`.

The import implementation distinguishes prepending/appending import paths from importlib loading. The guidance explains why testing an installed package can reveal defects that source-tree tests conceal. This audit therefore runs the actual ZIP contents in fresh directories, clears inherited import injection and tests an unrelated working directory. It preserves the original standalone scripts' pytest configuration. A future installable project should separately test its built distribution; this ZIP audit cannot replace that check.

License: [pytest 9.1.1 MIT license](https://github.com/pytest-dev/pytest/blob/9.1.1/LICENSE). Source was studied, not copied; no pytest code is redistributed by the new validator.

## PyTorch

- Project: [PyTorch](https://github.com/pytorch/pytorch), maintained by PyTorch contributors and its foundation ecosystem.
- Problem: tensor computation, autograd and backend execution.
- Relevant structure: `torch/` Python APIs, `aten/` compute implementations, `test/`, `docs/`.
- Official guidance read: [PyTorch 2.14 reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

The guidance limits reproducibility guarantees across releases/platforms and distinguishes controlling randomness from choosing deterministic operations. Consequently this delivery report keeps command/control success separate from byte equality. Today's CPU description differs from the saved experiment's, so mismatched hashes remain an observed reproduction limit. The audit makes no new numerical-cause claim.

License read: [PyTorch root LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE), a BSD-style notice with source/binary notice and non-endorsement conditions. The installed build's revision remains `5c4886908584029761b579af026dcfb627c84070`, as recorded in scientific captures. This study does not claim the mutable main license URL is a versioned snapshot of that revision. Historical pinned-source/license notes are preserved in the unchanged experiment archives. No PyTorch implementation or binary is copied into the new pack.

## Python packaging guidance

The [Python Packaging User Guide's packaging tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/) distinguishes project metadata/build configuration from arbitrary source folders. These experiment archives contain scripts and tool configuration, not complete distribution metadata. This audit documents their supported script entrypoints and does not invent a wheel-install claim.

## Original engineering application

The new code independently implements: hash-pinned input selection, bounded ZIP preflight, exact manifest coverage, non-overwriting extraction, supported report-envelope/hash checks, command supervision, post-run original-file checks and explicit reproduction comparison. Tests cover corrupt/ambiguous archives, wrong scientific identities, truthy non-Boolean flags, missing controls, nonfinite JSON, command failure/timeout and retained error receipts. The scientific tensor calculations and 8-ULP thresholds remain in their original artifacts unchanged.
