# Wine update-clock verification — 2026-11-06

Date: Asia/Kuala_Lumpur 2026-11-06; execution started on 2026-11-05 UTC.

## Baseline and scope

Read the confirmed portfolio task, current roadmap, experiments and latest daily report before work.
Engineering main was `feb37e8f502f0ac6e17d80acc0d1113236761976`, with no open PRs.
All 391 baseline file blobs were restored and matched before edits. The new work is Wine's optimizer-update
LR comparison, not a repeat of the earlier synthetic epoch StepLR. There are no deletions or unrelated
repository changes. The protocol was written before implementation/execution.

## Scientific evidence

`PYTHONPATH=src python -m pytorch_lab.wine_update_schedule_cli --output results/2026-11-06-wine-update-schedule.json`
completed successfully. Raw JSON SHA-256:
`11e08802e885e8e80b1a356d465beb70dd7b9800bb3534548dc6c02c40208d5b`.

24 comparison trajectories and 36 restored suffixes used 588 optimizer updates / 12,486 training
sample visits. There were 60 endpoint evaluations / 6,420 evaluation visits, all train-only. The
12 configurations passed every LR formula, matched-batch, budget and complete-state restoration check.
All constant-rate 14-step endpoint train CEs were lower than their decayed counterparts; no universal
ranking or heldout/generalization conclusion follows. The next LR after update 14 was not used.

The independent review compared all 12 constant-rate paths with the existing batch-budget implementation,
used a separate explicit SGD recurrence for all 168 scheduled updates, poisoned heldout rows, guarded
forwards and counted actual work. Restoration is only trusted same-input in-memory snapshots into
fresh objects in the same process. No new-process, file durability, arbitrary model, GPU, AMP or DDP claim.

## Local tests

Final full source suite: **1926 passed, 1 warning, 576.06 seconds (9m36s)**.
Together with the standalone 59 / 45 / 34 suites, **2064 unique tests passed**.
The 39 new tests are already included in the 1926 main-suite count.

Three standalone suites: 59 passed (8.33 seconds), 45 passed (12.67 seconds), and 34 unittest tests
passed (1.993 seconds). Final focused tests: 39 passed (8.85 seconds), including the one independent test;
these are already contained in the main suite, not additional unique tests.

Ruff lint and format checks passed; compileall and pip check passed. Python 3.12.14,
Torch 2.14.1+cpu, pytest 9.1.1 and Ruff 0.16.10. The optional missing-NumPy warning remains.

The first full-suite attempt was stopped after inspection found that its reused development interpreter's
editable installation still pointed to the older checkout. PYTHONPATH covered ordinary tests but isolated
`-I` subprocesses would use the wrong source. That incomplete run is not counted as a pass. The editable
binding was corrected, its `python -I` package path was verified against the final checkout, and the
whole suite was restarted with `python -I -m pytest -q`. No source or numerical tolerance was changed
for this environment correction.

An initial unittest invocation from the project root could not import its standalone module. It ran no
valid suite; rerunning from `tools/wine-delivery`, as required by its workflow, passed 34 tests. Initial
new-code lint found long-line / explicit strict-zip issues; corrected and rechecked. The independent
review's initial wheel-only interpreter had no pytest, so it used the development interpreter instead.

## Installed-wheel evidence

A fresh virtual environment installed 10 existing runtime wheels offline with the repository's exact
hash locks, then installed the newly built noneditable project wheel without dependency resolution.
55 package files were byte-identical between final source, wheel archive and installed directory.
The delivery checker actually ran all 23 installed CLIs from unrelated temporary directories with `-I`;
all passed. The complete new experiment report was exactly equal to source output in this host.
[Summary](2026-11-06-update-schedule-installed-wheel-summary.json) retains wheel and full local-report
SHA-256 identities and the command list. Old full Wine benchmarking is separate and includes heldout
scoring; those outputs did not select the new experiment's configuration.

No full pytest was run in the installed environment and no local Docker was run. CI is configured to
run full source suites, 23 source CLIs and 23 CLIs in the actual nonroot/read-only/offline container.
The source test counts must not be presented as container test counts.

## Publication

This record describes verified local work before publication. The engineering commit and exact remote
CI evidence are recorded in the learning repository's 2026-11-06 report after actual verification.
A local pass is not evidence of a pushed commit or green remote workflow.
