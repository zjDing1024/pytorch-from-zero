# Executed validation and limits

Prepared 2026-10-13, Asia/Kuala_Lumpur. Status: standalone local investigation complete; integration/publication incomplete.

- Implementation tests: 59 passed in 4.54 seconds.
- Independent execution: 72 passed in 4.86 seconds (the same 59 plus 13 additional contract checks).
- Ruff and formatting: implementation and tests passed.
- compileall: implementation and tests passed.
- Existing-environment pip check: no broken requirements found. No fresh install is claimed.
- Independent byte verification: frozen TRAIN preprocessing/batch/initial weights match the existing Wine recipe, and the original default CPU SGD reproduces the recorded A1 parameter gradients/buffers/update exactly.
- Independent repeated math.nextafter traversal reproduced all reported ULP distances.
- Plan, frozen inputs, original source and final source are retained. Both complete A-B-B-A captures are included; scientific hashes match by condition across initial/final code.
- No prior protected project files or publication manifests changed.

The initial capture preceded validation-only hardening; see docs/post-observation-changes.md. No model operation, input, seed, batch, optimizer parameter, stage order or tolerance was changed after observing results.

Not run for this increment: full-project integration suite, newly installed wheel, Docker or exact-commit remote CI. No GitHub commit or publication occurred. The earlier fixed-logits publication remains pending its existing approval.

The final standalone source SHA256 is 1918f6462fb00feb2844eeb9946bbf11e17358577cbb270f10a55977b165063e.

This package includes user-facing code, data/protocol, complete scientific evidence, source/license notes and daily report. Temporary execution files and internal audit work files are omitted.
