# Batch budget: actual local verification

Complete main suite **1852 passed,397.75seconds**, plus standalone59passed4.88s,
45passed5.24s,34unittest1.023s: **1990 total**,40new tests(13core20CLI7independent).
Focused33 and independent7 are included, not added again. Optional missing NumPy
warning remains; experiments do not depend on NumPy.

Ruff lint and final187-file format check passed (prebuild186), compileall and
pipcheck passed. Fresh offline noneditable wheel used10hashlocked dependencies,
executed21actual installedCLIs from unrelated working directories with isolated
Python;51package members match source/wheel/install byte-for-byte. Installed batch
JSON equals local source raw JSON in this same runtime. No local Docker or installed
fullpytest execution. Remote exact-commit checks must be recorded separately.

Main scientific evidence:18unique runs,216optimizer updates,4878training visits,
24fulltrain evalforwards/2568eval visits;5controls pass in all6configurations.
Equal-visits batch16 endpointCE lower6/6;equal-update batch32 lower4/6 but consumes
more examples. Neither comparison is compute-time matched or pure noise attribution.
Independent actual tensor tests validate handwritten SGD against all18trajectories
at1e-14 and guard poisoned-heldout isolation with real counts.

Frozen plan SHA256 `4f6ce0c6ff33edff8d3b9eb71e5aaba6be0f022ae03691853359cba09619609f`.
Raw evidence SHA256 `e39458c96b3dba47eb16272841090587720d4d9a8adc71950d88c0b4f9b062f4`.

Development record: initial unscoped lint included the historical frozen source
snapshot, which is excluded by the documented workflow; the correct workflow
exclusion passed and no historical source changed. Reviewer closure bindings were
fixed before freeze. Initial standalone delivery invocation from repo root failed
to import validate_delivery; running from its documented tools/wine-delivery cwd
passed34tests. No test removed, no numerical tolerance or budget relaxed. Existing
command count/list assertions were updated for21commands before the full run.
A finite initial/final CE guard was added before aggregate testing; raw outputs
are unchanged by the guard. Complete main passed on its first full run.
