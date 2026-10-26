# Evidence scope

`delivery-report.json` is the final release-validator execution, with its exact source SHA. `first-step/replay.json` and `saved-forward/replay.json` are complete new capture outputs copied byte-for-byte. Original capture outputs remain inside the unchanged input ZIPs.

The 34 validator tests, 59 first-step tests and 45 saved-forward tests are distinct reported suites; intermediate reruns are not added to the counts. Two earlier intermediate full executions and one review-ready full execution were performed while hardening the delivery harness, in addition to the final release execution. They used unchanged scientific scripts and inputs; they are not new scientific interventions.

New exported stdout/stderr logs replace only absolute audit-environment and audit-directory prefixes with placeholders. Numerical values, test outcomes and timing are preserved. Original input archives and raw scientific JSON are not redacted or modified. Original local logs are retained separately.

The environment was reused, not newly installed. `pip-check.txt` is dependency consistency only. Scientific equality flags are false even though delivery command/control checks pass.

Initial lint formatting issues and missing fail-closed validation around archive identity/report controls were fixed before final execution. These changes affect only the new validator/tests, not prior plans, fixtures, thresholds or scientific code.

## Public-log privacy correction — 2026-10-26

The two exported `pytest.stdout.txt` logs also contained one relative local environment prefix each. Those two prefixes are now replaced with the existing `<audit-environment>` placeholder. Test results, warnings, numerical values and timings are unchanged. This is a publication-only redaction, not a test rerun. The original logs remain in the private delivery archive. `SHA256SUMS.json` now describes these sanitized public files and this scope note; input ZIPs, scientific JSON, source code and historical scientific hashes are unchanged.
