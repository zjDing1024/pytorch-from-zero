# Post-observation changes and verification work

The prospective scientific protocol, fixture, original step reference, conditions and 8-ULP comparison budget are unchanged. The primary A-B-B-A capture used the source preserved as `source-before-first-capture.py`. Its SHA256 is `75b9836635d06378e411063d653e1624878cc8a9ef5eebbb4ca974f37fda1ab6`.

After that observation:

1. Formatting/import cleanup addressed initial lint errors; the original lint diagnostics are retained. No numerical calculations changed.
2. Cached control/relation validation changed to canonical type-exact comparison. This rejects JSON true→1/1.0 and saved index [0,1]→[false,true] substitutions that ordinary Python equality accepted.
3. Failed-worker reporting now preserves stdout/stderr byte count, full-stream SHA256, a bounded exact base64 byte prefix, truncation status and return code when available. The prefix limit is 2 MiB per stream. Previously only failure category and already-completed samples survived. The full stream is collected before this bound; it is not a hard resource sandbox. No environment dump is recorded.
4. Focused tests cover these regressions in addition to fixture/source integrity, observer parity, mutation detection, exact upstream, ULP boundaries, bad schemas, failures, no-overwrite behavior and tiny timeout.

The planned scientific capture is four workers. Additional numerical work is verification, not new conditions or independent scientific evidence: the first 38-test run used one A worker; the final 45-test run used one A worker; a final A-B-B-A rerun checks the hardened implementation against the unchanged initial scientific hashes. Each full numerical worker has the same fixed three independent original/observed first steps plus decomposition and matched VJP. Unit tests also include bounded scalar observer examples and an analytical sanity calculation. No fixture search, full fit, test-set tuning or dependencies were added.

Primary numerical failures and negative observations remain in the initial capture. A verification rerun is never used to replace them or select a more favorable result. Test counts are engineering checks, not statistical sample sizes or a learner-mastery score.

## Final verification

Final code SHA256: `3362f69e70217fd2c48359dda01a8e1166d39b74201c75ad0ce6370f8e9f0426`. The final A-B-B-A verification passed every capture control; each of its four scientific hashes equals the corresponding primary-capture hash exactly. All 45 focused tests, Ruff check, Ruff format check, compileall and the existing environment pip check passed. Complete records are in `results/abba-reviewed.json` and `results/verification.json`. No new installation, package integration, full-project test run, container run or remote CI is claimed.
