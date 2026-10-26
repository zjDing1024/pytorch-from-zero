# Integration plan and scope

This pack is local/private delivery evidence. No branch, commit, PR, release or repository file was created or changed. Earlier publication approval remains pending and was not retried.

Verified public baselines supplied for this audit:

- [pytorch-from-zero `aa56d5e3ae5ffcee428d1f4c996a04497df7bb7e`](https://github.com/zjDing1024/pytorch-from-zero/commit/aa56d5e3ae5ffcee428d1f4c996a04497df7bb7e)
- [ai-learning-roadmap `a5cc07e1931b8ad2f118c3d83018b02323d3ba4b`](https://github.com/zjDing1024/ai-learning-roadmap/commit/a5cc07e1931b8ad2f118c3d83018b02323d3ba4b)

The route, experiment index and latest published daily report (Oct 11) were read at the roadmap baseline. Later standalone deliveries are not presented as already integrated GitHub work.

## After explicit authorization covers publication

1. Re-read current remote heads and reconcile any new work. Do not overwrite the separately frozen pending increment or its manifests.
2. Decide the integration order of fixed-input, first-step and saved-forward increments. Preserve their inputs, plans, hashes and original failure evidence.
3. Place reusable delivery checks under the project's tooling/tests, adapting the explicit artifact contract rather than blindly copying hard-coded archive pins into a generic utility.
4. If the experiments become package modules, declare package data and entrypoints; build and test the installed wheel from outside the source tree, including missing-data negative controls.
5. Run the full current project suite, lint, all supported CLIs and the actual container workflow where available. The current 59/45 tests are standalone focused suites, not the full engineering repository regression.
6. Record the Chinese daily report and source study in the roadmap, linking exact released artifacts/commits only after they exist. Verify remote commit and CI before claiming publication success.

## Boundaries retained

- No new model, training protocol, threshold, random seed, split or test-set evaluation
- No dependence upgrades, new heavyweight downloads, paid resources or container-image publication
- No diagnosis of a native kernel defect or historical CI cause
- No inferred learner mastery
- No repeat numerical investigation required to close this delivery task

Useful next learning step: independently explain why source tests, fresh extraction, installed-wheel checks and scientific equality answer different questions, then reproduce the delivery validator on an appropriate existing CPU environment.
