# Wine learning curves and validation endpoint selection

## Question and frozen design

Does the earlier 14-update observation persist through a longer fixed budget, and can an endpoint be selected without reading test scores? This is a development experiment, not a fresh generalization benchmark. The old validation split has already informed project development and the historical test set is exposed.

[Plan](wine-learning-curves-plan.json) was written before executing this experiment on 2026-11-06 UTC / 2026-11-07 Asia/Kuala_Lumpur. The local pre-run timestamp is a process record, not an independently timestamped preregistration. No choices were changed after observing results.

All 12 trajectories run 20 epochs: linear/mlp16 × seeds 42/7/123 × constant/step7. Batch16 yields six batches of 16 plus one of 11, seven optimizer updates and 107 visits per epoch. Both arms share initialization and local-generator permutations. Only train rows fit standardization. The existing lr=.05, momentum=.9, weight_decay=.0001 and step7 gamma=.5 remain unchanged. We do not search a better decay point.

Every epoch records full-training and validation mean CE at the same fixed weights, after successful optimizer updates. This is not the average online batch loss collected while weights change. Evaluations use eval/no_grad and restore the previous mode. Candidate epochs are 2/5/10/20; all 16 architecture/arm/epoch candidates retain all three seed values. Select the smallest three-seed mean validation CE, exact ties resolved by earlier epoch, linear then mlp16, constant then step7. No adaptive early stopping. Missing/duplicate runs, incomplete/nonfinite curves or failed training abort rather than silently dropping a seed.

Candidate state_dict values are deep-copied; after training ends each is reloaded and checked against its recorded parameter digest and validation CE. These are trusted in-memory CPU snapshots, not persistent training checkpoints. No claim of optimizer restoration or deployable exported weights is made here.

## Reproduction

Use the repository's locked CPU installation, then:

```sh
python -m pytorch_lab.wine_learning_curves_cli --output results/my-new-curves.json
python -m pytest tests/test_wine_learning_curves_cli.py tests/test_wine_learning_curves_review.py -q
```

The CLI refuses existing files/directories/symlinks, serializes with nonfinite rejection, creates the output exclusively and prints identical JSON. The source workflow and installed-wheel checker now execute this 24th CLI.

## Actual observations

Full output: [final JSON](../results/2026-11-07-wine-learning-curves-final.json). The initial JSON is also retained: it has the same trajectories but predates dynamic accounting and the checker success field; use the final report for verification.

| Architecture / arm | Mean validation CE at 2 | at 5 | at 10 | at 20 |
|---|---:|---:|---:|---:|
| linear / constant | .161060930 | .078830766 | .057928514 | .039918180 |
| linear / step7 | .199811432 | .161460306 | .158500236 | .158406991 |
| mlp16 / constant | .282925259 | .076402867 | .050723974 | .039579835 |
| mlp16 / step7 | .449934316 | .263129305 | .246839678 | .246381034 |

Selected: mlp16/constant/epoch20; seed CE .05811863818/.01839533841/.04222552839. The linear constant endpoint is only .00033834479 worse in the aggregate. These are descriptive development scores, not evidence of significant superiority. The best observed candidate is the budget boundary; this does not establish convergence or an optimal training duration. No budget extension was chosen after observing this.

step7 at update 140 prepares lr_next=.05×.5^20=4.76837158203125e-8, while that update used 9.5367431640625e-8. Its later trajectory is nearly frozen. This describes a specific aggressive schedule, not the usefulness of scheduling in general.

Actual accumulated work: 1680 optimizer updates, 25680 training visits; 480 curve evaluation calls with 34320 visits; 48 endpoint reload evaluations with 1728 visits. Total 528 evaluation calls/36048 visits. Counts derive from produced traces and were independently matched against actual SGD.step and evaluate calls. Test scoring is absent from this experiment. Existing full-regression delivery checks separately execute the old Wine benchmark, whose scores are not inputs to the new selector.

## Official source study and licenses

Research project: [PyTorch](https://github.com/pytorch/pytorch), maintained by the PyTorch community. Current-stage value is understanding Module state/evaluation semantics rather than adopting another framework. Structure studied: torch/nn/modules/module.py (train/eval, state_dict) and the previously studied torch/optim scheduler interface. Original loop, selector, tests and report; no upstream implementation copied.

- [v2.14.1 Module source](https://github.com/pytorch/pytorch/blob/v2.14.1/torch/nn/modules/module.py): eval delegates to train(False); state_dict values reference model storage, so retaining an endpoint requires a copy.
- [Autograd modes](https://docs.pytorch.org/docs/2.14/notes/autograd.html): eval and no_grad are separate concerns.
- [Saving/loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html): deepcopy for retained best-model state.
- [StepLR](https://docs.pytorch.org/docs/2.14/generated/torch.optim.lr_scheduler.StepLR.html): update order and schedule state.
- [Reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html): reproducibility is not promised across versions/platforms.
- [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) and [common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html): validation selection bias and train-only preprocessing.
- [PyTorch v2.14.1 LICENSE](https://github.com/pytorch/pytorch/blob/v2.14.1/LICENSE): BSD-style terms and component notices. Existing Wine data CC BY4.0 and scikit-learn BSD-3-Clause attribution remain in THIRD_PARTY_NOTICES.md. The repository itself has not adopted an open-source license.

## Verification

[Actual local checks and failures](../results/2026-11-07-learning-curves-verification.md) and [installed-wheel evidence](../results/2026-11-07-learning-curves-wheel-summary.json).

## Boundaries and next work

Single tiny historical dataset, one reused split, three optimization seeds, no data-sampling uncertainty, CPU float64 only. No fresh test, GPU/AMP/DDP, production calibration, exported model package or file-resume claim. The result validates assistant-produced engineering artifacts, not the learner's independent mastery.

Next meaningful extension: package the validation-selected endpoint with its train-only preprocessing and explicit inference schema, and verify reload/prediction parity and malformed-input failure. Freeze that interface before implementation; do not add another schedule sweep just to improve the current validation score.

The selected item is a configuration aggregated across three seeds, not a particular seed model. Before export, freeze the seed or explicitly define an ensemble; do not choose the best seed using the same validation set.
