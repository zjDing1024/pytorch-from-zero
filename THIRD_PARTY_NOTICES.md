# Third-party data and notices

The original Python implementation in this repository does not copy scikit-learn or PyTorch implementation code. It does redistribute an explicitly attributed dataset:

## UCI Wine dataset

Aeberhard, S. & Forina, M. (1992). Wine [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5PC7J

Source: https://archive.ics.uci.edu/dataset/109/wine
License: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/

The original dataset is based on wine chemical analyses described by Forina et al. in the UCI metadata. Dataset creator names, source, and license are retained for attribution, without implying endorsement.

The exact distributed CSV comes from scikit-learn commit e316dbeeebfd8f38cf293d6443ce81aaa33686d3, sklearn/datasets/data/wine_data.csv. scikit-learn added a metadata header, moved class labels to the last column, and encoded them as 0/1/2 instead of UCI's 1/2/3. No further changes were made to the CSV. Full machine-readable provenance and hashes are in src/pytorch_lab/data/wine_provenance.json.

The upstream scikit-learn BSD-3-Clause notice is retained verbatim in src/pytorch_lab/data/LICENSE.scikit-learn. It is distinct from the UCI dataset license. Dataset provenance is based on UCI's recommended citation rather than inconsistent historical fields in scikit-learn's description file.


## 2026-10-10 runtime source study

Original runtime probe/checker/Dockerfile implementation; no pip or Python Official Image implementation was copied. Source study: PyPA pip (MIT), commit `01857ef79f59a98db592bacb6e7b48f354528c80`; Docker Community Python Official Image source (MIT), commit `688a0b86bb44289df16a363e9f41d90514c1a5f9`. Links and design lessons are in [runtime documentation](docs/runtime-reproducibility.md). Installed third-party wheels and the base image carry their own licenses; their installed license metadata is retained. Existing UCI Wine attribution and dataset/software license files remain unchanged.

## 2026-10-11 数值诊断源码研究

阅读PyTorch固定revision `5c4886908584029761b579af026dcfb627c84070`的`aten/src/ATen/native/DispatchStub.cpp`、`torch/testing/_comparison.py`和根LICENSE。上游根许可证为BSD-3-Clause-style条件，各依赖仍保留各自许可。新增比较器、捕获器和测试均为原创，没有复制上游代码或引用私有比较API。官方来源和学习范围见`docs/numerical-regression.md`。

## 2026-10-12 固定输入源码研究

阅读同一PyTorch revision `5c4886908584029761b579af026dcfb627c84070` 的CPU SoftMaxKernel、LogSoftmaxKernelImpl、LossNLL和double向量exp/log入口；参考链接与明确边界见[固定输入文档](docs/fixed-input-diagnostics.md)。代码、dyadic logits fixture和测试为原创，无上游代码复制。PyTorch根BSD-3-Clause-style条件与依赖各自许可不变；本次未重新分发上游实现，也未改变Wine署名/许可。


## TorchData stateful iterator study (2026-10-29)

Read-only design study of Meta/PyTorch TorchData v0.11.0, revision
`377e64c1be69a9be6649d14c9e3664070323e464`:
[stateful sampler source](https://github.com/meta-pytorch/data/blob/377e64c1be69a9be6649d14c9e3664070323e464/torchdata/stateful_dataloader/sampler.py),
[StatefulDataLoader](https://github.com/meta-pytorch/data/blob/377e64c1be69a9be6649d14c9e3664070323e464/torchdata/stateful_dataloader/stateful_dataloader.py),
and [BSD-3-Clause LICENSE](https://github.com/meta-pytorch/data/blob/377e64c1be69a9be6649d14c9e3664070323e464/LICENSE).
No source, comments, or snippets copied; no torchdata runtime dependency added.
The original bounded index-stream design and its limits are in
[the study note](docs/shuffled-index-recovery.md). This reference does not license this repository.


## Joint training recovery source study (2026-10-30)

Read-only official-source study of TorchData's stateful loader (BSD-3-Clause,
revision `377e64c1be69a9be6649d14c9e3664070323e464`) and Hugging Face Accelerate's
`src/accelerate/checkpointing.py` (Apache-2.0,
revision `90b233c86202a5ded4bf66407e75222c5fb007fc`). Exact source/license links,
responsibility boundaries and our independent safe-point design are in
[joint training recovery](docs/joint-training-recovery.md). No upstream code,
comments or source excerpts copied; no TorchData or Accelerate runtime dependency
added. Existing UCI data attribution and historical scientific records remain unchanged.

## CPU stochastic replay study (2026-10-31)

Official PyTorch v2.14.1 source study (revision
`5c4886908584029761b579af026dcfb627c84070`, BSD-3-Clause-style root license)
and Hugging Face Accelerate checkpointing study (revision
`90b233c86202a5ded4bf66407e75222c5fb007fc`, Apache-2.0).
Exact source/license links and version distinctions are recorded in
[the RNG boundary note](docs/dropout-rng-boundary.md).
Probe code, fixture, tests and explanations are original; no upstream snippets,
comments or implementation copied, and no new framework dependency added.
