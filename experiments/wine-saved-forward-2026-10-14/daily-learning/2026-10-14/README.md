# AI Engineer Daily Report

日期：2026-10-14，Asia/Kuala_Lumpur。执行方式：AI 助理原创实现、真实 CPU 实验与检查；个人独立理解仍待自测。

## 今日学习主题

推进前一天未闭合的问题：相同 logits 和 scalar loss，是否仍可能有不同的 CE 保存前向中间值？先固定协议和原始 fixture，再用公开 saved_tensors_hooks 观察原 CE，同时做匹配上游梯度的公开 log-softmax VJP。

## 今日研究项目

- 项目：[PyTorch](https://github.com/pytorch/pytorch)，PyTorch contributors
- 版本：2.14.1+cpu，源码 revision `5c4886908584029761b579af026dcfb627c84070`
- 阅读：公开 hooks 合同、F.cross_entropy、LossNLL.cpp、derivatives.yaml、SoftMax.cpp、CPU SoftMaxKernel.cpp、LogSoftmaxKernelImpl.h、saved_variable.cpp 和 LICENSE
- 技术价值：区分被保存的前向值、实际反向输入、单独分解计算和 native 内部算术；适合当前 PyTorch/复现工程阶段
- 代码结构：Python API → ATen loss/softmax → 自动微分规则 → CPU dispatch 实现
- 借鉴：保存证据、固定输入、观察者非干扰对照、类型严格验证与失败字节留存；未复制上游代码、未调用私有 backward API

## 今日完成内容

代码：独立 `saved_probe.py`、45 项聚焦测试。保留原始 Oct 13 实现和固定 fixture，未修改旧实验。

实验：新进程 A-B-B-A，AMD EPYC 9V74，有效 capability 为 AVX512 / DEFAULT / DEFAULT / AVX512。每 worker 的三个首步都从相同初始参数重新开始；没有完整 Wine 拟合或测试集评价。

结果：

1. CE 前向保存的四个槽全部逐位相同：两份 16×3 float64、16 个整数标签和一个 scalar。两份矩阵都与单独 log-softmax 的结果严格相同。
2. 原始 logits、loss、log-softmax 输出和 NLL 上游梯度全部逐位相同。
3. 在完全匹配的 logits 与上游梯度下，公开 log-softmax backward VJP 仍精确复现原 CE 的 5/48 个梯度差异，每个 1 ULP，最大绝对差 3.469446951953614e-18。
4. 原实现无 hook → 有 hook → 无 hook 的 19 个阶段全部逐位一致；A 与 A、B 与 B 严格重复。
5. 原首步整体严格比较仍失败；22 个阶段条目超过固定 8 ULP，最大 640 ULP。VJP 本身的 1 ULP 差异没有超预算。这两个结果不能混淆。

完整值、hash、控制和负结果见 `results/abba-initial.json`、`results/findings.json` 与 `docs/observations.md`。验证后复跑单独保留，方法和额外计算见 `docs/post-observation-changes.md`。

## 技术理解

今天增加的是可检查的工程证据：本 fixture 的差异可以进一步收窄到输入匹配的公开 log-softmax backward 操作。没有在已捕获的 CE 保存值中发现前向差异，但这不等于穷尽每个内部临时量。

这仍不是具体微内核缺陷、某一 exp/FMA 实现或数学错误的证明，也不能解释不同宿主上的历史 CI。分解的 log_softmax → nll_loss 是另一条公开计算，不伪装成原 CE 的指令级 trace。公开 hooks 会改变 autograd 机制并跳过通常的版本检查；本例的严格非干扰对照不能推广到任意有原地修改的程序。

遇到的问题：初始 lint 未通过，保留并完成格式修正；检查发现普通 Python equality 允许 bool/int 混淆，以及失败 worker 字节丢失，均已修复并增加回归测试。科学配置与阈值未改，失败数值未删除。

## GitHub 变化

新增仓库：无。新代码、文档、测试与完整结果已在独立交付包准备。Commit：无，未公开发布，也未执行远端 CI。

当前公开基线：工程 `aa56d5e3ae5ffcee428d1f4c996a04497df7bb7e`；成长 `a5cc07e1931b8ad2f118c3d83018b02323d3ba4b`。本增量没有越过既有发布确认，也没有覆盖旧固定输入增量。

## 求职价值分析

对应 AI / ML / Research Engineer 的受控调查、Autograd 理解、可复现证据和失败边界设计。价值在于把可观测结论变窄、将工程通过与科学失败分开，而不是增加仓库或测试数量。它还不构成生产训练、GPU、分布式或研究新贡献的证明。

## 当前能力变化

新增：可运行的 saved-forward 观察器、匹配上游 VJP 实验和严格验证。不足：学习者尚无独立解释/改写证据；native 具体算术路径未追踪；历史 CI 原因未闭合。

## 下一步计划

1. 先做个人自测：推导 NLL mean 的 −1/16，解释为什么 saved-forward 相同与 matched VJP 不同能缩小观察边界，但不能宣称 kernel bug。
2. 若需要继续研究 native 数值路径，先写一个新的、有限的实验问题和预算；不要仅为追逐 ULP 继续扩大工作量。当前阶段的主要缺口仍包括个人独立实现与更广的工程能力。
3. 后续集成时保留现有证据、失败与数据许可，重新跑项目聚合测试/构建，再核对授权范围与精确远端提交的 CI。当前局部测试不替代这些步骤。

## 最终本地验证

45 项聚焦测试通过；Ruff、格式、compileall 与既有环境 pip check 通过。加固后的 A-B-B-A 四个科学 hash 与最初捕获一一严格相同，所有控制继续通过。没有新增安装、项目整包聚合测试、容器执行或远端 CI。

独立审查另有 46 项自行编写的检查通过，覆盖原始证据、比较与异常/格式验证；与主套件的场景存在重叠，不合并为独立证据总数。当前局部交付没有剩余科学或已测工具阻塞。
