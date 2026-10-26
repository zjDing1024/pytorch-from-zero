# 首步观测记录

日期：2026-10-13，Asia/Kuala_Lumpur。协议：`wine-first-step-dispatch-v1`。

## 输入、顺序与来源

- 计划 SHA256：`5b79ea625d6693b4203a18bddd424cb01e04f3e8e1f14459d5bbefe9afcad951`
- 计划声明时间：2026-10-12T16:00:17+00:00；fixture 冻结时间：2026-10-12T16:03:54.662564+00:00
- fixture 文件 SHA256：`4db70a0c4f889e1a0adb5d3ed8c4c17e0eb36871b4c85e82885672e77fcfddf7`
- 首次捕获开始：2026-10-12T16:04:08.905949+00:00；加固后开始：2026-10-12T16:08:55.725339+00:00
- 固定 Wine TRAIN 107 行；train-only mean/std(correction=0) 只在冻结时计算一次。16 条 minibatch、标准化后的输入和 seed-42 MLP16 初始权重随后均加载固定字节，不在每个干预条件下重算。
- batch 原始行 ID：106, 65, 139, 112, 0, 114, 93, 29, 126, 4, 165, 7, 86, 83, 87, 88。
- 无验证或测试集计算，无架构/seed/学习率筛选、无完整拟合；每个 worker 恰好 1 次更新/16 个样本。

## 同主机四样本

实际 CPU：AMD EPYC 9V74 80-Core Processor。有效 capability：AVX512 / DEFAULT / DEFAULT / AVX512。请求/有效值分别保存。runtime/native package hashes、CPU/affinity 和非干预环境控制一致，同条件严格重复；全部六项捕获检查通过。

A 科学 SHA256：`1da7fc32436d2aaf4224f812bb7c4b2ce8d880f15e44da9742f9657c271f0011`

B 科学 SHA256：`a53def2a5f8c62c19c249e8e4859e817f32e17884831ef9fd7e4c9a4e56e9bda`

加固后完整四样本与初次观测按条件科学 hash 完全相同。当前交付代码 SHA256：`1918f6462fb00feb2844eeb9946bbf11e17358577cbb270f10a55977b165063e`。完整运行信息和每一阶段完整数值保存在 `results/abba-reviewed.json`；初版证据独立保留。

## A1 对 B1 的结果

19 个阶段、1,946 个 float64 记录值；12 个阶段有差异、263 个阶段条目逐位不同，22 个阶段条目超过 8 ULP。它们不是 263 个独立样本，也不能把在多个下游阶段重复传播的差异当作独立证据。

四个已记录前向阶段的 561 个数值全部逐位相同，scalar CE = 1.055606681687835。第一处**已记录**差异为 dL/dlogits，5/48 个数各差 1 ULP。第一权重 momentum buffer 的一个条目达到 640 ULP：

- A：4.3793394966989825e-06
- B：4.3793394966995246e-06
- 绝对差：5.421010862427522e-19

全阶段最大绝对差为 2.7755575615628914e-17。275 个更新后参数中 17 个逐位不同，最大 6 ULP，均在预算内。但整体阶段诊断仍因梯度/buffer 超限失败；严格 hash 也失败。没有放宽预算让它通过。

| Boundary | Values | Changed | Max ULP | >8 ULP | Max absolute difference |
|---|---:|---:|---:|---:|---:|
| hidden_pre | 256 | 0 | 0 | 0 | 0 |
| hidden_relu | 256 | 0 | 0 | 0 | 0 |
| logits | 48 | 0 | 0 | 0 | 0 |
| loss | 1 | 0 | 0 | 0 | 0 |
| logits_gradient | 48 | 5 | 1 | 0 | 3.4694469519536142e-18 |
| hidden_relu_gradient | 256 | 44 | 128 | 7 | 1.7347234759768071e-18 |
| hidden_pre_gradient | 256 | 21 | 128 | 3 | 1.7347234759768071e-18 |
| weight1_gradient | 208 | 72 | 160 | 6 | 6.9388939039072284e-18 |
| bias1_gradient | 16 | 9 | 8 | 0 | 2.7755575615628914e-17 |
| weight2_gradient | 48 | 7 | 8 | 0 | 6.9388939039072284e-18 |
| bias2_gradient | 3 | 0 | 0 | 0 | 0 |
| weight1_momentum | 208 | 72 | 640 | 6 | 6.9388939039072284e-18 |
| bias1_momentum | 16 | 9 | 8 | 0 | 2.7755575615628914e-17 |
| weight2_momentum | 48 | 7 | 8 | 0 | 6.9388939039072284e-18 |
| bias2_momentum | 3 | 0 | 0 | 0 | 0 |
| weight1_updated | 208 | 4 | 1 | 0 | 3.4694469519536142e-18 |
| bias1_updated | 16 | 9 | 6 | 0 | 8.6736173798840355e-19 |
| weight2_updated | 48 | 4 | 1 | 0 | 2.7755575615628914e-17 |
| bias2_updated | 3 | 0 | 0 | 0 | 0 |

## 解释边界

本 fixture 在这台主机/构建下支持“ATen capability 干预会改变至少一个记录边界”的假设。它提供了比完整训练输出更窄的观察范围，但还没有确定差异起源。

即使 logits 和最终 scalar loss 逐位相同，cross_entropy 内未记录的 log-softmax 输出等被保存的前向中间值仍可能不同。dL/dlogits 是第一个被记录到不一致的边界，不能直接等同于差异始于 backward kernel。后续矩阵乘法、归约和正则项可能传播/缩放舍入差异；这里没有做基于同一上游梯度的额外对照来隔离它们。

ATen capability 不完全控制 BLAS/MKL 的内部 dispatch，本次没有追踪实际微内核。640 ULP 不意味着 640 倍模型误差或质量退化；接近零时 representable spacing 很小。小绝对差也不抵消预声明的 8-ULP 失败。

前一手写 16×3 logits fixture 在 AMD 上的完整负结果保留；本次不替代它。历史完整 Wine 干预在另一 Intel Xeon Platinum 8573C 主机，旧 CI 缺少相关硬件记录，本次不能归因它们。原计划“on this host”错误的既有更正保存在 `context/fixed-input-plan-erratum.json`。

## 验证与状态

主实现 59 个测试通过；独立审查另运行 13 个测试（合计 72 个），覆盖原流程/数据与 SGD 首步等价、错误留存和完整性。Ruff/格式主实现与测试、compileall、既有环境 pip check 通过。独立验证与具体范围见顶层 `VALIDATION.md`。

未做新 wheel 包集成、全项目聚合套件、Docker、远端 commit/CI；没有把局部检查冒充这些验证。实现和测试的 Ruff/格式检查独立通过；独立验证也通过。
