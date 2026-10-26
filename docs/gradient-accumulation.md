# 梯度累积：按真实有效批次的元素数归一化

日期：2026-10-27（Asia/Kuala_Lumpur）。本次原创实现仍处于 PyTorch 训练循环阶段，不是引入大型 Agent 框架。工程运行证据不等于学习者已独立掌握。

## 问题与明确边界

旧 `train_one_epoch` 每个 DataLoader batch 更新一次，原接口保持不变。本增量提供 `train_accumulated_epoch(model, batches, optimizer, accumulation_steps)`：逐 microbatch 反向传播，只在一个累积窗口结束时更新一次，包括 epoch 最后不足一个窗口的情况。

支持 CPU float64 `AffineRegressor`、固定输出宽度、`torch.optim.SGD`（实验使用 momentum 和 weight decay）。不声称支持 BatchNorm、Dropout、AMP、DDP、GPU、任意 batch-dependent 模型，也没有测量显存或吞吐量。流式训练函数不会把窗口的输入或计算图全部存起来；对照实验为了构造参考批次，才会拼接 fixture。

## 为什么不能平均各 microbatch 的均值

对窗口内第 j 个 microbatch，其样本数为 n_j，输出宽度为 K。目标是整个有效批次的元素均方误差：

`L = sum_j sum((prediction_j - target_j)^2) / (sum_j n_j * K)`。

本实现先对每个 microbatch 的平方误差总和执行 backward，窗口结束时将每个累积梯度除以实际 `sum(target.numel())`，然后执行一次 optimizer.step。分母在完整窗口、短 microbatch 和不完整末窗口都采用实际值，不使用配置容量。

若每个 microbatch 先取 mean 再除以 microbatch 数，则小 batch 会获得与大 batch 相同的总权重。若无条件除以配置的 accumulation_steps，末窗口不足该数量时还会被额外缩小。这是两个不同问题，实验分别构造了反例。

梯度归一化必须在 optimizer.step 之前。SGD momentum、weight decay 及任何外部按更新计数的 scheduler 都涉及更新频率；每个 microbatch 都执行 optimizer.step 后再平均参数不是本算法。本接口不调用 scheduler、不做 gradient clipping。多输出 K>1 不能遗漏分母中的 K。

## 固定实验协议（运行前声明）

- 三个局部生成器种子：42、7、123；三个 fixture：N=23/micro=5/accumulation=3，N=11/micro=4/accumulation=2，N=10/micro=4/accumulation=3。
- 输入 3 维、输出 2 维；局部生成的有界尺度仿射数据；零初始化；各跑 3 个 epoch；相同样本顺序，不引入 shuffle 干预。
- SGD：lr=0.03，momentum=0.8，weight_decay=0.01。参考方法在相同窗口拼接输入，每窗口一次正常 MSE 更新。
- N=23 的窗口大小为 15 和 8，第二个窗口含 5+3 两个 microbatch。N=11 的最后窗口只有一个 microbatch，隔离固定除数问题。
- 比较**每次更新后的全部参数及 momentum buffer**，不是只比较最终 loss。
- 预声明 `atol=1e-12, rtol=1e-12`：用于这个小尺度 CPU float64 仿射实验的工程近似一致性标准，给有限次归约顺序差异留出余量；不是数学误差上界、通用数值保证，也不替换历史 strict / 8-ULP 标准。
- 两个故意错误的控制只出现在实验模块：窗口内 mean-of-means；固定 accumulation_steps 除数。它们不是可选训练策略。

## 实际运行与证据

[原始结构化结果](../results/2026-10-27-gradient-accumulation.json)含 9 个 case、逐 epoch 工作计数、每次更新的参数/momentum 最大差距及正负控制结果。该次 weighted/reference 的最大绝对差为 `2.220446049250313e-16`，所有逐更新比较满足预声明标准；不等长 microbatch 的 mean-of-means 错误和不完整末窗口的固定除数错误均被检测到。

另有[独立解析梯度与手写 SGD oracle](../scripts/verify_accumulation_analytic.py)及[结构化结果](../results/2026-10-27-accumulation-analytic.json)：96 个配置各运行 4 个 epoch，包含 K=1/3、普通 momentum、dampening、Nesterov、maximize 和 weight decay；最大参数/momentum 绝对差 `4.440892098500626e-16`。这些覆盖与单元测试有重叠，单独报告，不能叠加为测试总数。

这是算法/边界验证，不是模型性能优越性结果。没有验证集调参、没有查看或改变旧 Wine 测试集，没有扩展到未运行的设备。测试与发布状态见[本次验证说明](../results/2026-10-27-gradient-accumulation-verification.md)。

## 失败语义

- accumulation_steps 必须是正整数，布尔值不视为整数配置。
- 空输入、空 batch、形状不匹配、错误 dtype、NaN/Inf、loss/gradient 溢出均被拒绝。
- optimizer 必须恰好持有本模型的参数，防止在错误模型上更新或遗漏梯度。
- 开始时清理旧梯度，正常结束及迭代/验证/backward 异常时清理未完成窗口的梯度。
- 已经完成的 optimizer 更新**不回滚**；optimizer 内部失败也不保证参数事务性。失败 epoch 应重新加载上一个可信 checkpoint，而不是继续使用半完成状态。
- 返回的 online_mse 混合了每个窗口更新前的模型状态，不是 epoch 结束后固定模型的评估指标。
- CLI 使用 exclusive-create，不覆盖已有结果文件；退出码 0 仅表示结果中的已定义检查通过。

## 复现

```bash
python -m pytest -q tests/test_accumulation.py
python -m pytorch_lab.accumulation_cli --output results/local-accumulation.json
```

输出同时写到文件和 stdout；同名输出存在时选择新文件名。源码 CI 增加第九个 CLI；安装 wheel 的检查脚本也增加该入口。真正的容器执行结果只能以对应提交的 CI 为准。

## 官方资料、研究项目与许可

本次筛选 Hugging Face Accelerate 的梯度累积实现作为研究对象，但没有把它加入依赖或复制实现。其工程价值在于显式管理同步和 optimizer 更新边界；本项目先验证单机数学契约，再讨论分布式拓展。

- [Accelerate 官方 variable-size accumulation 指南](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation)：其可变 token 示例强调总损失与有效元素总数，不能将各 batch 均值简单平均。本项目将该原则独立应用到固定输出宽度 MSE，不照搬 token/多设备公式。
- [固定版本 accelerator.py](https://github.com/huggingface/accelerate/blob/8c83b448d19f6356d4facd65c1d4763bf9fa51b2/src/accelerate/accelerator.py)：阅读 `_do_sync`、`accumulate`、`backward` 的职责；未复制代码。[同版本 Apache-2.0 LICENSE](https://github.com/huggingface/accelerate/blob/8c83b448d19f6356d4facd65c1d4763bf9fa51b2/LICENSE)。
- [PyTorch MSELoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MSELoss.html)：mean reduction 的分母是目标元素数量，不仅是样本数。
- [Optimizer.zero_grad](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.zero_grad.html)：本实现显式 set_to_none=True；不能把 None 梯度与零梯度在优化器中的语义混为一谈。
- PyTorch 的既有依赖许可与第三方声明见 [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md)。未改变本仓库许可，也没有再分发 Accelerate 源码。

## 待学习者自测

不看实现写出 N=23、micro=5、accumulation=3 的两个窗口分母和 optimizer.step 次数；解释 K=3 为什么需要除以样本数的三倍；独立构造 mean-of-means 失败案例；解释 momentum 更新频率以及失败 epoch 的恢复边界。没有这些独立证据前，不把本次助手完成的产物记作已掌握能力。
