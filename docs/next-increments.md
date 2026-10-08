# 已完成增量与下一步验收门槛

## 2026-10-08：模块化小批次训练已实现并在 CPU 运行

原创 `nn.Module`、Dataset/DataLoader、torch SGD 和小批次训练已加入，原手写全批次基线完整保留。参数注册、train/eval/no_grad、尾批损失加权、局部 seed/打乱、输入错误与无泄漏拟合由自动化测试覆盖。三种子原始记录及同配置全批次交叉验证见 [模块化训练说明](minibatch-training.md)。工程运行成功不等于用户已经独立掌握。

## 2026-10-08 第三增量：epoch 边界恢复

[Checkpoint 协议](checkpoint-recovery.md)已实现并完成本地验证：种子42/7/123新进程实验各15项断言通过、连续/恢复参数差0；319项测试、静态检查与三条CLI通过。最终文档/清单验收与发布已完成，精确提交远程CI已通过。范围为固定CPU float64仿射实验、torch SGD momentum、`num_workers=0`和完整epoch边界；保存模型、完整优化器、配置、数据指纹、历史与两个局部 Generator。验收包括新进程中的连续/恢复一致性、遗漏 momentum/重置 shuffle 负对照、严格加载与无覆盖保存。原两阶段源码和证据继续保留。

本地检查和实际实验已经完成，详见[验证记录](../results/2026-10-08-checkpoint-verification.md)；工程增量 [9a5707696c064523ee95bcaddb49ab1f07dc4467](https://github.com/zjDing1024/pytorch-from-zero/commit/9a5707696c064523ee95bcaddb49ab1f07dc4467) 已发布；精确提交对应的 [CPU checks #37751219715](https://github.com/zjDing1024/pytorch-from-zero/actions/runs/37751219715) 已成功完成，319项测试与三条CLI通过。`weights_only=True`、checksum 和大小上限不会使未知来源文件自动可信。此阶段不实现 mid-batch、GPU、AMP、DDP 或 scheduler 恢复。

## 2026-10-08 第四增量：手写 Momentum / StepLR

原创函数式momentum更新与逐参数/缓冲torch对照、StepLR调用顺序、等更新/样本预算对照、完整scheduler状态恢复已实现。三种子40轮/7轮分段新进程实验各24项断言通过、恢复参数差0。固定LR与调度LR在三种子留出测试上没有一致胜者。协议与边界见[scheduler文档](scheduler-recovery.md)；聚合检查及发布阶段见[本次验证记录](../results/2026-10-08-scheduler-verification.md)。本地与全新环境独立审查各679项测试通过；Ruff/39文件格式、pip check、compileall及四条CLI通过。最终清单核对及公开发布已完成；[工程提交2ac5b1b](https://github.com/zjDing1024/pytorch-from-zero/commit/2ac5b1b7f4aa1fce36557b09ec85ce8e837cbdd8)的[CPU checks #37785797239](https://github.com/zjDing1024/pytorch-from-zero/actions/runs/37785797239)成功，远程679项测试与四条CLI通过。

## 下一阶段验收门槛

第四增量已于2026-10-08提前完成，2026-10-09不再重复实现momentum/StepLR。后续先检查学习者独立自测，再推进公开授权真实数据、固定划分与误差分析；不把计划当作已经执行的成果。

1. **独立自测**：不看实现重写 Dataset、单轮训练和最小 epoch 恢复，解释尾批加权、模式、momentum 与 RNG 状态。保存独立实现与失败修复证据。
2. **更新与调度独立解释**：工程对照已完成，学习者仍需不看答案重写momentum、预测每轮LR、构造错序/丢状态反例并解释恢复切点。手写更新模块与torch优化器恢复实验的边界要分清。
3. **真实数据与质量**：选公开授权的小型数据集，建立固定划分、泄漏检查、误差分析与多种子报告；与简单基线公平比较。
4. **运行时工程**：Docker、配置管理、日志、资源计量与测试矩阵。只有实际跑过才标已验证。
5. **GPU 能力**：取得可用 GPU 后测试 AMP、数值稳定、正确计时与显存；CPU 环境先学习原理，不虚构结果。
6. **分布式**：待单机训练与恢复稳定后再研究 DDP、采样器、梯度同步和故障恢复。

达到独立训练、恢复、测试和可解释实验的门槛后，再把这个项目作为完整训练工程求职案例。当前增量可以展示材料质量和验证方法，不能单独代表岗位胜任力。
