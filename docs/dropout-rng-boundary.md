# CPU Dropout：随机前向的状态边界

## 问题与非目标

昨天的联合训练器只有确定性仿射前向。本项先隔离一个新问题：相同权重、输入与 seed，为什么恢复后的 Dropout 输出仍可能不同？这是可复用的**随机前向／反向重放探针**，不是新训练器。没有优化器更新，没有扩展 `JointTrainer` 的快照协议。

`DropoutReplay` 使用固定 CPU float64 的 Linear(4,8) → Dropout → Linear(8,2)。第 t 步输入是 5×4 的 arange / 8 − 1 + t / 32，目标是 5×2 的 arange / 16 − 1/4，损失为残差平方之和。每步记录输出、损失、输入梯度、四组参数梯度和前后 RNG 摘要；权重不变，每次反向前清空参数梯度。

## RNG 所有权与恢复顺序

1. 两个 CPU `Generator` 的初始状态分别控制模型初始化与随机前向；这里只使用它们生成默认 CPU RNG 的起始状态。
2. 普通 `nn.Dropout` 没有接收这个局部 Generator。执行时进入 `fork_rng(devices=[])`，临时设置进程默认 CPU RNG 为探针拥有的状态，完成前向、反向与结果提取。
3. 离开上下文后调用者默认 CPU RNG 恢复。只有完整成功，才提交下一私有 RNG 状态和步骤数。
4. 恢复时先构造模型，再加载权重和捕获的 RNG 状态。把随机状态设好后又运行随机初始化，会消耗本应给下一次 Dropout 的随机序列。
5. 快照包含版本、fixture 身份、`p`、训练模式、初始化／前向 seeds、权重、步骤数和完整 CPU RNG 字节状态。seed 只能定义起点，不能替代已经前进的状态。`state_dict()` 只覆盖模型数值，不能替代训练模式或 Dropout 概率约定。

探针限制最多 1,000,000 步；达到上限的快照仍可加载、检查，但不能继续执行。直接改动模块 mode 或 Dropout p 会使 step/snapshot 拒绝，避免配置与实际行为分离。模型结构、参数与 hook 不能在探针运行中由外部改写。

## 异常与文件边界

本模型无可变 buffer、无更新步骤。在被测前向、反向与结果提取阶段的受控异常后保留私有 RNG 与步骤数，默认 CPU RNG 仍由上下文还原。失败的反向可能留下部分 `.grad`，下一次尝试先全部清空；快照不包含这些临时梯度。这是固定无副作用模型的重试设计，**不是任意 Module 的事务回滚**。独立测试覆盖真正消耗随机数或写入梯度之后的异常。明确注入的 KeyboardInterrupt 不等于任意时刻杀进程；最终私有状态提交与返回之间的任意异步中断及调用者确认协议不在本项保证内。

`fork_rng` 临时修改的是进程全局状态，因此仅支持单线程、没有并发随机消费者的调用环境。它不是线程隔离、互斥锁或 GPU RNG 保存器。不保存 Python、NumPy、CUDA RNG，也不支持 worker、随机数据变换、AMP、DDP 或跨版本／跨平台快照交换。

只加载本项目自己产生且可信的 JSON；格式／形状／有限性检查与 SHA-256 用于检测误配和损坏，不是签名、历史认证或恶意文件沙箱。文件采用不覆盖写入；写入失败可能留下无效文件，不声称原子保存或断电持久性。

## 三种对照

- 真实新进程：每个 seed 用一个基线进程保存第 1、3、5 步快照，另一个恢复进程分别继续到第 8 步。三 seed 共 6 个 worker、9 个恢复案例、45 个剩余步骤。比较完整记录与最终快照，而非只比较最终 loss。
- 错误恢复：保留权重与步数却只重设 seed；或先恢复 RNG 再运行随机初始化。对照必须确实改变输出，不能仅因 RNG 摘要不同就宣布捕获了数值问题。
- 模式与干扰：默认 p=0.35，在相同初始随机状态下，train 与 train + no_grad 的输出相同、都消耗 RNG；no_grad 仅关闭梯度记录。eval 不进行 Dropout 随机采样，但可以记录梯度。p=0 与 p=1 是另测边界，不要求它们消耗随机数。外部默认 CPU 随机采样不应改变探针所拥有的随机序列。

```bash
python -m pytorch_lab.dropout_rng_cli --output results/local-dropout-rng.json
```

输出路径必须不存在。CLI 退出码非零表示实验检查失败。实际结果见[原始紧凑 JSON](../results/2026-10-31-dropout-rng.json)和[验收记录](../results/2026-10-31-dropout-rng-verification.md)。独立手写链式法则审查使用相同上游 Dropout kernel 生成 mask，独立的是前向／反向方程，不是新的 RNG 算法。

## 官方项目研究与原创性

研究对象一：PyTorch，维护者 PyTorch 社区，解决张量运算、自动微分与模块执行。固定 v2.14.1 commit [5c4886908584029761b579af026dcfb627c84070](https://github.com/pytorch/pytorch/tree/5c4886908584029761b579af026dcfb627c84070)。[torch/random.py](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/random.py)是默认 RNG 与上下文的 Python 入口，[dropout.py](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/nn/modules/dropout.py)组织模块概率与训练模式，实际运算委派底层 kernel。根 [LICENSE](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/LICENSE)为 BSD-3-Clause-style 条件，依赖另有许可。官方2.14 [Dropout](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout.html)、[no_grad](https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html) 与 [random](https://docs.pytorch.org/docs/2.14/random.html)分别说明模式、梯度上下文与 CPU RNG。实际运行 torch 2.14.1+cpu，源码阅读固定到对应 v2.14.1 tag。

研究对象二：Hugging Face Accelerate，组织模型、优化器、设备与训练状态的工程框架。固定 [checkpointing.py](https://github.com/huggingface/accelerate/blob/90b233c86202a5ded4bf66407e75222c5fb007fc/src/accelerate/checkpointing.py)观察组件恢复与随机状态恢复的先后；[LICENSE](https://github.com/huggingface/accelerate/blob/90b233c86202a5ded4bf66407e75222c5fb007fc/LICENSE) 为 Apache-2.0。其名为 torch_manual_seed 的载荷实际保存 RNG 字节状态，而不是单个 seed；该版本某些 RNG 恢复异常会记录日志而不向外抛出，因此本项目额外以实际下一步重放验证成功，不能仅凭加载函数返回判断。该观察不构成对上游框架所有版本的评价。

适配阶段 1 的价值在于状态归属、初始化顺序、异常边界与可证伪比较。本项目代码、fixture、协议与测试均原创；没有复制上游代码、注释或片段，没有新增 Accelerate 依赖。

## 求职与个人自测

作品可以展示对可复现性和随机前向的故障定位，不能据此宣称训练性能提高或生产容灾能力。本人还需独立解释 mode 与 no_grad、重写最小恢复反例、预测初始化多一次导致的后果，并给出自己的运行和修复记录。助理运行通过不等于本人掌握。

下一步先设计随机前向加入联合训练安全点的协议，特别是失败时模型 RNG、待用梯度、索引进度的共同归属；本项尚未实现它。
