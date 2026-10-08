# Momentum、StepLR 与可恢复的学习率轨迹

本增量由 AI 助理原创实现并执行，学习者的独立理解/重写仍待自测。它只扩展固定 CPU float64、3→1 仿射模型、单参数组 SGD momentum、num_workers=0、完整 epoch 边界实验。旧 checkpoint v1 及旧结果不修改、不自动迁移；新格式使用独立入口。

## 1. 要验证的问题

1. [手写 momentum SGD](momentum-sgd.md) 是否在每一步、每个参数、每个 buffer 上与实际 torch SGD 一致？
2. optimizer 与 scheduler 谁先执行？记录的 LR 是本轮使用值，还是下一轮值？
3. 固定 LR 与 StepLR 在同样更新/样本预算下分别得到什么结果？不预设 scheduler 更优。
4. 新进程恢复后，下一次更新能否用对 LR，后续模型、momentum、scheduler、shuffle 与历史能否完全重放？

## 2. 固定实验协议与调用顺序

默认 seed42，训练/验证/测试分别96/32/32，batch20（尾批16），初始LR0.05，momentum0.8，40轮。StepLR(step_size=5, gamma=0.5)；固定LR对照除gamma=1外完全同配置、初始化、数据、打乱序列和训练预算。双方都是200次 optimizer 更新、3,840次训练样本访问。验证/测试样本不参与更新；没有用留出结果选择gamma、step_size或报告种子。无吞吐/统计显著性声明。

每轮先保存当前 optimizer LR，完成所有小批次更新，再评估本轮最终模型，最后调用一次 scheduler.step。记录 lr_used（刚完成一轮用的LR）及 next_lr（下一轮用的LR）。batch内不调度。

数学索引：第 e 轮（从1开始）的 LR 是 initial_lr × gamma^floor((e−1)/step_size)；完成 e 轮后的 next_lr 是 initial_lr × gamma^floor(e/step_size)。例如第1–5轮用0.05，第6–10轮用0.025。实现验证用重复乘gamma，与torch运行顺序相同，避免幂函数的末位舍入差异。

安装版本2.14.1+cpu的StepLR构造会完成一次初始状态设置：last_epoch=0、_step_count=1，但仍保留初始LR。每轮结束后它们分别是e及e+1，不能把这个构造动作误算成训练了一轮。

scheduler_order_demo 用真正的 torch SGD + StepLR(step_size=1) 做常梯度三步对照：正确顺序用0.1/0.05/0.025；提前scheduler.step用0.05/0.025/0.0125，跳过初始值并产生torch警告。这个小例子专门隔离顺序效应；训练实验用step_size=5。

## 3. checkpoint 格式与严格验证

新格式 pytorch-lab-step-lr-checkpoint，schema_version=1；与旧 pytorch-lab-epoch-checkpoint 互相拒绝。保存旧协议全部状态，加完整StepLR state_dict、step_size/gamma、LR历史，以及optimizer里的initial_lr/当前lr。完整epoch边界位于scheduler.step之后，epoch0也支持。

加载顺序：

1. 用weights_only=True、CPU映射和8MiB文件上限读取可信文件；不回退到普通pickle加载。
2. 验证完整键集合、版本、配置与可选expected_config、epoch0–10000、数据指纹、模型张量、momentum缓冲、局部Generator、历史和内容checksum。
3. 独立计算每个边界的期望LR。严格核对scheduler所有字段、last_epoch/_step_count/base_lrs/_last_lr，optimizer参数顺序、initial_lr/当前lr，以及每行lr_used/next_lr。类型也必须匹配，bool不能伪装计数。
4. 构造新的模型、optimizer与scheduler。先加载scheduler状态，再加载optimizer状态（含已经衰减的LR），再恢复其余状态。第一次后续更新前不调用scheduler.step。

公共config（旧RecoveryConfig）及schedule_config（完整ScheduleConfig）都是只读属性。旧模型/数据/RNG/momentum验证通过适配后的私有候选复用，旧API没有放宽。外部不能通过恢复失败获得半加载的在用trainer；候选构造和验证均不推进全局RNG。私有模型/优化器/loader/scheduler不支持调用者修改。

调度配置仅支持整数正step_size和0<gamma≤1；gamma=1明确代表固定LR对照。初始LR正数，极端衰减可能在float64 Python数值表示中下溢到0，届时遵循实际StepLR数值行为，而不是承诺所有配置有学习进展。训练失败后对象不可继续或保存，应从上一个完整边界恢复。

## 4. 新进程与遗漏状态反例

verify 在四个独立Python解释器内运行：连续StepLR、分段StepLR、恢复StepLR、固定LR。默认分别40轮、7轮、从7恢复到累计40轮。恢复比较不是只看最终损失：模型、完整优化器、scheduler、两个Generator、全历史、报告与内容checksum都要相同。

负对照使用单独的早期诊断切点 min(请求split_epoch,7)，分别重置scheduler、清空momentum或重置shuffle；比较到诊断切点+step_size+1，默认第13轮。请求较晚恢复时另建同配置早期状态，真正的连续/恢复对照仍使用请求的split_epoch。这个独立范围跨越一次衰减，避免后期LR接近0导致参数差不可观测却被误判为失败；也不会越过10000轮上限。不把诊断轮数计入公平比较预算。

重置scheduler但保留optimizer当前LR时，有一个重要例外：如果cut恰好是step_size的倍数，重置后的衰减相位可能仍对齐，LR/参数轨迹可以相同，但scheduler绝对计数不同。实验按是否相位错位检查预期，测试覆盖第5轮这种切点；不能声称“任何StepLR状态遗漏都立即改变参数”。已知epoch/config可以按专门协议重建某些调度器；这里选择显式保存并严格验证，而不把该结论推广到任意scheduler。

## 5. 运行

```bash
python -m pytorch_lab.scheduler_cli verify --output results/local-scheduler.json
python -m pytorch_lab.scheduler_cli train --epochs 7 --checkpoint /tmp/scheduled-cut.pt
python -m pytorch_lab.scheduler_cli resume --checkpoint /tmp/scheduled-cut.pt --epochs 40 --save-checkpoint /tmp/scheduled-final.pt
python -m pytorch_lab.scheduler_cli train --gamma 1 --epochs 40 --checkpoint /tmp/fixed-final.pt
pytest -q
ruff check .
ruff format --check .
```

--epochs是累计目标，不是新增轮数。train可调整seed/batch-size/learning-rate/momentum/noise-std/step-size/gamma；resume采用文件配置。verify固定比较协议，仅支持seed/epochs/split-epoch。输出checkpoint和可选JSON分别原子无覆盖保存，不是两文件事务：后一个输出失败可能留下前一个已完成文件。需重试时用新路径。

原始三种子结果：[42](../results/2026-10-08-scheduler-cpu.json)、[7](../results/2026-10-08-scheduler-seed7.json)、[123](../results/2026-10-08-scheduler-seed123.json)；[聚合验证记录](../results/2026-10-08-scheduler-verification.md)。恢复逐位一致只限本次软件/CPU环境，不保证跨PyTorch版本、平台或硬件。三种子小型合成线性任务不支持普遍优化结论。

## 6. 安全与未支持范围

只加载自己或可信来源文件。weights-only、checksum、类型/大小限制是纵深防御，不是敌对文件安全沙箱；checksum不认证来源，也不证明历史值必然由真实训练产生。Linux同目录硬链接保证发布时完整可见、不覆盖，不能保证断电后的目录持久性。

未支持mid-batch、多个参数组、任意模型/数据、任意scheduler、手写优化器checkpoint、GPU、AMP、DDP、外部LR修改、调用者并发修改训练对象。手写momentum模块独立核验更新方程；恢复实验仍使用torch SGD，不能混淆这两个验证边界。

## 官方依据

- [SGD算法与momentum初始缓冲语义](https://docs.pytorch.org/docs/2.14/generated/torch.optim.SGD.html)
- [StepLR规则与状态API](https://docs.pytorch.org/docs/2.14/generated/torch.optim.lr_scheduler.StepLR.html)
- [学习率调度调用顺序](https://docs.pytorch.org/docs/2.14/optim.html#how-to-adjust-learning-rate)
- [Optimizer加载顺序说明](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.load_state_dict.html)

完整方程、安装包阅读定位与原创边界见[手写momentum说明](momentum-sgd.md)和[源码阅读补记](source-reading.md)。
