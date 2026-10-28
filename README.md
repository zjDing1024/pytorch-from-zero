# pytorch-from-zero

## 2026-10-29：受控打乱索引的恢复合同

新增原创 CPU `ShuffledBatchStream`：局部 RNG、完整排列、已发出游标和显式 epoch 边界分开保存，
覆盖部分 epoch、尾批/`drop_last`、空轮次及下一轮随机状态。**只恢复 index emission，不恢复训练**；
`next_batch()` 发出即推进，调用方后续失败不会自动回退。无 workers/prefetch/DDP 或任意 DataLoader 声称。

[状态协议与官方源码研究](docs/shuffled-index-recovery.md) ·
[实现](src/pytorch_lab/shuffle_stream.py) · [测试](tests/test_shuffle_stream.py) ·
[紧凑实验报告](results/2026-10-29-shuffle-stream.json) ·
[独立 oracle 审查](results/2026-10-29-shuffle-stream-independent-review.json) ·
[实际验证范围](results/2026-10-29-shuffle-stream-verification.md)

```bash
python -I -m pytorch_lab.shuffle_stream_cli --output results/local-shuffle.json
```

实验按精确索引序列比较，并通过错误游标/seed/RNG 对照证明检查能发现重复、遗漏和下一轮偏移。
本轮完整本地验收为 1,062 + 59 + 45 + 34 = **1,200 项测试**，干净非 editable wheel 的 **11 个 CLI** 全部通过。
旧十个安装 CLI 的顺序保持，追加第十一个；既有 Wine 科学证据与阈值保持不变。本人独立自测仍待完成。


## 2026-10-27：样本加权梯度累积

新增原创 CPU float64 仿射训练循环：逐 microbatch 反向传播，按实际有效批次的目标元素数归一化，只在窗口边界更新 SGD，正确处理尾批和不完整末窗口。每次参数与 momentum 更新对照真实拼接批次，并保留两个错误实现反例。旧训练循环和历史科学证据不变。

- [算法契约、官方来源与边界](docs/gradient-accumulation.md)
- [9 个固定 case 的实际结果](results/2026-10-27-gradient-accumulation.json)
- [本次测试与发布状态](results/2026-10-27-gradient-accumulation-verification.md)

本次不声称 GPU/显存/吞吐量收益，不代表学习者本人已掌握。源码和安装 wheel 的 CI 检查新增第九个 CLI。

## 2026-10-26：四批工程成果发布记录

固定输入诊断、首步定位、saved-forward 探针和交付验证器的发布状态统一以本节为准。下方旧日期中的“未公开”“本地候选”保留为历史事实。

- 工程内容提交：[c57de064e929bd388261e0514dd93015d10d8256](https://github.com/zjDing1024/pytorch-from-zero/commit/c57de064e929bd388261e0514dd93015d10d8256)。
- 该内容提交的远端 CI：[实际运行记录](https://github.com/zjDing1024/pytorch-from-zero/actions/runs/40322000534)。核验结果：成功。source 作业通过 1,018 项测试（880 + 59 + 45 + 34）、Ruff 检查/格式及 8 个 CLI；installed-cpu-container 作业成功完成实际容器中的 8 个 CLI。主套件 880 项耗时 246.79 秒；可选 NumPy 缺失警告未导致失败。容器没有运行完整测试套件。。
- 10 月 22 日本地验收的 1,018 项测试及 source / installed-wheel 各 8 个 CLI 是当日证据，不计作 10 月 26 日重跑。新 CI 的 source 与 container 范围分别说明，不把容器 CLI 检查写成容器完整测试套件。

本次补齐已有成果的公开交付及精确提交验证。公开副本的隐私处理范围：仅将两份历史测试日志的本地环境路径替换为中性标记，更新脱敏说明与对应 SHA256 清单，共 4 个文件；原始证据单独保留，科学数据、源码、输入 ZIP 与数值阈值未变。本次对公开副本重新运行的 34 项交付验证测试和两份输入包静态检查均通过。冻结原件单独保留，不新增科学结论或放宽数值阈值。个人独立实现、讲解和排错能力仍待自测。详情见[发布整合记录](docs/release-integration.md)。

## 2026-10-22：四批工程交付的重新执行验收

固定输入、首步定位、saved-forward 和交付验证器已经合并到本候选，并在新的 CPU 环境重新执行测试。详见[本次原始结果与验证边界](results/2026-10-22-release-verification/README.md)。旧日期段落是历史记录，不代表当前发布状态。

本次保留所有原始科学输入、阈值、失败和源代码快照，没有新增 demo 或改写科学结论。本地执行证据不代替远端精确 commit 的 CI；发布结果须以实际提交和对应 Actions 为准。

## 2026-10-18 历史本地整合状态与导航

本地恢复候选已补齐四批工作的入口，仍未作为新提交公开。此段记录当时整合状态；最新公开状态以顶部 10 月 26 日记录为准。已核验的公开工程基线仍是 [aa56d5e](https://github.com/zjDing1024/pytorch-from-zero/commit/aa56d5e3ae5ffcee428d1f4c996a04497df7bb7e)。

- [整合范围、验证分层与发布阻断](docs/release-integration.md)
- [固定输入逐阶段诊断](docs/fixed-input-diagnostics.md)
- [首个训练更新定位：原始交付](experiments/wine-first-step-2026-10-13/README.md)
- [保存前向的交叉熵反向探针：原始交付](experiments/wine-saved-forward-2026-10-14/README.md)
- [交付验证工具：原始交付](tools/wine-delivery/README.md)

2026-10-17恢复验收的1,018项测试和source/wheel各8条主项目CLI属于当天证据。今天没有重跑整套训练/数值实验，没有更新阈值或覆盖历史失败。新增的是路线整合和独立只读发布检查；这不代表新的精确commit CI已经通过。

**A CPU-first lab connecting PyTorch behavior, mathematical derivations, and executable tests.**

当前内容：Tensor 布局、Autograd、三路梯度校验、独立留出集回归、nn.Module、Dataset/DataLoader 和小批次 SGD。2026-10-08 同日第三增量加入 epoch 边界 checkpoint 与 momentum SGD 恢复，第四增量加入手写 momentum 与 StepLR 等预算/恢复验证，最终状态见下文。2026-10-09新增公开授权UCI Wine真实数据分类、固定划分、训练集归一化、验证选择与误差分析。代码原创编写，官方源码只用于阅读与核对；随包数据有单独署名和许可。

> 证据边界：本增量由 AI 助理实现并在 CPU 环境验证。它提供可复查的工程材料，用户的独立理解与实现能力仍需 [自测](docs/learner-self-check.md)。当前项目尚不足以证明生产级训练、GPU 优化或分布式能力。

## 2026-10-10：CPU运行时与资源证据

新增[运行时协议](docs/runtime-reproducibility.md)：目标平台限定的10个依赖wheel哈希锁、新建最小环境的已安装wheel检查、六条CLI、原创的新进程资源探针和超时/损坏产物负对照。固定Wine协议每次6次拟合，共5,040更新/77,040样本访问；source与wheel各3次实测的科学输出均与前日结果一致。

运行时间、CPU时间和Linux高水位RSS分开记录，不把RSS当张量内存，也不因环境不同的数值差异宣称性能提升。[修复后工程提交e162eaf](https://github.com/zjDing1024/pytorch-from-zero/commit/e162eafd654e44bfea3473c391491fa071b443d0)的[CPU checks #37959687015](https://github.com/zjDing1024/pytorch-from-zero/actions/runs/37959687015)已成功：source job通过784项测试（243.42秒，1条可选NumPy警告）、Ruff/格式和6条CLI；container job实际构建并以非root、只读文件系统和断网方式完成6条已安装wheel CLI。本机仍无容器引擎，容器执行证据来自上述GitHub CI。跨环境科学hash出现差异，重复性结论限定为各环境内部，详见协议。实际验证/公开状态见[本次验证记录](results/2026-10-10-runtime-verification.md)。个人理解仍待独立自测。

```bash
# 先按运行时文档安装hash-locked wheel；本命令不会覆盖已有结果。
python -I -m pytorch_lab.runtime_probe --output results/local-runtime.json
```

## 背景与解决的问题

入门实验容易只验证“能跑”，却漏掉目标张量广播、梯度缩放、累加与数据泄漏。本项目把行为写成可失败的测试，并同时保留数学解释、数据划分、运行环境和原始 JSON 结果。

首个版本回答：

1. 为什么转置可以共享内存，却不能总用 `view` 展平？
2. 为什么广播偏置的梯度会沿广播维度相加？
3. 为什么不清理 `.grad` 会得到双倍梯度？
4. 手算、Autograd、数值差分是否给出一致结果？
5. 梯度下降是否接近最小二乘解，并在未参与更新的样本上超过基线？

## 技术架构与核心实现

```text
src/pytorch_lab/
  tensors.py      布局、广播、梯度累加实验
  regression.py   严格输入契约、MSE、解析梯度、有限差分、训练循环
  experiments.py  实验组合、判定条件、环境与结果记录
  __main__.py     原始 CLI、JSON 输出、失败退出码、防止覆盖证据
  minibatch.py    Module 参数注册、Dataset/DataLoader、模式切换与小批次 SGD
  minibatch_cli.py 模块化增量独立 CLI；保留原始实验入口
  checkpoint.py   epoch 边界训练状态、严格加载、无覆盖保存
  checkpoint_cli.py train / resume / verify 独立入口
  checkpoint_experiment.py 新进程恢复对照与遗漏状态的负对照
  momentum.py     原创函数式SGD更新及逐参数/缓冲torch参考轨迹
  scheduled_checkpoint.py StepLR边界训练、LR历史及严格状态恢复
  scheduler_cli.py / scheduler_experiment.py 调度顺序、等预算、跨进程对照
  wine.py / wine_cli.py 真实数据分类、无泄漏预处理、验证选择与误差分析
  runtime_probe.py Linux新进程资源测量、科学输出/包文件hash与重复性检查
  numerical_probe.py 完整科学JSON、CPU/dispatch证据与受控新进程捕获
  scientific_compare.py 独立的严格hash、精确字段与8步binary64诊断
  fixed_input_probe.py 冻结logits、显式算子边界和独立捕获/数值诊断
  data/           固定Wine数据、来源/许可、完整划分行ID
tests/            数学正确性、坏输入、重复运行、CLI 集成测试
examples/         独立运行入口
docs/             推导、源码阅读、质量边界、自测与扩展计划
results/          已执行的原始结果和检查日志
```

`affine_mse` 使用 `X[N,D] @ W[D,K] + b[K]`，损失对 `N*K` 个元素取均值。手写解析梯度、Autograd 与中心差分相互核对；多输出测试专门检查分母遗漏 `K` 的错误。训练循环直接更新叶子参数，并在每步清理梯度。输入拒绝空维度、形状不符、不同 dtype、整数和非有限数值。

## 环境配置

已验证：Linux x86_64；本地Python3.12.14/glibc2.41、source CI的Python3.12.15/glibc2.39、容器Python3.12.14/glibc2.36，均使用PyTorch2.14.1+cpu。全量测试使用pytest9.1.1、ruff0.16.10，第七增量的容器路径运行已安装wheel的7条CLI，最新精确提交证据见下文。项目声明Python>=3.11，其余Python/操作系统组合尚未验证。无需数据下载、GPU、付费 API 或账户密钥。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .
```

安装来源：[PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)。CPU 包固定为实验实际运行版本；未宣称该版本是所有平台的最佳选择。顶层依赖已固定，尚无完整跨平台传递依赖锁文件。全新隔离环境已按以上步骤安装并通过检查。由于本实验没有 NumPy 依赖，纯净环境可能显示 PyTorch 的可选 NumPy 初始化警告；本项目不调用 NumPy 互操作功能。

## 运行方式

```bash
python -m pytorch_lab --output results/local-run.json
python examples/run_lab.py --seed 7
python -m pytorch_lab.minibatch_cli --output results/local-minibatch.json
python -m pytorch_lab.wine_cli --output results/local-wine.json
pytest -q
ruff check .
ruff format --check .
```

同一输出路径不能重复写入，请使用新文件名。命令失败会返回非零退出码。`--steps` 和 `--learning-rate` 允许探索失败的学习率/步数，不保证任意参数都能收敛。默认参数固定后才评估测试集，不用测试集选择参数。

## 首个增量实验结果

2026-10-08 UTC 实测结果见 [原始 JSON](results/2026-10-08-cpu.json)。单个训练种子 42、float64、200 步、学习率 0.1，数据来自自包含合成过程；96/32/32 个样本分别用于训练/验证/测试。

| 检查 | 实测 |
|---|---:|
| Autograd 与解析梯度最大绝对误差 | 5.55e-17 |
| Autograd 与中心差分最大绝对误差 | 8.99e-10 |
| 初始 → 最终训练 MSE | 7.805889 → 0.003311 |
| 验证集 MSE | 0.002961 |
| 测试集 MSE | 0.004406 |
| 训练集均值预测的测试 MSE | 3.891327 |
| 梯度下降与最小二乘参数最大差距 | 3.56e-14 |
| 单元/集成测试 | 52 项通过 |

这说明实现通过了本实验的数值与行为检查。合成线性任务的成功不能外推到真实复杂任务，单种子结果也没有统计置信度。最小二乘底层库可能造成末位差异，不承诺跨版本、跨硬件逐位一致。

## 新增：Module / Dataset / DataLoader 小批次训练

[实现与比较协议](docs/minibatch-training.md) 保留原始数学基线：注册 weight/bias 参数，以独立 Dataset 保存数据，每批由 torch SGD 更新，训练用 train 模式，评估用 eval + no_grad。默认 batch size 20，尾批16条样本不丢弃，损失按实际样本数加权。训练 loader 与评估 loader 使用独立局部 Generator，不推进全局 RNG。

2026-10-08 CPU 实测：固定40轮、学习率0.05，种子42训练 MSE 7.805889 → 0.003311246，测试 MSE 0.004424504，训练均值基线3.891327。种子7/123同配置也通过5项实验断言；3次 Module 全批次与原手写同设置参数差均为0。

小批次与40轮手写全批次均访问3840个训练样本，但分别更新200次和40次；这里没有吞吐或等更新预算的优越性结论。在线损失和本轮最终模型损失分开记录。原始证据：[42](results/2026-10-08-minibatch-cpu.json)、[7](results/2026-10-08-minibatch-seed7.json)、[123](results/2026-10-08-minibatch-seed123.json)。

## 第三增量：epoch 边界 checkpoint 恢复

状态：**已实现、发布并通过本地及远程验证：319项测试、三种子新进程恢复实验通过。** [恢复协议与边界](docs/checkpoint-recovery.md)固定CPU float64、3→1仿射模型和 `num_workers=0` DataLoader；默认batch20、学习率0.05、momentum0.8。完整epoch边界保存模型、完整优化器、配置、数据指纹、历史与训练/指标Generator状态；新进程加载后继续到累计目标轮数，epoch0也可保存。

[种子42原始结果](results/2026-10-08-checkpoint-cpu.json)：连续40轮与“7轮后保存、新进程恢复到40轮”的模型、优化器、RNG、完整历史和报告相同，参数最大差距0，15项断言通过，测试MSE为0.0043601093。遗漏momentum、重置shuffle后再训练一轮的参数差分别为0.0144504611、0.0036747311。种子[7](results/2026-10-08-checkpoint-seed7.json)/[123](results/2026-10-08-checkpoint-seed123.json)同配置也各通过15项断言、恢复参数差0。完整本地及远程测试各319项通过；精确提交与CI链接见下方。旧无momentum实验的原始结果保持独立。

只加载自己或可信来源的 checkpoint。受限 weights-only 加载、内容 checksum、文件/epoch 上限和严格验证不构成不可信文件安全沙箱。保存拒绝覆盖，Linux 本地文件系统的原子可见性不等于断电持久性。详细命令及未支持场景见[checkpoint 文档](docs/checkpoint-recovery.md)。


## 当前增量：手写 Momentum / StepLR / 调度状态恢复

状态：**本地与全新环境独立审查各679项测试通过，静态检查、四条CLI及三种子实验通过；最终清单验收、公开发布和精确提交远程CI已完成。** [工程提交2ac5b1b](https://github.com/zjDing1024/pytorch-from-zero/commit/2ac5b1b7f4aa1fce36557b09ec85ce8e837cbdd8)对应的[CPU checks #37785797239](https://github.com/zjDing1024/pytorch-from-zero/actions/runs/37785797239)成功，远程679项测试与四条CLI通过。 [手写更新方程](docs/momentum-sgd.md)覆盖None/零梯度、初始buffer、dampening、coupled weight decay、Nesterov、maximize与变化LR；六组配置×六步×两参数逐步对照torch，参数/buffer最大差1.11e-16/2.22e-16，容差1e-12。函数式手写模块不调用torch.optim，恢复实验仍用torch SGD。

[StepLR协议与新CLI](docs/scheduler-recovery.md)固定每轮全部optimizer更新后调用一次scheduler。三种子42/7/123连续40轮与7轮后新进程恢复到40轮的模型、优化器、scheduler、RNG、完整历史及报告相同，参数差均0，各24项实验断言通过。新格式严格校验当前LR、initial_lr、scheduler计数和每轮LR历史；旧checkpoint格式保持不变。

固定LR与StepLR均为200次更新/3840次样本访问、相同数据/初始化/打乱。测试MSE（StepLR / 固定LR）：42为0.004408268 / 0.004360109；7为0.003279841 / 0.003395952；123为0.001914461 / 0.001906621。结果没有一致胜者，不声称调度器普遍更好。原始证据：[42](results/2026-10-08-scheduler-cpu.json)、[7](results/2026-10-08-scheduler-seed7.json)、[123](results/2026-10-08-scheduler-seed123.json)。

重置scheduler、遗漏momentum和重置shuffle的负对照另行记录；StepLR在整周期切点重置可能保留相同LR轨迹，不能把默认第7轮反例推广到所有切点。[完整验证记录](results/2026-10-08-scheduler-verification.md)保存独立审查、冻结清单、真实提交和本增量远程CI证据。

## 性能分析

- 每个训练步矩阵乘法量级为 `O(NDK)`；存储以输入、参数和自动微分中间量为主。当前训练规模很小，不用于吞吐结论。
- 解析梯度与反向传播复用矩阵运算。中心差分对每个参数做两次前向，成本约为 `2(DK+K)` 次前向，仅适合校验。
- 输入的有限数值检查每次扫描张量；它优先服务教学安全。在 GPU 上会引入同步成本，需要重新设计校验边界后再做性能优化。
- 未执行 GPU、混合精度、显存或多机基准；没有速度提升声明。

## 文档、质量与下一步

- [数学、设计与失败模式](docs/design.md)
- [固定版本源码阅读](docs/source-reading.md)
- [Checkpoint 恢复协议](docs/checkpoint-recovery.md)
- [Checkpoint 验证记录](results/2026-10-08-checkpoint-verification.md)
- [学习者自测](docs/learner-self-check.md)
- [后续工程计划](docs/next-increments.md)
- [贡献与证据规范](CONTRIBUTING.md)

`nn.Module`、`Dataset/DataLoader`、小批次SGD及epoch边界checkpoint都已有本地验证。手写momentum SGD等价对照与StepLR顺序/状态实验已完成；2026-10-09继续完成UCI Wine真实数据评估。2026-10-10完成依赖/wheel复现与资源实测；2026-10-11继续完成完整科学输出、CPU证据和固定数值诊断，并通过最新831项source测试及7条容器CLI。下一步继续个人独立自测和预声明的逐阶段差异定位，再设计受控profiling。混合精度和分布式仍属后续计划。

2026-10-08 本地验证：首轮52项测试与独立隔离安装检查通过；新增量合计152项测试、lint、格式和两条CLI通过，详见[本次验证记录](results/2026-10-08-minibatch-verification.md)。远程 CI 的最新状态请查看 [Actions](https://github.com/zjDing1024/pytorch-from-zero/actions)，本地结果不替代远程运行证据。未选择开源许可证，不应将公开可读等同于已授予再分发许可。

第三增量本地验证：319项测试（原152项+checkpoint契约151项+CLI16项）、Ruff检查/29文件格式、pip check、compileall和三条CLI全部通过；最终代码独立复跑319项及种子42精确重放也通过。详见[checkpoint验证记录](results/2026-10-08-checkpoint-verification.md)。最终文档/清单验收和发布已完成。工程增量 [9a5707696c064523ee95bcaddb49ab1f07dc4467](https://github.com/zjDing1024/pytorch-from-zero/commit/9a5707696c064523ee95bcaddb49ab1f07dc4467) 已发布；精确提交对应的 [CPU checks #37751219715](https://github.com/zjDing1024/pytorch-from-zero/actions/runs/37751219715) 已成功完成，319项测试与三条CLI通过。该证据专属于本次恢复增量。


## 2026-10-09：真实数据、验证选择与错误分析

本地与独立隔离环境各709项测试、五条CLI及离线wheel验证通过，最终冻结清单验收无遗留阻塞。[工程提交a274382](https://github.com/zjDing1024/pytorch-from-zero/commit/a27438230cbb2a9b515a6ca355029e5c6889ec26)已公开发布，其精确提交的[CPU checks #37808159705](https://github.com/zjDing1024/pytorch-from-zero/actions/runs/37808159705)已成功，远程709项测试、Ruff/格式和五条CLI全部通过。

[Wine协议与数据卡](docs/wine-evaluation.md)把已有训练机制扩展到公开授权的178条真实化学分析数据；这是评估工程示例，不是困难基准。数据SHA-256和107/36/35固定分层划分随包保存，离线运行；仅训练集拟合归一化，检查行重叠/完全重复特征，验证集选择模型，测试集最后评估。[数据署名与许可](THIRD_PARTY_NOTICES.md)及原始文件均已保留。

120轮、batch16、相同SGD配置和种子42/7/123；线性42参数与小MLP275参数分别每次840更新/12,840次样本访问。验证CE均值选择MLP（0.014833，对照线性0.017577），但最终测试准确率均值为MLP96.19%、线性97.14%，训练类先验/多数类基线40%。按原协议保留MLP选择，没有用测试结果倒改。两个模型都误判行68，MLP种子123另外误判行134；35条测试样本的一条错误对应2.86个百分点，不宣称统计显著性或深度模型优越。

[首次完整原始JSON](results/2026-10-09-wine-cpu.json)包含每个种子的预测/概率、混淆矩阵、macro-F1、错误行ID、训练统计和公平预算；[验证记录](results/2026-10-09-wine-verification.md)区分本地、独立审查与远程状态。三种子只刻画同一划分下初始化/打乱的变化。尚未证明跨地区/年份泛化、部署、GPU或个人独立掌握。

## 2026-10-11：完整科学输出与数值回归诊断

第七增量保留每次Wine完整JSON，新增CPU型号/特征、dispatch与运行库证据，以及原创严格类型/结构比较器。原runtime严格hash门保持不变；另行预声明的8相邻binary64步诊断门不会按实测放宽。历史两输出最大75 ULP，有932项超过该门，虽预测/选择不变仍保留失败。

同主机A-B-B-A仅改变ATEN_CPU_CAPABILITY：AVX512与DEFAULT各自重复一致但彼此不同；不能由此解释历史CI的两个hash。完整计划、复现命令、数值与推断边界见[数值回归协议](docs/numerical-regression.md)及[本次验证](results/2026-10-11-numerical-verification.md)。[修复后工程提交025a1d7](https://github.com/zjDing1024/pytorch-from-zero/commit/025a1d7da1b68ccd1965b9bfb355eba643ee7505)的[CPU checks #38132126861](https://github.com/zjDing1024/pytorch-from-zero/actions/runs/38132126861)已成功：source job通过831项测试（186.64秒、1条可选NumPy警告）、Ruff/格式和7条CLI；container job实际完成Linux amd64镜像构建及非root、只读、断网的7条已安装wheel CLI，两路完整数值artifact均已上传并下载核验。首次source CI的进程观测竞态及测试修复也保留在验证记录中。科学诊断未通过的结果仍然保留，软件CI通过没有改写该结论。助理实现仍不等于个人掌握。


## 2026-10-12：固定输入的数值定位

新增[固定logits协议](docs/fixed-input-diagnostics.md)和第八条source/wheel CLI：原创16×3 dyadic fixture、softmax/log-softmax/CE及logits梯度、显式max→shift→exp→sum→normalize中间记录。零训练更新，原Wine协议、严格hash与8 ULP诊断不变。

预声明A-B-B-A在本次AMD EPYC 9V74观察到AVX512/DEFAULT切换，但14个阶段、465个值全部逐位相同，所有捕获/控制检查通过。前次Wine干预位于另一Intel主机；原计划背景误写同一主机，现保留原文并附[事后更正](docs/fixed-input-plan-erratum.json)。保留这个负结果，不改fixture寻找差异；它只说明该固定输入下未见差异，Wine训练路径及历史CI原因仍未定位。首次捕获与最终冻结代码[四样本原始证据](results/2026-10-12-fixed-input-reviewed.json)均保留。完整检查/公开状态见[本次验证](results/2026-10-12-fixed-input-verification.md)，个人能力仍待独立自测。

```bash
python -I -m pytorch_lab.fixed_input_probe --output results/local-fixed-input.json
python -I -m pytorch_lab.fixed_input_probe --dispatch-study --output results/local-fixed-dispatch.json
```

## 2026-10-28：未完成梯度累积窗口的恢复

新增原创 [窗口内恢复合同](docs/accumulation-window-recovery.md)：在完整 microbatch 安全点保存
未归一化 pending gradients、实际分母、cursor、模型和 SGD momentum，异常后对象失效，
可信文件严格校验且不覆盖。固定顺序 CPU float64 fixture，无 live RNG、无 shuffled loader。
[原始实验](results/2026-10-28-accumulation-recovery.json)比较 3 seeds × 7 interruption cuts
的每个剩余安全点完整状态，并以错误梯度/momentum 反证仅存模型不够。
代码、46 项新增单元测试、独立配置网格审查和第十条 source/wheel CLI 组成可复核工程证据，
不等同于本人已掌握；完整结果和限制见 [验证记录](results/2026-10-28-accumulation-recovery-verification.md)。

```bash
python -I -m pytorch_lab.accumulation_recovery_cli --output results/local-recovery.json
```
