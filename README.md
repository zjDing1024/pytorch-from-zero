# pytorch-from-zero

**A CPU-first lab connecting PyTorch behavior, mathematical derivations, and executable tests.**

当前增量：Tensor 布局、Autograd、三路梯度校验、带独立留出集的线性回归，以及新增 nn.Module、Dataset/DataLoader 和小批次 SGD。代码原创编写，官方源码只用于阅读与核对。

> 证据边界：本增量由 AI 助理实现并在 CPU 环境验证。它提供可复查的工程材料，用户的独立理解与实现能力仍需 [自测](docs/learner-self-check.md)。当前项目尚不足以证明生产级训练、GPU 优化或分布式能力。

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
  minibatch_cli.py 新增量独立 CLI；保留原始实验入口
tests/            数学正确性、坏输入、重复运行、CLI 集成测试
examples/         独立运行入口
docs/             推导、源码阅读、质量边界、自测与扩展计划
results/          已执行的原始结果和检查日志
```

`affine_mse` 使用 `X[N,D] @ W[D,K] + b[K]`，损失对 `N*K` 个元素取均值。手写解析梯度、Autograd 与中心差分相互核对；多输出测试专门检查分母遗漏 `K` 的错误。训练循环直接更新叶子参数，并在每步清理梯度。输入拒绝空维度、形状不符、不同 dtype、整数和非有限数值。

## 环境配置

已验证：Linux x86_64、Python 3.12.14、PyTorch 2.14.1+cpu、pytest 9.1.1、ruff 0.16.10。项目声明 Python >=3.11，但其他 Python/操作系统组合尚未验证。无需数据下载、GPU、付费 API 或账户密钥。

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

## 性能分析

- 每个训练步矩阵乘法量级为 `O(NDK)`；存储以输入、参数和自动微分中间量为主。当前训练规模很小，不用于吞吐结论。
- 解析梯度与反向传播复用矩阵运算。中心差分对每个参数做两次前向，成本约为 `2(DK+K)` 次前向，仅适合校验。
- 输入的有限数值检查每次扫描张量；它优先服务教学安全。在 GPU 上会引入同步成本，需要重新设计校验边界后再做性能优化。
- 未执行 GPU、混合精度、显存或多机基准；没有速度提升声明。

## 文档、质量与下一步

- [数学、设计与失败模式](docs/design.md)
- [固定版本源码阅读](docs/source-reading.md)
- [学习者自测](docs/learner-self-check.md)
- [后续工程计划](docs/next-increments.md)
- [贡献与证据规范](CONTRIBUTING.md)

本次已完成 `nn.Module`、`Dataset/DataLoader` 与小批次 SGD。下一步先做独立自测，再实现 checkpoint 的模型、优化器与随机状态恢复；调度器、混合精度和分布式仍是后续计划。

2026-10-08 本地验证：首轮52项测试与独立隔离安装检查通过；新增量合计152项测试、lint、格式和两条CLI通过，详见[本次验证记录](results/2026-10-08-minibatch-verification.md)。远程 CI 的最新状态请查看 [Actions](https://github.com/zjDing1024/pytorch-from-zero/actions)，本地结果不替代远程运行证据。未选择开源许可证，不应将公开可读等同于已授予再分发许可。
