# nn.Module、Dataset/DataLoader 与小批次训练

2026-10-08 新增原创实现。原 `regression.py`、`python -m pytorch_lab` 和首轮原始结果保留不变。以下描述工程事实，学习者独立掌握仍待自测。

## 从数学参数到 nn.Module

`AffineRegressor` 在 `super().__init__()` 后把 `weight[D,K]` 与 `bias[K]` 赋为 `nn.Parameter`。因此 `named_parameters()`、`parameters()`、`state_dict()` 能发现它们；SGD 直接消费注册参数。前向仍为 XW+b，与原推导一致。它没有套用现成训练框架，也没有复制上游实现。

这里有意保留 W[D,K] 的原数学布局；`nn.Linear` 的权重布局为 [K,D]，未来替换时必须注意转置。单仿射层零初始化方便精确对照，不代表深层神经网络也应全部零初始化。

`model.train()` 和 `model.eval()` 设置模式；`torch.no_grad()` 控制是否构建求导图。两件事职责不同。这个仿射层没有 Dropout/BatchNorm，切模式不会改变其预测，但训练和评估仍使用明确的模式契约。评估在 `no_grad` 下前向，返回时恢复入口的模式；测试用 hook 观察前向时的模式与梯度开关。

## 数据、打乱与可复现

- 自定义 map-style `RegressionDataset` 实现 `__len__` 和 `__getitem__`，保存输入的独立 detached 副本，防止调用者后续修改原张量影响实验或把数据制作图带入训练。
- 160 条合成样本仍按 96/32/32 划分，没有新增下载或个人数据。拟合函数只接受训练 Dataset，均值基线也只拟合训练目标；验证/测试不参与更新、早停或选参数。
- DataLoader 使用 `num_workers=0`、`drop_last=False`。默认 batch size 20 时，每轮是 20+20+20+20+16，最后一批不丢弃。
- 数据生成、训练 loader、训练指标 loader、各留出集 loader 均使用局部 CPU Generator。即使 `shuffle=False`，DataLoader 创建迭代器也可能消耗 base seed，所以评估 loader 同样显式传 generator。
- shuffle generator 跨 epoch 持续推进，不能每轮重新设 seed；另建指标 loader，避免评估遍历改变训练打乱序列。固定配置在已验证 CPU 环境可重放，不承诺跨 PyTorch 版本/硬件逐位一致。

Dataset 副本隔离的是原始输入；其公开 `features`/`targets` 仍是可变张量，调用者应将其视为只读。此实现未声明多进程 worker、随机数据增强、GPU 或分布式确定性。

## 损失加权：尾批不能等权

每批损失 `L_b = sum(error²)/(N_b K)`。固定输出宽度 K 时，全数据 MSE 是：

`sum(N_b * L_b) / sum(N_b)`。

直接对所有 batch loss 取均值，会把只有 16 个样本的尾批当成 20 个样本的整批。测试专门使用不整除的小数据和多输出，令错误聚合给出明显不同的结果。

`train_one_epoch` 的 `mse` 是各批**更新前**的在线损失加权值：一轮中模型持续变化，它不是最终模型对全训练集的损失。历史中分别记录 `online_train_mse` 和重新评估同一最终模型得到的 `train_mse`，避免混用。

每批 `zero_grad(set_to_none=True)` → 前向 → 反向 → 有限梯度检查 → `optimizer.step()` → 有限参数检查。每轮结束清理梯度。形状、dtype、NaN/Inf、预测/平方溢出和空 loader 都有明确错误。训练失败会停止，未实现回滚或 checkpoint 恢复。

## 比较协议与实际运行

预先固定 epochs=40、learning rate=0.05、noise std=0.05；种子 42/7/123 使用同一配置，没有根据测试集选择最佳参数或最佳种子。主结果使用种子42。

1. 小批次 Module：batch size 20，40轮，200次更新，3840次训练样本访问。
2. 原手写全批次：每轮96条，40轮，40次更新，3840次训练样本访问，其他设置一致。
3. Module 全批次：每轮96条，40轮，与第2项比较参数，3个种子的最大参数误差均为0。

这是相同数据遍历次数的比较，不是相同更新次数或吞吐公平基准。原第一增量的200步/learning rate0.1结果另行保留，不能混成这里的同设置对照。

| 种子 | 小批次训练 MSE | 验证 MSE | 测试 MSE | 训练均值基线测试 MSE |
|---|---:|---:|---:|---:|
| 42 | 0.0033112462 | 0.0029567797 | 0.0044245039 | 3.8913270718 |
| 7 | 0.0028159888 | 0.0022142473 | 0.0032732652 | 6.9399290932 |
| 123 | 0.0031495186 | 0.0022639518 | 0.0019115776 | 7.3263768430 |

种子42的同设置手写全批次训练/测试 MSE 为 0.0053090310/0.0055929752。有限轮数和不同更新频率解释了差异，不能由此推出小批次普遍更优。3个合成种子也不构成真实任务鲁棒性或统计显著性证据。

原始配置、环境、每轮历史、对照和判定条件：

- [种子42](../results/2026-10-08-minibatch-cpu.json)
- [种子7](../results/2026-10-08-minibatch-seed7.json)
- [种子123](../results/2026-10-08-minibatch-seed123.json)

```bash
python -m pytorch_lab.minibatch_cli --output results/local-minibatch.json
python -m pytorch_lab.minibatch_cli --seed 7 --output results/local-minibatch-seed7.json
python -m pytorch_lab.minibatch_cli --seed 123 --output results/local-minibatch-seed123.json
```

## 官方接口核对

阅读版本为运行环境对应的 PyTorch 2.14 文档；官方页用于核对接口职责，数学推导、实现与实验为本项目原创。

- [Module](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html)：参数注册、train/eval、state_dict。
- [DataLoader/Dataset](https://docs.pytorch.org/docs/2.14/data.html)：map-style 数据、generator、shuffle、drop_last 与 worker 行为。
- [SGD](https://docs.pytorch.org/docs/2.14/generated/torch.optim.SGD.html)：注册参数、梯度清理与更新接口。
- [Reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html)：随机性控制与跨环境限制。

下一步是可信 checkpoint 的模型/优化器/随机状态恢复，再考虑 momentum/scheduler；当前尚未实现这些能力。
