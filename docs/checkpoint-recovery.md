# Epoch 边界 checkpoint：恢复同一条训练轨迹

2026-10-08 同日第三增量。**当前状态：已实现、发布并通过本地及远程验证：319项测试、静态检查、三条CLI及三种子恢复实验通过。** 下文区分恢复契约、本地实际结果与远程发布状态。原始全批次和第二增量的小批次源码、CLI、结果继续保留。

本增量由 AI 助理准备。即使工程验证通过，学习者也需完成[独立自测](learner-self-check.md)后，才能把它计入个人能力。

## 要验证什么

同一受支持环境、数据和配置下，比较：

1. 从 epoch 0 连续训练到 epoch 40。
2. 训练到 epoch 7，保存 checkpoint 并结束进程。
3. 在新进程中加载 epoch 7 checkpoint，继续训练到总计 epoch 40。

验收要求是完整恢复后与连续运行的模型、优化器状态、相关局部 RNG 状态、完整历史和结果一致；不能只比较最后一个损失是否接近。另设置“丢失 momentum”和“重置训练 shuffle 状态”两个反例，检查恢复协议确实捕获这些遗漏。

这个问题独立于上一增量的全批次/小批次公平比较。这里的连续运行、分段运行与恢复运行使用相同 momentum SGD 配置，不与无 momentum 的旧结果混作优劣比较。

## 明确缩小的支持范围

| 项目 | 本增量契约 |
|---|---|
| 数据 | 自包含合成仿射回归，训练/验证/测试为 96/32/32；配置重建后校验数据哈希 |
| 模型 | 固定 `AffineRegressor`，3 个输入、1 个输出；CPU float64 |
| 优化器 | torch SGD；默认 learning rate 0.05、momentum 0.8，无 scheduler |
| 默认运行 | seed 42、batch size 20、noise std 0.05；分段点 7、总计 40 轮 |
| DataLoader | `num_workers=0`、保留尾批；独立局部训练/指标 Generator |
| 可保存位置 | epoch 0 或一个完整 epoch 结束后；无未完成的 batch 或待处理梯度 |
| 上限 | checkpoint 文件 8 MiB；累计 epoch 不超过 10,000 |

`RecoveryConfig` 固定 seed、batch size、learning rate、momentum、noise std；momentum 必须在0和1之间，不包含端点。配置对象不可变，trainer上的 `config` 也是只读属性，不能通过重新赋值让已有loader/优化器与保存配置脱节。数据尺寸、模型结构、dtype 和运行设备是本格式的固定条件，不提供自由架构恢复。

`EpochTrainer` 拥有模型、优化器和loader的内部状态，按单线程使用；调用者不应修改私有对象。`EpochTrainer.train_to(total_epochs)` 的参数表示**累计目标 epoch**。在 epoch 7 调用 `train_to(40)` 是再训练 33 轮。epoch 0 可保存，便于检查尚未产生 momentum buffer 的合法初始状态。

失败的 epoch 可能已经更新过部分 batch；本增量不回滚内存中的参数，也不允许将这个不完整状态保存成有效 checkpoint。恢复应重新加载上一个已保存的完整 epoch，再执行下一轮。若只保存到 epoch 7，在 epoch 10 中途退出，就从 epoch 7 开始恢复。

## 恢复的是训练状态

仅保存模型权重适合重建该时刻的预测。继续 momentum SGD 还需要优化器的历史缓冲与采样顺序。官方[保存/加载教程](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)说明训练恢复需要模型及优化器等状态；本项目在固定小实验上实现完整性检查和反例。

Schema 1 保存以下内容：

- 模型 `state_dict`，包括注册的 weight/bias。
- 完整优化器 `state_dict`，包括参数组设置和已产生的 momentum buffer。
- `RecoveryConfig`、数据指纹、已完成 epoch 数和完整逐轮历史。
- 保存时的精确 PyTorch 版本。
- 训练 shuffle Generator 和指标 Generator 的状态。
- 规范化内容的 checksum，用于发现意外损坏或内容不一致。

seed 描述起点，Generator 状态描述“随机数序列已经走到哪里”。重置为同一个 seed 并不能自动继续 epoch 7 后的序列。指标 DataLoader 即使不 shuffle，也会在创建迭代器时使用 generator；因此它和训练 shuffle 分开，并一并保存。

本模型无 Dropout 等随机层，初始化和数据生成使用受控局部状态；当前训练不会消耗全局 torch RNG，也不依赖 Python、NumPy、CUDA RNG。因此不保存无关随机源。未来加入随机变换、随机模型、GPU 或 worker 后，必须重新盘点所有状态；不能直接宣称此格式仍覆盖恢复需求。PyTorch 的[可复现说明](https://docs.pytorch.org/docs/2.14/notes/randomness.html)也明确限制跨版本、平台及设备的一致性。

## 加载先验证，再返回新对象

`save_checkpoint(trainer, path)` 保存当前合法 epoch 边界。`snapshot()` 返回深拷贝的独立状态；`report()` 的留出集评估使用新建的局部 Generator，不推进训练或逐轮指标状态。`load_checkpoint(path, expected_config=None)` 返回一个新的、通过验证的 trainer；它不接受已有 trainer 作为恢复目标，因此坏文件不会部分覆盖调用者原有模型。

加载契约检查：

1. 文件大小和读取结果满足限制；内容能够以受限加载路径解码。
2. schema、字段集合、字段类型、精确 PyTorch 版本和配置合法；若传入 `expected_config`，必须匹配。
3. 模型/优化器的 tensor key、shape、dtype、设备和有限数值符合固定实验；优化器参数组及 momentum 状态一致。
4. epoch 范围、完整历史长度及各记录有效，epoch 0 与后续 epoch 的状态不同之处得到检查。
5. RNG 状态有效；数据重建后指纹吻合；内容 checksum 一致。
6. 验证成功后恢复模型、优化器、局部 Generator 和历史，返回可继续训练的新对象。

**加载失败不会自动降级到宽松格式。** 对异常字段、损坏内容或版本不兼容，应保留原文件及错误信息，查明原因；不要为了“先跑起来”跳过校验。

## 可信文件与安全边界

显式使用 `torch.load(..., weights_only=True, map_location="cpu")`，不回退到 `weights_only=False`，不添加自定义类 allowlist。官方[序列化说明](https://docs.pytorch.org/docs/2.14/notes/serialization.html)解释了受限加载器的作用及残余风险；[`torch.load` 参考](https://docs.pytorch.org/docs/2.14/generated/torch.load.html)也提醒谨慎对待不可信来源。

仅加载自己产生或来源可信的文件。`weights_only=True` 和 8 MiB 文件上限都不是任意不可信输入的安全沙箱：较小的序列化文件仍可能导致过量资源消耗；结构和数值检查也不证明来源可信。

checksum 是完整性自检，**不提供身份认证或防篡改保证**。能修改文件的人也可以重算 checksum。数据哈希用于核对实验输入，不是文件发布者身份的证明。仓库不接收、下载或执行第三方 checkpoint 来演示这一功能。

## 保存与文件系统边界

保存流程是在目标文件的同一目录创建临时文件，写入并 flush/fsync，随后用 hard link 以 no-clobber 方式建立最终路径，最后清理临时路径。

- 已有目标文件不能覆盖；需要新的文件名。
- 在本次支持的 Linux 本地文件系统中，最终路径要么尚不存在，要么指向完整写好的文件，避免把半写文件暴露为完成品。
- 这描述原子可见性，不承诺断电持久性、所有操作系统、网络文件系统或分布式存储的一致语义。
- 进程被强制终止可能留下临时文件。该文件不能当成已完成 checkpoint；下一次训练使用上一个有效最终文件。
- CLI的checkpoint与可选JSON报告分别原子保存，**不是两个文件共同提交的事务**。checkpoint成功后若报告写入失败，命令会报错，但有效checkpoint可能已经保留；重试前应检查已有文件，不要假定它们一起回滚。

## CLI 与复现实验

以下为本增量的复现入口。用新的临时目录避开已有证据文件：

```bash
run_dir=$(mktemp -d)
python -m pytorch_lab.checkpoint_cli train \
  --epochs 7 --checkpoint "$run_dir/epoch7.pt"
python -m pytorch_lab.checkpoint_cli resume \
  --checkpoint "$run_dir/epoch7.pt" --epochs 40 \
  --save-checkpoint "$run_dir/epoch40.pt"
python -m pytorch_lab.checkpoint_cli verify \
  --seed 42 --output results/local-checkpoint.json
```

`verify` 使用独立新进程运行连续、分段和恢复路径，默认分段点 7、总计 40；可用 `--epochs` 与 `--split-epoch` 改变比较位置，但必须满足 `0 < split_epoch < epochs`。训练和加载API支持epoch0，负对照实验需要已经发生过更新，因此不使用epoch0作为分段点。

正常恢复与两个错误恢复对照使用同一配置；负对照只比较分段点之后的下一轮。`verify` 的临时checkpoint会随实验目录清理，JSON保留配置、检查、历史、字节数与指纹。需要保留可恢复文件时，使用 `train --checkpoint` 和 `resume --save-checkpoint`。`resume` 不带保存路径时仅继续训练并报告，不另存最终checkpoint。JSON输出拒绝覆盖已有文件；每次复核请选择新路径。

## 已执行结果与验证边界

2026-10-08在Linux、Python3.12.14、PyTorch2.14.1+cpu中运行种子42/7/123。每个种子的连续40轮与7轮后恢复到40轮都使用三个独立Python进程，15项实验断言全部通过：模型和优化器张量逐位一致，两个Generator、完整历史、报告及规范化内容checksum相同，最终参数最大差距均为0。三种子预先使用同一配置，不根据测试集选参或挑选种子。

| 种子 | 训练MSE | 验证MSE | 测试MSE | 连续/恢复参数最大差距 |
|---|---:|---:|---:|---:|
| [42](../results/2026-10-08-checkpoint-cpu.json) | 0.0033463182 | 0.0028868156 | 0.0043601093 | 0 |
| [7](../results/2026-10-08-checkpoint-seed7.json) | 0.0028570668 | 0.0022004279 | 0.0033959516 | 0 |
| [123](../results/2026-10-08-checkpoint-seed123.json) | 0.0031506270 | 0.0022858665 | 0.0019066210 | 0 |

三个种子在第7轮保存的checkpoint均为14,037字节。两个负对照都从该分段点开始，只运行到第8轮：

| 种子 | 遗漏momentum后的参数最大差距 | 重置shuffle后的参数最大差距 |
|---|---:|---:|
| 42 | 0.0144504611 | 0.0036747311 |
| 7 | 0.0191405113 | 0.0048320308 |
| 123 | 0.0481323629 | 0.0065692380 |

这些差异证明相关状态在本例中影响恢复轨迹，不构成任意模型/数据的充分性证明。3个合成种子不证明统计稳健性或跨环境逐位一致。

完整本地pytest为319项通过（原152项、checkpoint契约151项、checkpoint CLI 16项），用时77.76秒；Ruff检查/29文件格式检查、pip check、compileall及原始/小批次/checkpoint三条CLI全部通过。独立审查检查了fsync失败注入和真实三进程同路径写入竞争，并推动只读config修复。修复后独立复跑319项测试通过（78.55秒），lint/29文件格式、pip check、compileall和种子42精确重放也通过。最终文档/清单验收和发布已完成，详见[本次验证记录](../results/2026-10-08-checkpoint-verification.md)。工程增量 [9a5707696c064523ee95bcaddb49ab1f07dc4467](https://github.com/zjDing1024/pytorch-from-zero/commit/9a5707696c064523ee95bcaddb49ab1f07dc4467) 已发布；精确提交对应的 [CPU checks #37751219715](https://github.com/zjDing1024/pytorch-from-zero/actions/runs/37751219715) 已成功完成，319项测试与三条CLI通过。

## 未覆盖与下一步

本增量不支持 mid-batch 游标、恢复任意 DataLoader worker、随机数据增强、GPU/CUDA RNG、AMP scaler、DDP、scheduler 或任意架构。跨环境逐位一致、生产故障恢复、性能及安全审计均不在结论中。

本阶段本地恢复检查已完成，下一工程增量是**手写 momentum SGD 与 torch SGD 的逐步状态对照，再加入 scheduler 的顺序及状态恢复实验**。这与当前“使用 torch SGD momentum 并恢复其状态”是不同能力。学习者需独立解释 momentum 丢失和 shuffle 重置的后果、重写最小恢复路径，并设计自己的失败测试。
