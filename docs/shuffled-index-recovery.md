# 受控 shuffled index stream：把排列、游标与下一轮 RNG 分开

这是原创、可安装的 **CPU batch-index 发出器**，不是 DataLoader 替代品，也没有接入训练。
它承接累积窗口恢复之后的数据顺序问题，不重复固定顺序梯度恢复实验。
已有训练实现、Wine 数据、科学结果和数值阈值均不变。

## 先定义消费语义

`next_batch()` 返回不可变的 index tuple；返回前游标已经增加。
这里的“已消费”严格指 **已发出（issued）**，不代表调用者完成了加载、forward、backward 或更新。
调用者拿到 batch 后失败或直接丢弃，发出器不知道这件事，从之后的快照恢复不会自动重发。
因此本项只能证明 index emission 的恢复；不能声称训练 exactly-once。

对象仅限单线程独占。保存点必须没有并发调用；不支持 worker、预取队列、随机变换、分布式
分片、任意 DataLoader 或任意指令崩溃恢复。私有属性不是扩展接口。
后续训练整合必须另定义“发出—应用—持久化”的共同边界，并验证模型/optimizer/梯度与游标一致。

## 状态机与尾批策略

初始化拥有一个独立 `torch.Generator(device="cpu")`，只在构造时设 seed。
第 0 轮立刻调用一次 `randperm(N)`；每次显式 `advance_epoch()` 再调用一次。
不使用 Python/global torch RNG，不接受调用者共享的 generator。
本协议不承诺与 PyTorch `RandomSampler` 或 DataLoader 的随机数消耗次数/序列相同。

| 状态/动作 | 约定 |
|---|---|
| `epoch` | 当前已准备排列的轮次，从 0 开始；不是“已训练轮数” |
| `cursor` | 当前排列前缀中已发出的 index 个数；只能在 batch 边界或最后一个尾批后 |
| `permutation` | 完整 `0..N-1` 排列，冻结当前轮剩余顺序 |
| `generator_state` | 当前排列生成**之后**的 CPU RNG bytes；负责下一轮，而非补算当前轮 |
| `drop_last=False` | 发出全部 N 个 index，最后一批可以较短 |
| `drop_last=True` | 仅发出排列前 `floor(N/B)*B` 个，尾部明确舍弃；不称作意外漏样 |
| 到达终点 | `next_batch()` 稳定抛 `StopIteration`，不隐式换轮、不消耗 RNG |
| `advance_epoch()` | 只能在当前轮耗尽后调用；未耗尽则拒绝，不允许隐式跳过 |
| 空数据/全部被 drop | 初始即耗尽，仍需显式换轮；每次换轮只准备一轮，不自动循环 |

为维持有界验收，`N` 为 0–10,000、batch size 为 1–10,000、seed 为 0–2³²−1，
轮次为 0–1,000,000。生成下一轮时先在独立候选 generator 上准备排列；即使候选抽样后异常，
现有轮次、游标、排列和 RNG 不改变。单次 index 发出不执行任何用户回调。

## 快照合同与信任边界

JSON schema 固定包含 format/version、精确 PyTorch 版本、完整配置、轮次、游标、排列、RNG bytes
和 canonical JSON 内容 SHA-256。没有 pickle，不在公开结果中保存真实数据或本机路径。

- `snapshot()` 与 `restore_snapshot()` 双向隔离可变数据；恢复建立新对象，不抽取随机数。
- 配置严格检查整数与 bool（拒绝把 bool 当 int）、界限和小写 SHA-256 格式。
- 排列必须恰好含所有合法 index 一次，游标必须符合尾批策略。
- RNG bytes 长度/类型/范围、CPU `set_state` 可接受性和 seed 一致性均检查。
- `dataset_sha256` 是**调用方提供**的有序数据/schema 指纹。本模块不读取 dataset，不能证明该指纹真实。
  恢复时传入独立持有的 `expected_config` 才能核对实际期望的数据身份与策略；省略时只验证快照内部结构。
- 文件读取至多 1,000,001 bytes，拒绝超限、重复 JSON key、无效编码或异常 schema。
  仅使用自己产生的可信文件；有界 JSON 解析不等于敌对文件安全沙箱。
- 保存复用项目自己的同目录临时文件、fsync、hard-link 发布机制，拒绝覆盖已有目标；失败清理临时文件。
  依赖支持 hard link 的本地文件系统，不宣称断电后的目录持久性。

checksum **不是身份认证，也不是过程证明**。重新计算 checksum 的合法游标移位、排列交换或
同 seed RNG 重置可能通过结构校验。必须用独立预期序列/消费协议发现其语义错误，不能靠 checksum 掩盖。

## 最小使用与复现

```python
import hashlib
from pytorch_lab.shuffle_stream import (
    ShuffleConfig,
    ShuffledBatchStream,
    save_shuffle_checkpoint,
    load_shuffle_checkpoint,
)

# 示例身份；真实接入时应从有序数据内容和 schema 计算。
identity = hashlib.sha256(b"demo:ordered-records-0-through-22:v1").hexdigest()
config = ShuffleConfig(dataset_sha256=identity, samples=23, batch_size=5, seed=42)
stream = ShuffledBatchStream(config)
issued = stream.next_batch()  # cursor 已变为 5；这不是训练成功确认。
save_shuffle_checkpoint(stream, "own-indices.json")  # 已存在则拒绝覆盖。
resumed = load_shuffle_checkpoint("own-indices.json", config)
while not resumed.exhausted:
    batch_indices = resumed.next_batch()
resumed.advance_epoch()  # 只在显式边界准备下一轮排列。
```

```bash
python -m pytest tests/test_shuffle_stream.py -q
python -I -m pytorch_lab.shuffle_stream_cli --output results/local-shuffle.json
python validation/verify_shuffle_stream.py
```

CLI 的固定协议为 seeds 42/7/123、23 samples、batch 5、两种 drop 策略、三轮，保存每个 batch
边界和显式轮次起终点，共 99 个 checkpoint。一个新 producer process 写快照，另一个新 consumer
process 加载全部快照；不是每个 checkpoint 各起一个进程。直接从独立局部 `randperm` 加 Python
切片构造预期序列，验证连续发出与每个恢复后缀，并比较最终完整状态 digest。

反证固定在发出 10 个 index 后：游标回退到 0 导致重复 10 个；前移到 15 导致缺失 5 个。
同 seed RNG 重置不改变当前轮后缀，却改变下一轮；错误 expected seed 被拒绝，错误 seed 从头运行
也不能匹配正确 oracle。故障注入不是模型性能或训练收益。

实际执行结果见[紧凑 CLI 报告](../results/2026-10-29-shuffle-stream.json)、
[独立审查](../results/2026-10-29-shuffle-stream-independent-review.json)与
[验证范围](../results/2026-10-29-shuffle-stream-verification.md)。

## 官方项目研究：TorchData StatefulDataLoader

- **项目/作者**：Meta/PyTorch 社区的 [TorchData](https://github.com/meta-pytorch/data)。
  阅读固定 v0.11.0 revision `377e64c1be69a9be6649d14c9e3664070323e464`，不假设为最新版。
- **问题与技术**：[StatefulDataLoader 文档](https://docs.pytorch.org/data/0.11/torchdata.stateful_dataloader.html)
  描述 mid-epoch state 保存/恢复；其状态边界比模型 checkpoint 更接近数据消费进度。
- **实际阅读结构**：[`stateful_dataloader.py`](https://github.com/meta-pytorch/data/blob/377e64c1be69a9be6649d14c9e3664070323e464/torchdata/stateful_dataloader/stateful_dataloader.py)
  的接口/状态与 worker 相关说明；[`sampler.py`](https://github.com/meta-pytorch/data/blob/377e64c1be69a9be6649d14c9e3664070323e464/torchdata/stateful_dataloader/sampler.py)
  的 sampler iterator、generator state 和 yielded count 责任。不声称全面审计整个框架。
- **值得学习**：把迭代器进度和 RNG 状态一起建模；batch 级进度不能简单当成训练成功。
  上游可保存此前 generator 状态，再重建排列并跳至已发出位置；本项目选择显式保存当前排列和
  post-permutation RNG，换取更直接、可检查的有限状态。这是设计取舍，不是优于上游的结论。
- **当前阶段适配**：先排除 worker/预取的复杂度，验证小型无 replacement 索引序列。
  没有增加 torchdata 依赖，也没有测试它在本运行时的安装兼容性。
- **许可**：上述固定版本 [LICENSE](https://github.com/meta-pytorch/data/blob/377e64c1be69a9be6649d14c9e3664070323e464/LICENSE)
  为 BSD-3-Clause；仅研究机制，全部实现、测试与说明原创，未复制第三方源码或代码片段。
  研究许可证不是本仓库的许可证。

[PyTorch Generator API](https://docs.pytorch.org/docs/2.14/generated/torch.Generator.html)
支持完整状态的取得/恢复，而不只重设 seed；
[DataLoader 官方文档](https://docs.pytorch.org/docs/2.14/data.html)说明 batch sampler、drop_last 与预取的职责；
[可复现性说明](https://docs.pytorch.org/docs/2.14/notes/randomness.html)提醒跨版本/平台不能保证相等。
因此本项只报告同环境 CPU 的 exact index 序列，不扩展为跨环境随机算法承诺。

## 本人自测与下一项边界

助理测试通过不代表学习者已掌握。本人需独立画出“已发出/已应用/已持久化”三个状态，预测并复现
游标回退、游标前移和 RNG 重置的不同错误，再解释 `drop_last` 的有意遗漏。
下一工程问题可以是把受控索引与现有累积训练的安全点绑定，但必须先定义共同提交边界；本项未实现。
