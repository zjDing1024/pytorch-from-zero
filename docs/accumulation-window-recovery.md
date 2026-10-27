# 累积窗口内恢复：先定义安全点，再验证重放

本增量是助理原创实现与运行，不代表学习者已独立掌握。它接续
[样本加权梯度累积](gradient-accumulation.md)，不是重复 epoch 边界 checkpoint。

## 预声明合同

仅支持本项目固定顺序的 CPU float64 3→2 affine 合成 fixture、SGD momentum、
同一 PyTorch 版本和执行环境。默认 23 个样本、microbatch 5、窗口 3，3 epochs。
fixture 使用隔离的局部 generator；训练不消费随机数，模型零初始化，故没有待恢复的
全局 RNG。seed、生成后的 features/targets 指纹和顺序一起绑定配置。
不支持 shuffled/arbitrary loaders、workers、dropout、BatchNorm、GPU、AMP、DDP。

安全点是 `step_microbatch()` **完整成功返回之后**：backward、必要的 optimizer step、
清梯度和 cursor 推进构成一个操作。窗口内 `.grad` 保持 **loss sum 的未归一化梯度和**；
只在更新前除以实际 target elements。这里每个样本两个目标，不能把样本数直接当分母。
末尾不足窗口立即 flush；已完成窗口的 pending counts 全为 0，梯度为 None。

checkpoint 保存 model、SGD momentum 和参数顺序、epoch、下一未消费样本 cursor、
已提交更新数、pending microbatch/sample/element counts、pending gradients、配置与指纹。
每个 snapshot 深复制，不引用活跃参数。schema、精确类型（拒绝 bool 冒充 int）、shape、
dtype、有限性、配置、数据、cursor/counts 与 optimizer 状态全部验证后才构造返回 trainer。
这不是通用 optimizer state loader；SGD groups 的允许结构明确固定。
规模上限为 1,000 样本、1,000 microbatch size、1,000 accumulation steps、100 epochs；
这是有界教学实验，未提供 loss history/metric 聚合接口，也不声称恢复未记录的统计。

若 backward/update 期间异常，原对象失效，不能继续或保存；不承诺 rollback，必须重载
此前有效快照。保存使用同目录临时文件、文件 fsync 和 no-clobber hard link；不覆盖旧文件。
只读取自己可信文件，`weights_only=True`、CPU 映射和 8 MiB 上限；不声称能沙箱化恶意文件。
checksum 检测意外损坏，不认证来源；能改写合法 tensor 并重算 checksum 的人仍可制造假状态。
不承诺任意指令崩溃恢复、并发写安全或断电后目录持久性。

## 最小 API 使用

```python
from pytorch_lab.accumulation_recovery import (
    WindowTrainer,
    save_window_checkpoint,
    load_window_checkpoint,
)

trainer = WindowTrainer()
for _ in range(4):
    trainer.step_microbatch()  # 第四批处于末尾未完成窗口，已有 momentum。
save_window_checkpoint(trainer, "own-window.pt")  # 文件存在则拒绝覆盖。
resumed = load_window_checkpoint("own-window.pt", trainer.config)
resumed.train_to(3)
assert resumed.epoch == 3 and resumed.cursor == 0
```

请勿修改 `_model`/`_optimizer` 等私有成员或共享 trainer 做并发操作；
它们只在白盒测试中用于故障注入，非公开扩展接口。

## 验收与反证

在执行之前，CLI 固定 seeds 42/7/123、cuts 0/1/3/4/5/6/15（全程 15 microbatches）。
覆盖初始、窗口内、完整窗口后、已有 momentum 的末尾短窗口内、epoch 边界及训练结束。
连续生产进程与每次恢复的新进程比较**每个剩余安全点的完整 typed-state digest**，
包含参数、momentum、pending gradients 和所有 counters，而非仅比较最终 loss。
要求同环境 bit-exact；不要求 torch.save 文件字节一致，也不宣称跨版本/硬件精确一致。

缺失 pending gradients、错误元素数、错误 cursor 即使重算 checksum 仍应拒绝。
另对合法形状的梯度和 momentum 归零：loader 无法判断其历史真实性，应允许结构校验，
但下一次更新必须呈现非零参数漂移。此反证区分“结构合法”和“训练过程正确”。
独立 harness 还覆盖单微批、单窗口、窗口大于 epoch 等配置，和此前无状态累积实现比较。

```bash
python -m pytest tests/test_accumulation_recovery.py -q
python -I -m pytorch_lab.accumulation_recovery_cli --output results/local-recovery.json
python validation/verify_accumulation_recovery.py
```

输出拒绝覆盖已有 JSON；换一个新文件名进行重复实验。实验结果与完整检查见
[原始 JSON](../results/2026-10-28-accumulation-recovery.json)、
[独立审查](../results/2026-10-28-accumulation-recovery-independent-review.json)、
[验证记录](../results/2026-10-28-accumulation-recovery-verification.md)。

## 开源研究与原创边界

研究 [Accelerate checkpoint 指南](https://huggingface.co/docs/accelerate/usage_guides/checkpoint)
及固定 revision
[`checkpointing.py`](https://github.com/huggingface/accelerate/blob/8c83b448d19f6356d4facd65c1d4763bf9fa51b2/src/accelerate/checkpointing.py)，
其 [Apache-2.0 license](https://github.com/huggingface/accelerate/blob/8c83b448d19f6356d4facd65c1d4763bf9fa51b2/LICENSE)
及 model/optimizer/RNG/dataloader 分离保存的设计提醒我们：model state_dict 不等于未完成窗口状态。
该常规保存路径没有单独保存 parameter.grad，不等于断言框架有缺陷或所有使用方式不能恢复。
[PyTorch Optimizer.state_dict 官方文档](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.state_dict.html)
说明参数按位置匹配；本实现因此明确检查顺序、group 配置和 momentum shape。
未复制上游代码、注释或数据；研究许可证不自动成为本仓库许可证。

## 本人自测与下一步

独立解释为什么 tail window 分母是 16 而不是 30；为什么只保存 model/optimizer 会丢失
pending gradients；在一个合法 checkpoint 中注入 cursor 或梯度故障并预测拒绝/漂移结果。
本人尚未进行这些自测。下一步先独立复现，然后设计 shuffled sampler 的 order/cursor/RNG
契约与恢复实验；不能仅删除当前限制就声称支持一般 DataLoader。
