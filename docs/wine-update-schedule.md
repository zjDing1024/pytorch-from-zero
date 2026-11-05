# Wine 的 optimizer-update 学习率时钟

## 问题与实现范围

本项承接真实 Wine 的批次双预算与成本实验，不重复 10 月 8 日合成数据的 epoch StepLR。
固定训练集、初始化、样本排列和 14 次 SGD 更新，比较 constant 与每 7 次更新减半的学习率。
恢复验收只在同一进程，把可信内部快照放入新模型、优化器和 scheduler 对象；不是通用
checkpoint 文件格式，也没有把本实验接入既有 JointTrainer。

[执行前协议](wine-update-schedule-plan.json)冻结两架构、三种子、batch16/32、14 次更新、
初始 LR=.05、momentum=.9、coupled weight decay=.0001。只计算训练集终点 CE，不访问
验证／测试评分，不根据历史测试结果调参。

## 时钟推导

令 u 为从 1 开始的已执行更新序号。每次先 optimizer.step，再 scheduler.step，则：

- 本次使用 `lr_used(u) = .05 × .5**floor((u-1)/7)`。
- 本次结束后 `lr_next(u) = .05 × .5**floor(u/7)`。
- 第 7 次仍使用 .05；第 7 次完成后 .025 供第 8 次使用。
- 第 14 次仍使用 .025；结束时存 .0125，但本实验没有执行第 15 次。

StepLR 参数名 `last_epoch` 不会自动知道 DataLoader 的 epoch。调用者每次更新后调用，
它在本实验中就计数更新。初始化本身建立初始 scheduler 状态，不能把构造计成一次训练。
每步原始 trace 保存实际 LR、后继 LR、参数／momentum 哈希、批次行索引、训练 loss
和实际更新范数。实际记录与独立公式核对，不以最终 loss 相近替代时钟验证。

固定与衰减分支的前 7 次参数应一致，第 7 次的后继 LR 已不同。缩小学习率不等于
重新开始动量；momentum buffer 仍累积过去梯度，coupled decay 仍进入 SGD 方向。

## 数据与预算

标准化仍只用 107 条训练样本拟合；shuffle 排列由局部 Generator 生成，两个 LR 分支
共享同一批序，调用者全局 RNG 不变。batch16 的 14 更新访问 214 样本，batch32 的
14 更新访问 385 样本，都是保留尾批后的真实计数。同一次比较中的 LR 分支预算相同；
跨 batch 的相同更新数不等样本数，也不等时间，不把它解释为纯梯度噪声因果实验。

24 条主轨迹共 336 次更新。12 条 scheduled 主轨迹分别从第 6、7、8 次边界恢复，
每配置后缀总计 8+7+6=21 次，另 252 更新。总计 588 更新、12486 训练访问。
每条主轨迹和恢复后缀只做一次完整训练集终点评估，共 60 次、6420 评估访问。
单元测试、独立审查与交付会重复执行，以上是单次主实验而非整个开发成本。

## 恢复合同与故障边界

快照深拷贝模型、SGD完整状态、scheduler完整状态、trace与固定批序，trace长度即下一
批游标。构造模型、optimizer、scheduler后，再加载scheduler和optimizer状态，避免
scheduler初始化覆盖保存的LR。每一条剩余更新边界都对照完整状态，不只比较最终参数。
切点6/7/8跨越衰减前后。输入必须是同一份可信训练张量，不做任意输入指纹认证或恶意
snapshot校验；不接受外部 checkpoint 文件，不承诺跨进程或断电持久性。

执行更新前把对象标记为不可用；只有 backward、optimizer.step、scheduler.step、trace
追加和梯度清理全部成功才重新可用。中途异常后禁止 step/snapshot/report，须恢复上一
完整快照。它不是内存自动回滚。固定模型没有随机前向或 worker 状态，不声称覆盖这些。

负对照分别漏 momentum、丢 scheduler 相位、额外 scheduler.step。丢 scheduler 相位在
cut6会推迟衰减：下一步参数仍可相同，但 lr_next 不同，随后分歧。cut7重置而保留当前
LR可在本短前缀内维持相同参数轨迹；完整scheduler状态仍不同，不能要求所有错误都立即
导致参数差。测试保留这个反例。

## 实际结果与运行

[完整原始结果](../results/2026-11-06-wine-update-schedule.json)保存12配置全部轨迹。
所有预算、LR公式、前七步一致性和36条恢复后缀逐边界完整状态检查通过。12配置的固定
LR终点训练CE均低于本次衰减分支；没有继续调参寻找逆转。这只是14步训练前缀的结果，
不是调度器普遍更差、固定LR泛化更好或最终收敛更优的证据。没有计时排名或GPU结果。

```bash
python -m pytorch_lab.wine_update_schedule_cli --output results/local-update-schedule.json
python -m pytest -q tests/test_wine_update_schedule*.py
```

CLI拒绝覆盖既有输出，严格JSON不允许NaN/Inf，检查失败返回非零。source CI与已安装
wheel验收均接入此入口；具体通过状态以[验证记录](../results/2026-11-06-update-schedule-verification.md)
与精确提交CI为准。

## 官方项目研究与许可证

研究对象是 PyTorch，由 PyTorch 社区／基金会维护，解决张量计算、自动微分和训练状态
管理。与当前阶段最相关的是 `torch/optim/optimizer.py`、`sgd.py` 与 `lr_scheduler.py`
的职责边界：参数组LR归optimizer，调度计数归scheduler，调用频率由训练循环决定。

- [固定版本 LRScheduler / StepLR 源码](https://github.com/pytorch/pytorch/blob/v2.14.1/torch/optim/lr_scheduler.py)：读取构造初始化、状态字典、调用顺序及StepLR相位逻辑。
- [官方优化器与调度说明](https://docs.pytorch.org/docs/2.14/optim.html)。
- [官方 StepLR API](https://docs.pytorch.org/docs/2.14/generated/torch.optim.lr_scheduler.StepLR.html)。
- [PyTorch v2.14.1 LICENSE](https://github.com/pytorch/pytorch/blob/v2.14.1/LICENSE)：BSD-style，保留版权／许可／免责声明，不以贡献者名称背书。
- Wine数据的UCI来源及CC BY 4.0署名沿用 [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md)。

本实验代码、协议、测试与文字均为本任务原创；未复制上游实现。公开仓库未新增整体
开源许可，不能把可读取等同任意再分发授权。

## 求职证据与下一步

可展示训练循环的时钟定义、对照实验、状态恢复边界和负对照设计。助理运行结果不代表
学习者已独立掌握。巩固题：推导第7/8步LR，解释为何恢复必须保留动量、为何cut7的相位
错误可能不改变短期参数。下一项应冻结更长训练预算，对照训练／验证曲线与终点选择规则；
不再增加同类短前缀scheduler变体，也不按历史测试集结果选策略。
