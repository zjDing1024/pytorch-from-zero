# 2026-10-08 Epoch checkpoint 本地验证记录

执行者：AI 助理；用户独立自测仍未完成。这里记录第三增量的实际运行，不覆盖前两阶段证据。

## 环境与范围

- Linux x86_64、Python 3.12.14、PyTorch 2.14.1+cpu；pytest 9.1.1、ruff 0.16.10。
- 复用上一阶段已隔离安装的环境，本轮未重新下载或宣称新做一次全新依赖安装。
- 原创新增 checkpoint 状态协议、CLI 与恢复对照；原来的数学、全批次和无 momentum 小批次源文件、测试、原始JSON及验证记录保持原哈希。
- 固定 CPU float64、合成96/32/32划分、3→1仿射模型、单进程/单worker0、完整epoch边界；没有GPU/AMP/DDP或mid-batch验证。

## 实际命令与结果

以下命令在项目根目录、已激活实验环境运行，输出使用此前不存在的路径：

```bash
ruff check .
ruff format --check .
python -m pip check
python -m compileall -q src tests examples
pytest -q
python -m pytorch_lab --output /tmp/checkpoint-stage-original-cli.json
python -m pytorch_lab.minibatch_cli --output /tmp/checkpoint-stage-minibatch-cli.json
python -m pytorch_lab.checkpoint_cli verify --seed 42 --output results/2026-10-08-checkpoint-cpu.json
python -m pytorch_lab.checkpoint_cli verify --seed 7 --output results/2026-10-08-checkpoint-seed7.json
python -m pytorch_lab.checkpoint_cli verify --seed 123 --output results/2026-10-08-checkpoint-seed123.json
```

- Ruff检查通过；29个文件格式检查通过（18个Python、11个Markdown；本版本Ruff同时检查Markdown）。
- pip check通过；compileall通过。
- 最终修复后pytest聚合319项测试通过，77.76秒：保留152项、新增151项核心恢复测试及16项CLI测试。
- 原CLI的10项、无momentum小批次CLI的5项断言再次通过。
- 新checkpoint CLI三种子各15项断言全部通过。只读配置修复后已再执行三种子，训练历史、检查、负对照和内容哈希与保存的原始记录完全一致。每次verify确实执行三个新的Python进程，原始JSON保留PID及实际输出。
- 保留1条未安装可选NumPy的PyTorch警告，本项目不使用NumPy互操作。

## 关键实测

固定40轮、在7轮后保存并结束进程、另起进程继续至40轮；学习率0.05、momentum0.8、batch20、float64。每轮保留尾批16条。

| seed | 最终训练MSE | 测试MSE | 连续/恢复最大参数差 | 丢失momentum下一轮参数差 | 重置shuffle下一轮参数差 |
|---|---:|---:|---:|---:|---:|
| 42 | 0.003346318184384481 | 0.0043601093251817566 | 0 | 0.014450461073759024 | 0.003674731052935476 |
| 7 | 0.002857066755759132 | 0.0033959515501669167 | 0 | 0.019140511255903547 | 0.004832030816348087 |
| 123 | 0.0031506269861967494 | 0.0019066209577537597 | 0 | 0.048132362913458016 | 0.006569238032294589 |

完整恢复的模型、优化器、两个局部Generator状态、逐轮历史及报告全部相同，规范化内容哈希一致。负对照比较的是第8轮，故没有用后续收敛掩盖状态遗漏。所有比较都发生在同一软件/CPU环境，不承诺跨环境逐位一致，也不把新momentum结果与旧SGD混作公平优劣评估。

## 自动化失败场景

151项新增核心契约测试覆盖多seed/分段点、epoch0、重复恢复、全局RNG隔离、snapshot独立性、深层字段/配置/模型及momentum张量/RNG/历史损坏、强制weights-only CPU加载、大小限制、坏版本、no-clobber竞争、序列化/link失败清理和中断epoch禁止保存。CLI集成测试另覆盖真正的独立进程、坏参数、累计epoch语义、输出冲突、损坏加载及verify断言。

独立审查对最终源码再次执行319项测试（78.55秒）、Ruff29文件检查/格式、pip check、compileall和种子42新进程复现实验，全部通过；记录的训练历史、15项检查、负对照、checkpoint字节数及文件/内容哈希一致。额外执行了fsync失败注入（新路径/已有路径）及真实三个进程竞争同一路径的探针，均通过；这些是审查探针，不计入上述pytest数量。发现公开配置重赋值可能与loader状态不一致后，已把`trainer.config`改为只读属性并新增回归测试。

加载返回新的私有候选对象；验证失败不会将候选状态写入调用者已有trainer。保存为同目录临时文件加flush/fsync/hardlink提交，已有目标字节不变。checkpoint和可选JSON各自原子提交，不是两个文件的共同事务；JSON写入失败后可能已有有效checkpoint。不承诺断电目录持久性或网络文件系统语义。只加载自己生成或可信来源checkpoint；weights-only、8MiB上限和checksum不能替代不可信文件沙箱或身份认证。

## 尚未代表的证据

上述章节记录本地执行和独立复跑；完整发布清单及文档已通过最终独立审查。精确提交的发布与远程CI证据见下方补记，未用本地通过替代远程结果。学习者的独立实现、数学解释和失败定位仍待自测。下一工程增量是手写momentum逐步对照及scheduler顺序/状态恢复，而非重复宣称本次epoch恢复尚未实现。

## 发布与远程 CI 补记

工程增量 [9a5707696c064523ee95bcaddb49ab1f07dc4467](https://github.com/zjDing1024/pytorch-from-zero/commit/9a5707696c064523ee95bcaddb49ab1f07dc4467) 已发布；精确提交对应的 [CPU checks #37751219715](https://github.com/zjDing1024/pytorch-from-zero/actions/runs/37751219715) 已成功完成，319项测试与三条CLI通过。

- 发布时已核对工程 main 指向上述实现提交，15个变更文件原子提交，全部41个工程文件的Git blob哈希与审查通过的清单一致，26个未改文件原样保留。
- 远程执行环境：Ubuntu 24.04.5、Python 3.12.15、PyTorch 2.14.1+cpu。依赖/项目安装、Ruff检查与29文件格式检查、319项测试（60.83秒）及三条CLI全部成功；原CLI10项、小批次5项、checkpoint15项断言全部为true。
- 远程保留1条可选NumPy未安装警告，以及官方Actions的Node 20/24与punycode弃用提示，未影响成功结论。
- 本节是实际CI完成后的独立文档补记；实现、测试和三份原始checkpoint JSON未因补记更改。后续纯文档提交的CI与此实现提交的证据分开记录。
