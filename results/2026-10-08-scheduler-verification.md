# 2026-10-08 Momentum / StepLR 验证与发布记录

执行者：AI 助理；学习者独立理解、解释和重写仍待自测。第四增量不覆盖前三阶段原始结果，不使用第三增量的绿色CI冒充本次远程验收。

## 环境与版本基线

- Linux x86_64、Python3.12.14、PyTorch2.14.1+cpu、pytest9.1.1、ruff0.16.10。
- 本轮实现复用此前隔离安装的环境，未宣称实现者重新安装依赖。独立审查环境与检查结果在审查完成后另记。
- 工程远程基线360fe210ef7397988bcf4b6e936e1ce8c700249a；成长记录基线49e7dfd7765c380ea1d58309ca6246c987fd268b。实现前逐文件核对Git blob哈希，恢复本地过期文档的已发布/远程CI闭合记录，再保存SHA-256基线。
- 原有源码、319项测试、旧结果与旧checkpoint格式保持原内容；新增调度功能通过独立模块/CLI实现。
- 默认固定CPU float64、单参数组、3→1模型、96/32/32合成划分、batch20、momentum0.8、LR0.05、StepLR(5,0.5)、40轮/7轮切点。

## 实际命令

在项目根目录运行，每个输出路径原先不存在。临时检查不写入已发布证据文件：

```bash
python -m pytest -q
ruff check .
ruff format --check .
python -m pip check
python -m compileall -q src tests examples
python -m pytorch_lab --output /tmp/scheduler-stage-original-final.json
python -m pytorch_lab.minibatch_cli --output /tmp/scheduler-stage-minibatch-final.json
python -m pytorch_lab.checkpoint_cli verify --output /tmp/scheduler-stage-checkpoint-final.json
python -m pytorch_lab.scheduler_cli verify --seed 42 --output results/2026-10-08-scheduler-cpu.json
python -m pytorch_lab.scheduler_cli verify --seed 7 --output results/2026-10-08-scheduler-seed7.json
python -m pytorch_lab.scheduler_cli verify --seed 123 --output results/2026-10-08-scheduler-seed123.json
```

聚合pytest：679项通过，1条可选NumPy警告，225.39秒。保留原319项，新增momentum82项、调度恢复236项、scheduler CLI42项。Ruff检查通过；39文件格式检查通过（25个Python、14个Markdown，本版本Ruff也检查Markdown代码块）。

原CLI10项、小批次CLI5项、旧checkpoint CLI15项断言全部通过。新scheduler CLI三种子各24项断言通过；每次实际执行四个新Python解释器，原始结果保留PID、版本与完整训练历史。pip check、compileall通过。保留PyTorch未安装可选NumPy的警告，本项目不依赖NumPy互操作。

## 逐步更新核验与预算比较

[手写参考轨迹](../docs/momentum-sgd.md)包含六组条件×六步×两个参数，共72条参数/缓冲比较；参数最大差1.1102230246251565e-16，buffer最大差2.220446049250313e-16，全部小于1e-12。另有独立从当前参数计算梯度的二次损失连续更新测试。原始JSON完整保存每步每参数的误差和buffer存在性，不只保存最终最大值。

固定LR与StepLR双方每次都做200次更新、3,840次训练样本访问，数据、初始化和打乱相同。预先固定三种子，不根据测试集挑选配置。

| 种子 | StepLR训练MSE | StepLR验证MSE | StepLR测试MSE | 固定LR测试MSE | 连续/恢复参数最大差 |
|---|---:|---:|---:|---:|---:|
| 42 | 0.0033105343 | 0.0029607174 | 0.0044082680 | 0.0043601093 | 0 |
| 7 | 0.0028153266 | 0.0022097383 | 0.0032798407 | 0.0033959516 | 0 |
| 123 | 0.0031450185 | 0.0022697370 | 0.0019144612 | 0.0019066210 | 0 |

模型、optimizer、scheduler、两个Generator、完整历史、报告与内容checksum全部相同。逐位相等只限本CPU/软件环境；三种子小型合成线性任务没有一致的策略胜者，不能外推统计显著性、真实性能或普遍优化优势。

默认第7轮诊断切点至第13轮，重置scheduler/遗漏momentum/重置shuffle的参数差分别为：

| 种子 | 重置scheduler | 遗漏momentum | 重置shuffle |
|---|---:|---:|---:|
| 42 | 0.0010718179 | 0.0006984198 | 0.0006912794 |
| 7 | 0.0004691747 | 0.0001433146 | 0.0013280306 |
| 123 | 0.0013551489 | 0.0006289497 | 0.0011247995 |

第5轮等整周期切点若保留当前optimizer LR，仅重置StepLR可能仍保持衰减相位和参数轨迹；相应测试明确检查这个例外，不声称任意遗漏都改变下一步参数。

## 审查中发现并修复的边界

独立审查实际执行 verify --epochs240 --split-epoch237，发现此时LR已极小，负对照参数差可能舍入为0，而连续/恢复仍精确一致。原先硬编码差值>1e-12因此造成错误的总体失败。靠近10000轮上限时，原诊断终点也可能超界。

修复：将状态遗漏演示明确限定为独立早期诊断切点min(请求split_epoch,7)，再训练step_size+1轮；请求的连续/恢复对照依旧使用原始切点。报告分别标明requested_recovery_split_epoch、start_epoch和end_epoch。真实237→240轮回归通过，9999切点诊断终点固定13；原第5轮相位对齐测试保留。三份正式证据已在此修复后重新执行生成，早期草稿不冒充最终证据。CLI说明也改为实际的四个进程。

## 边界与发布门槛

仅可信文件、固定CPU float64单组SGD/StepLR、完整epoch边界。新格式严格验证配置、scheduler全部状态、当前/初始LR和逐轮LR历史；旧格式不变，不支持自动迁移。未验证任意scheduler、手写优化器checkpoint、多组、mid-batch、GPU、AMP、DDP或跨版本逐位恢复。

checkpoint和可选JSON各自原子无覆盖，不是两文件事务；weights-only、checksum和大小限制不是敌对文件安全沙箱。原生测试覆盖输入/状态损坏、内容checksum、参数顺序、类型、加载隔离、失败epoch、观察不推进RNG、保存失败/竞争、边界切点与CLI错误路径。

## 独立审查与公开发布

另建全新隔离环境，重新从官方CPU wheel和固定开发依赖安装；Python3.12.14、PyTorch2.14.1+cpu、pytest9.1.1、ruff0.16.10。最终679项测试通过（222.88秒，1条可选NumPy警告），Ruff检查/39文件格式、pip check、compileall和四条CLI通过。独立复算种子42的完整正式结果，除执行时间和PID外逐项相同。

扩展审查覆盖76组momentum配置×12步×2参数，参数/buffer最大差均8.88e-16；40组调度配置/切点、6类重算checksum后的语义损坏拒绝、下溢LR往返、全局RNG与在用状态隔离。额外种子919在0/2/3/4/7/12轮切点的新进程完整报告一致；未见种子887默认验证、237→240晚期恢复及9999→10000上限恢复均通过。修复后的发布清单无未解决阻塞。

发布前后按Git blob逐文件核对：工程54文件，本次修改/新增18个，其中13个新增；原27个源码、测试和原始结果文件保持字节相同，无删除，无缓存、凭据或机器专属路径。成长记录9文件的冻结清单也完成核对。

[工程提交2ac5b1b](https://github.com/zjDing1024/pytorch-from-zero/commit/2ac5b1b7f4aa1fce36557b09ec85ce8e837cbdd8)已发布；精确提交的[CPU checks #37785797239](https://github.com/zjDing1024/pytorch-from-zero/actions/runs/37785797239)已成功结束，远程679项测试、Ruff及四条CLI全部通过。此次证据独立于第三增量的319项CI。后续补写发布记录的文档提交仍以各自Actions结果为准，不以本次绿色状态替代。

下一阶段先做独立自测，再推进公开授权真实数据与误差分析；2026-10-09不重复实现本次已完成的momentum/StepLR。未选择开源许可证，公开可读不等于已授予再分发许可。
