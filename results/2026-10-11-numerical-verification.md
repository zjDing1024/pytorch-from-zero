# 2026-10-11 数值回归验证记录

日期：Asia/Kuala_Lumpur。本文记录本地冻结阶段，远程发布/CI必须在精确提交执行后单独补记。

## 起点与不变项

工程基线`17aed9e5ea934807477652cabbc35a4de8c86a7e`，成长基线`b2fe45be071883791d15fea6b9c5020a2f9547ff`。开工逐一核对远端80/13个文件，先同步过期本地副本，再实现本增量。原Wine协议、33个旧测试/结果文件、旧数据/许可和原训练源码字节保留。原runtime唯一生产源码变化为_worker可选返回完整Wine对象，默认输出、result_digest和严格重复检查逻辑未变。唯一旧测试文件例外是test_runtime_probe.py的CLI外层监督预算公式；原784个测试case和全部科学断言保留，原版本可由基线commit复查。

## 冻结前协议

`docs/numerical-regression-plan.json` SHA-256为`f2e643ebc643406e380b9266e1e52cc7d8ff8bef8e3a578aad9893a1ac1b99c8`，在新Wine实验前固定且后续不变。仅指定连续输出允许8个相邻有限binary64数；没有非零absolute floor；其余元数据/配置/结构/type/离散/计数派生值精确。严格canonical hash始终另报。

## 原始结果与明确失败

| 证据 | 严格hash | exact字段 | 连续8步诊断 | 关键量 |
|---|---|---|---|---|
| 历史b7ab / 28ce | 失败 | 通过 | 失败 | 1299个float不同；max abs3.608224830031759e-16，max75 ULP，932项超限 |
| 最终A / A重复 | 通过 | 通过 | 通过 | 同主机正常AVX512，两次28ce… |
| 最终B / B重复 | 通过 | 通过 | 通过 | default干预，两次d6e828… |
| 最终A / B | 失败 | 通过 | 失败 | 1441个float不同；max abs4.440892098500626e-16，max144 ULP，1024项超限 |

完整科学hash：

- 历史本地：`b7ab96864bf8265cc5ec54c7318cb65bc78cbb0404174ef868a803ce3eb75fc1`
- 历史最终CI / 当前A：`28ce527e3828b3d49a30a4be246ae4c1c19a38f20f4eb561450484a88ba94ab5`
- 当前B：`d6e828f353d775a6757cf0334d1b53bd073edc73489562dfcccd02b7f8348c09`

最终正式产物：`2026-10-11-prior-ci-wine.json`、`2026-10-11-prior-reviewed.json`、`2026-10-11-dispatch-reviewed.json`、`2026-10-11-installed-reviewed.json`。来源、wheel hash和各个预冻结快照的地位见[provenance JSON](2026-10-11-numerical-provenance.json)。历史来源为[run37961087696 / job113923794801](https://github.com/zjDing1024/pytorch-from-zero/actions/runs/37961087696/job/113923794801)：仅去除日志每行时间前缀，保留JSON数字字面类型；重算科学hash确认与已公开28ce一致。不公开整个CI日志。

A-B-B-A每样本包含6次固定120轮拟合，5,040更新/77,040样本访问。CPU为XeonPlatinum8573C；torch2.14.1+cpu、Python3.12.14、GCC13.3、oneMKL2024.2、oneDNN3.12.0。完整环境以原始JSON为准。最终报告的5项capture/control检查均通过，独立重算每个比较报告与保存值完全一致，所有包文件hash与最终代码一致。

本结果仅建立“本机ATen dispatch干预改变科学输出”；DEFAULT结果是第三个hash，不能解释历史b7ab/28ce之差。没有用小绝对差宣称8步通过，没有调阈值，也没有性能收益或个人掌握声明。

## 实际检查

工作目录为工程仓库；使用CPU开发venv和先前建立的隔离wheel依赖venv。后者include-system-site-packages=false，本次重新构建/安装项目wheel；不声称重新下载过整个依赖闭包。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q
ruff check .
ruff format --check .
python -m compileall -q src tests scripts
python -m pip check
python -m pytorch_lab.numerical_probe --dispatch-study --output results/2026-10-11-dispatch-reviewed.json
python -m pip wheel --no-index --no-deps --no-build-isolation . -w dist
# 在隔离的依赖环境中安装该wheel，然后：
python -m pip install --no-index --no-deps --force-reinstall dist/pytorch_from_zero_lab-0.1.0-py3-none-any.whl
python -I scripts/check_installed.py --output results/2026-10-11-installed-reviewed.json
```

- 聚焦测试48项通过（45.44秒）：42个新增测试 + 6个旧安装检查器测试。该聚焦结果发生在监督预算修复前；后续唯一旧测试文件改动限于CLI helper预算，见后文。
- Ruff通过；全部39个拟发布Python文件格式检查通过（排除本地wheel构建副本）；compileall与两个环境pip check通过。
- 最终已安装wheel的7条检查器CLI实际通过：baseline、minibatch、checkpoint、scheduler、wine、runtime、numerical。外部临时目录、`-I`、noneditable与distribution导入位置检查都执行。
- 安装报告中3个runtime样本及2个完整数值样本的全部包hash均等于最终源码，科学hash均28ce…；不是仅看版本号相同。
- 已安装比较器另在源码目录外执行：同文件strict比较退出0；历史差异比较退出1并保留失败报告，且报告与源码最终比较一致。
- 最终项目wheel SHA-256：`4e1d8daabf010d9caca99124e9a20d5821b862bc193da99d415929e3e87b361d`。这标识此构建产物，不保证重建字节一致。
- 工作流PyYAML可解析，16个run块全部通过bash -n。查阅[官方upload-artifact v4输入](https://github.com/actions/upload-artifact/blob/v4/README.md)，配置的30天retention和缺文件失败与公开接口一致。
- 独立审查完成34项边界/CLI探针（其中ULP属性覆盖10,000个随机bit-pattern，核验其中9,998个有限值）、9项留存/退出探针及真实1-epoch四新进程干预。它们是独立审查覆盖，不累加到仓库pytest数量。
- 独立最终完整运行实际结果为825 passed、1 failed，952.45秒；826项全部执行、0 skipped，源码冻结hash未变。唯一失败为旧test_runtime_probe.py::test_real_two_fresh_processes的外层subprocess.TimeoutExpired（60秒），无stdout/stderr或measurement JSON。该原测试、原60秒超时保持不变，单独重跑1 passed（29.40秒，调用29.32秒）。全量失败与独立通过分开记录，不能合并成826通过。测试显式OMP/MKL/OpenBLAS线程上限为1，没有跳过测试、改断言或放宽容差。旧scheduler多解释器用例实测103.34/92.13/71.70/58.41/53.74秒，显示当前运行有显著启动/执行延迟，但不把这些观测宣称为唯一成因的证明。此前一次冻结运行因执行环境session恢复中断，仅保存758个完成点、无summary，同样不作全量通过证据。较早两个混合修改版本的全量尝试已取消/不用于验收；其中一个日志出现1个F，随后以中断退出130，未产生最终traceback/计数。按收集顺序映射，该位置对应旧的“超时不产生文件”测试，而新的失败留存设计会写partial报告；这是顺序映射推断，不当作已捕获traceback。最终冻结完整运行不得跳过该项；只接受整套实际结果。

最终验收：监督预算修复后完整828项通过（906.70秒、退出0、0失败/错误/跳过），生产代码与正式实验/wheel匹配；原失败记录和孤立复测保持分开，未拼接成通过结果。

## 审查发现与修复

1. 独立审查发现文档strict比较示例遗漏必需output参数，已修正。
2. 发现后续child失败会丢弃之前的完整捕获。现捕获器保存已完成样本与脱敏failure，CLI退出2；安装检查器保留先前报告及失败子命令已写的结构化结果，再返回非零。测试覆盖失败发生在成功样本之后。
3. 一个重跑与修复并行，包hash控制门失败，`dispatch-interrupted.json`保留；科学hash重复本身不能让它作为有效受控试验。
4. 一个安装预检查跨CLI发生重新安装，虽然子检查status通过，runtime/numerical包hash不一致；`installed-prefreeze.json`明确无效作最终安装证明。随后冻结代码重建并完成全新一轮7CLI，逐样本核对hash一致。

5. 根据真实失败检查监督关系，旧测试将整个CLI限制为60秒，但该命令实际允许2个顺序child各300秒（默认值），外层预算小于所监督的合法内层预算。只修改测试helper为repeats × configured_child_timeout + 60秒启动/报告余量；runtime默认repeats3、numerical默认2、dispatch模式4。生产CLI的300秒默认、显式0.001秒真实超时负对照、严格hash/8步诊断及原断言均未改变。另加2个测试检查预算公式和命令参数保持原样；预算与真实CLI/超时负对照共6项通过（43.66秒）；既没有通过改生产超时来过关，也没有隐藏原完整失败。修复后的完整重跑实际828 passed，906.70秒，0 failed / 0 errors / 0 skipped，退出0；JUnit计数与冻结源码hash复核一致。该次旧runtime真实双进程用例16.33秒，新完整捕获用例15.42秒。

不覆盖预冻结原始记录，provenance逐一解释哪些是正式证据、哪些被后续冻结结果替代。进程被外部强杀/磁盘写入失败仍可能失去内存中证据，不能把本次受控错误留存宣称为任意故障下的持久化保证。

## 尚未验证

本机无Docker/Podman等容器引擎；本增量Docker build/run与artifact上传尚待发布后精确SHA的GitHub CI执行。先前第六增量绿灯不证明新提交。未验证GPU、分布式、跨硬件一般误差界、性能收益、动态原生库完整身份或个人独立掌握。

## 发布状态

本次仍是本地冻结/独立审查阶段，尚未向远端提交。监督预算修复前完整本地测试并非全绿；修复后828项完整重跑已全部通过（906.70秒），先前825/1失败记录完整保留，不以单项重跑覆盖它。发布时必须用基线HEAD保护、核对完整远端Git blob清单，并跟踪该精确工程SHA的source与container CI；失败先诊断修复，不把数值调查未通过的结果删掉或改称成功。
