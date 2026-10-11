# 2026-10-11 数值回归验证记录

日期：2026-10-11，Asia/Kuala_Lumpur。本文保留本地冻结与失败历史，并补记实际公开提交及精确SHA远程CI。

## 起点与不变项

工程基线`17aed9e5ea934807477652cabbc35a4de8c86a7e`，成长基线`b2fe45be071883791d15fea6b9c5020a2f9547ff`。开工逐一核对远端80/13个文件，先同步过期本地副本，再实现本增量。原Wine协议、32个旧测试/结果文件、旧数据/许可和原训练源码字节保留。原runtime唯一生产源码变化为_worker可选返回完整Wine对象，默认输出、result_digest和严格重复检查逻辑未变。两处旧测试文件例外为test_runtime_probe.py的CLI外层监督预算公式，以及test_installed_checker.py对已消失进程的/proc观测竞态修复；原784个测试case和全部科学断言保留，原版本可由基线commit复查。

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

- 聚焦测试48项通过（45.44秒）：42个新增测试 + 6个旧安装检查器测试。该聚焦结果发生在监督预算修复前；首次发布前的旧测试改动限于CLI helper预算，发布后另修复/proc观测竞态，均见后文。
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

本机仍无Docker/Podman等容器引擎；本增量的Docker build/run与artifact上传已在下述精确SHA的GitHub CI执行，本机不另作容器执行声明。未验证GPU、分布式、跨硬件一般误差界、性能收益、动态原生库完整身份或个人独立掌握。

## 发布状态

本地监督预算修复后的828项完整测试通过（906.70秒），先前825/1失败记录保留。公开feature提交为[`6765eac6ddbf00bb13d89f02e15810c20b4f1a1e`](https://github.com/zjDing1024/pytorch-from-zero/commit/6765eac6ddbf00bb13d89f02e15810c20b4f1a1e)。其[首轮CI #38131487107](https://github.com/zjDing1024/pytorch-from-zero/actions/runs/38131487107)实际失败：source为827 passed / 1 failed（167.96秒），container构建、7条已安装CLI及artifact上传成功。失败为旧`test_timeout_terminates_real_descendant`读取`/proc/<pid>/stat`时进程消失，引发`ProcessLookupError`（ESRCH）；source后续7条CLI被跳过，always-upload因没有文件也失败。没有把成功的container当作整个run成功。

修复仅修改`tests/test_installed_checker.py`的观测辅助函数：在原有ENOENT之外接受ESRCH，仍拒绝持续存活的后代并传播权限/I/O等无关错误；生产清理逻辑、所有deadline和科学门保持不变。新增3个针对性回归用例，本地独立11项检查通过（1.07秒），另有9项独立探针；恢复旧处理器的内存变异确实被新用例拒绝，真实“先打开/proc、再终止/回收、最后读取”也复现ESRCH。它们不累加到pytest计数。

[修复后工程提交025a1d7](https://github.com/zjDing1024/pytorch-from-zero/commit/025a1d7da1b68ccd1965b9bfb355eba643ee7505)的[CPU checks #38132126861](https://github.com/zjDing1024/pytorch-from-zero/actions/runs/38132126861)已成功：source job通过831项测试（186.64秒、1条可选NumPy警告）、Ruff/格式和7条CLI；container job实际完成Linux amd64镜像构建及非root、只读、断网的7条已安装wheel CLI，两路完整数值artifact均已上传并下载核验。完整修复SHA：`025a1d7da1b68ccd1965b9bfb355eba643ee7505`。这是独立于本地828项和失败feature run的完整831项运行；没有拼接局部结果。首次失败commit/run仍可追溯。

两次公开更新均使用expected-SHA及非强制快进；逐一核对工程98个远端Git blob，零删除。feature为27个修改/新增（18新增），随后仅上述1个旧测试文件修复；52个其余受保护旧源码/测试/结果保持原字节。生产源代码、固定计划及已提交的所有原始JSON在后续修复和文档补记中不变。

### 完整产物核验

修复run的[source artifact](https://github.com/zjDing1024/pytorch-from-zero/actions/runs/38132126861/artifacts/11698719921)和[container artifact](https://github.com/zjDing1024/pytorch-from-zero/actions/runs/38132126861/artifacts/11699980377)已下载。ZIP SHA-256分别为`823269ce601b30b5b35831c43af6c2f5594441c0f89767253c37e5d76b673455`、`abc7a8fd1de46d61b03bd5b79d70622512e559444a11c1dd308ee5049860c9f0`，与GitHub元数据一致。source包含6个匹配local-ci-*的JSON（baseline的local-ci.json没有包含在此匹配中）；container包含完整installed报告。实际7条CLI是否执行以各job步骤为准。

source的2个runtime + 2个numerical样本、container的3个runtime + 2个numerical样本共9份，每份23个包代码/数据/许可hash全部与冻结源码相同。该run两路科学hash均为b7ab9686…；数值捕获记录AMD EPYC 9V74、AVX512。相比本地Intel Xeon/AVX512的28ce527e…，这再次表明capability标签单独不足以解释输出差异，不能据此给历史run归因，也不把本次相同hash升级为跨平台保证。首轮成功container另记录AMD EPYC 7763/AVX2与b7ab9686…。

artifact配置保留30天，本次到期时间为2026-11-10 UTC；永久实验材料仍是仓库results中的正式JSON与失败快照。未把带机器路径的整份CI日志复制进仓库。后续文档补记提交须单独验证自己的source/container CI，最终精确SHA与结果同步到成长仓库2026-10-11日报。
