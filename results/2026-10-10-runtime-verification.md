# 2026-10-10 runtime verification

本记录区分本地验证、独立审查、公开提交与精确提交CI。代码/实验由助理完成，个人独立技能仍待自测。

## 起点与保存范围

修改前逐一核对当前远端工程66个文件、成长11个文件的Git blob，零差异。工程基线`bd5ba6883bee133f2fdbd68ebda168edea0ae767`，成长基线`1d680ae56851d036b0ccc17b542e3e87ac29a384`。不删除旧文件；所有旧src、tests、数据、许可证及历史原始JSON保持字节不变。

## 实际运行环境与安装

CPython3.12.14、Linux x86_64、glibc2.41、torch2.14.1+cpu。builder使用setuptools84.0.0、pip25.0.1。现有开发venv可见额外系统包（含NumPy2.3.5），不称为最小或完全隔离安装。

另建venv不启用system-site-packages；10个运行依赖均从本地wheelhouse按完整目标平台hash-lock验证安装，随后`--no-deps`安装自产wheel。没有用已安装全局依赖假装重建。最小环境12个distribution为10个运行依赖加项目与pip；无可选NumPy，PyTorch提示已披露。`pip check`通过。pip/bootstrap不是hash-lock的一部分，不能因此宣称全系统可重复构建。

[产物JSON](2026-10-10-runtime-artifacts.json)保留实际依赖wheel文件名/字节数/SHA-256、来源索引和最终项目wheel哈希。项目wheel重建不保证ZIP字节相同；包内实际源码/数据/许可证哈希与源码一致。

## 已执行检查

- 首轮全量778项测试通过（230.84秒）；审查后新增6项进程检查器测试，修复后全量784项通过（237.07秒）。最终冻结代码完整重验784项通过（217.02秒）；不是把聚焦检查当作全量结果。
- 新增75项聚焦测试通过（6.54秒）。真实后代进程已启动，再由超时进程组终止；退出、缺失执行文件、平台范围、嵌套timeout预算均有检查。修复/proc读取与reap竞争后6项再次通过（1.05秒）。
- Ruff检查与格式检查通过；src/tests/scripts compileall通过；最小运行环境pip check通过。
- 已安装wheel于临时工作目录执行6条CLI：baseline、minibatch、checkpoint verify、scheduler verify、完整120轮Wine及3次runtime probe。每条退出0、stdout与保存JSON一致、all_checks_passed为true。
- 实际负对照：已下载filelock wheel翻转一字节后，pip --require-hashes拒绝；本地wheelhouse缺少要求产物时失败；源码editable安装被检查器拒绝。没有依赖联网回退。
- [source正式报告](2026-10-10-runtime-source-cpu.json)和[wheel正式报告](2026-10-10-runtime-wheel-cpu.json)各3次120轮，每样本5,040次更新/77,040样本访问。六个科学输出与前日完整Wine结果同hash；未重选架构或调参。

资源计量用workload wall、process CPU、进程生命周期高水位RSS和child end-to-end wall四个指标；范围、单位、原始数据及环境差异见[协议](../docs/runtime-reproducibility.md)。初始导入/线程设置排除在workload计时之外，期间的懒加载仍计入。共享宿主存在其它测试活动；不作速度、内存优化或显著性结论。

## 独立审查与修复

独立审查在单独的干净环境再次完成安装与六条CLI，核对最终wheel中的21个包文件与源码一致，并验证损坏/缺失wheel、editable、已有文件及断链符号链接的失败路径。

审查发现初始外层timeout只杀直接父进程，可能留下嵌套runtime worker。改为Linux独立进程组，timeout时killpg并communicate回收；3×60秒子预算小于300秒外层预算。新增真实子/孙进程测试证明超时后无继续运行的后代。独立全量784项通过（235.07秒），最终75项聚焦检查与Ruff通过，/proc竞争修正后6项另行通过（1.04秒）。正式项目wheel和source/wheel资源JSON都在修复后重新生成，旧试跑不作为最终公开结果。

最终文档/全文件冻结清单独立验收通过：工程80个文件、成长13个文件全部SHA-256/Git blob/字节数准确；分别19与7个修改/新增文件，其中14与2个新增，零删除。43个旧源码/测试/原始JSON与原数据/许可保持原字节，发布前初始冻结阶段的40个实现/配置文件与当时清单一致；相对链接全部有效，无凭据或机器路径发现。无未解决工程阻塞；容器执行的后续真实CI证据见下节。

## 容器与CI边界

本机查无docker、podman、buildah、buildctl或runc；没有安装引擎、更改安全设置或宣称本地容器执行。

已通过官方Docker registry按digest获取基础index，验证`sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254`及linux/amd64子manifest `sha256:9c47360a2a0355e2da18516d0b1c2126ec22c195d2185e97347c9d98398c5bef`。该本地检查只解析metadata，没有在本机拉取镜像层或build/run；后续真实CI执行见下段。无漏洞扫描或最新安全版本保证。

CI最小两路径：source job使用同一runtime hash-lock跑全量pytest、Ruff与6条CLI；installed-cpu-container job实际Docker build并以UID/GID10001、断网/只读/Linux capabilities全部移除/资源上限运行wheel六条CLI。容器job不跑全量pytest。[修复后工程提交e162eaf](https://github.com/zjDing1024/pytorch-from-zero/commit/e162eafd654e44bfea3473c391491fa071b443d0)的[CPU checks #37959687015](https://github.com/zjDing1024/pytorch-from-zero/actions/runs/37959687015)已成功：source job通过784项测试（243.42秒，1条可选NumPy警告）、Ruff/格式和6条CLI；container job实际构建并以非root、只读文件系统和断网方式完成6条已安装wheel CLI。

## Publication boundary

首个[工程提交114a5a1](https://github.com/zjDing1024/pytorch-from-zero/commit/114a5a1857975ca7d20a99a45b26def291ac8ab2)对应[运行#37958785497](https://github.com/zjDing1024/pytorch-from-zero/actions/runs/37958785497)在工作流解析阶段失败，没有任何source/container job执行。原因是YAML plain run scalar包含`--only-binary=:all:`后的冒号空格。此前Python全量检查与代码审查没有覆盖工作流解析，这个缺口保留在记录中。

只把该run命令改为block scalar，79个其它工程文件逐字节不变。修复前fixture被PyYAML与actionlint拒绝，修复后通过PyYAML结构检查、官方actionlint1.7.12与全部15个run块的bash -n；独立审查确认修复范围。[修复后工程提交e162eaf](https://github.com/zjDing1024/pytorch-from-zero/commit/e162eafd654e44bfea3473c391491fa071b443d0)的[CPU checks #37959687015](https://github.com/zjDing1024/pytorch-from-zero/actions/runs/37959687015)已成功：source job通过784项测试（243.42秒，1条可选NumPy警告）、Ruff/格式和6条CLI；container job实际构建并以非root、只读文件系统和断网方式完成6条已安装wheel CLI。工程完整SHA为`e162eafd654e44bfea3473c391491fa071b443d0`；源测试job113919046724，容器job113919046357。

首次发布及修复均使用expected-SHA检查和非强制快进，远程80个文件与对应冻结清单逐一核对。此后本次补记只更新README、后续计划、运行时协议与本验证记录4份Markdown，不改变源码、数据、测试、配置或原始结果JSON。补记提交还要单独观察其精确SHA的两个CI job，最终提交与CI记录写入成长仓库日报；不借用旧绿色结论。成长仓库没有CI。

跨环境边界也实际观察到了：本地source/wheel六次及远程source两次的科学结果hash为`b7ab96864bf8265cc5ec54c7318cb65bc78cbb0404174ef868a803ce3eb75fc1`，而该次CI容器三次均为`28ce527e3828b3d49a30a4be246ae4c1c19a38f20f4eb561450484a88ba94ab5`。三组都在各自环境内重复一致，但容器科学JSON的canonical字节与本地不同。21个包代码/数据/许可证hash和10个锁定依赖版本相同；本地glibc2.41、source CI的Python3.12.15/glibc2.39、容器Python3.12.14/glibc2.36不同，不能据此断言差异成因。现有容器日志只保存科学hash，未保留完整科学JSON，尚未定位差异字段/量级，也不声称误差低于某个容差。

无GPU/AMP/DDP、付费算力、生产部署、全平台锁或个人技能已掌握的声明。
