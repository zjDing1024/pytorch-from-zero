# 2026-10-09 Wine increment verification

执行日期：2026-10-09 Asia/Kuala_Lumpur（2026-10-08 UTC）。本次为新增真实数据评估增量，不能用此前679项远程CI替代本次验证。

## Baseline and preservation

修改前读取当前远程main并核对全部Git blob：

- `pytorch-from-zero`: `c3b889348cd3bbf5588d08c67bff2e372e29df99`，54个文件
- `ai-learning-roadmap`: `52c2b9caf41ae79bff29b3064df7b809ff32c810`，9个文件

全部本地文件与远程一致后才开始。旧38个源码/测试/原始结果文件保持字节不变；没有覆盖任何2026-10-08实验JSON。本次没有新建仓库。

## Local checks

环境：Linux、Python3.12.14、PyTorch2.14.1+cpu、pytest9.1.1、ruff0.16.10；无GPU、无付费计算、无运行时数据下载。

```bash
ruff check .
ruff format --check .
ruff check src tests examples
ruff format --check src tests examples
python -m pip check
python -m compileall -q src tests examples
pytest -q
python -m pytorch_lab --output results/local-base.json
python -m pytorch_lab.minibatch_cli --output results/local-minibatch.json
python -m pytorch_lab.checkpoint_cli verify --output results/local-checkpoint.json
python -m pytorch_lab.scheduler_cli verify --output results/local-scheduler.json
python -m pytorch_lab.wine_cli --output results/local-wine.json
```

Ruff检查与格式检查通过；实际src/tests/examples29个Python文件，pip check无依赖冲突，compileall通过。原四条CLI的10/5/15/24项断言全部通过。新增Wine CLI完成六个120轮运行，5项数据/预算检查全部通过。新增30个聚焦测试独立运行通过（5.88秒）。原679项回归先单独通过（228.76秒）；最终新增合并后的全量709项聚合运行通过（246.06秒）。

新增测试覆盖固定数据hash/schema/counts、固定划分与泄漏、总体方差/常数列、CE手算与logit平移、混淆矩阵方向/macro-F1、局部随机性、尾批预算、确定性、验证优先选择反例、no_grad/mode恢复、完整六候选smoke、CLI无覆盖/并发占位/fsync失败/非有限输出。没有把模型胜过基线写成所有配置必须通过的正确性条件。

## Reproduction and package verification

首次正式120轮结果保存在 `2026-10-09-wine-cpu.json`。另用新Python进程重跑同一默认协议，除 `created_at_utc` 外完整JSON逐字段相同；包括模型选择、训练历史、预处理统计、全部预测/概率/错误行。复跑耗时4.44秒是该环境下的一次端到端观测，不是跨模型公平速度基准。复跑只校验可复现性，没有改变协议或搜索新参数。

在本地构建wheel，用离线无依赖安装到源码目录外，在另一工作目录导入安装包运行六候选1轮smoke。20个package成员与最终源码、wheel和安装后文件逐字节一致，数据/划分/来源/许可均在包中。最终wheel SHA-256为 `43cecc9794c2a252fa19ae16074f1d395c278079807859a7e981309568503adb`。源码的package-data声明是持久交付内容，临时构建物不提交。

## Independent review

独立审查在新建隔离环境完成五条CLI，默认120轮完整Wine结果除时间戳外完全复现。额外核对：30组标量数学用例CE最大差6.66e-16；按保存概率/标签独立重算18份集合指标；留出数据扰动不改变训练参数/归一化，测试变动不改变验证选择；未见种子999在两架构得到相同实际batch序列。核对38个旧源码/测试/原始结果保持原hash。

独立全量709项测试通过（237.98秒，1条未安装可选NumPy的警告），Ruff检查/格式、pip check和compileall通过。重建后的最终wheel在源码目录外独立安装，数据/许可与源码一致，完整六候选1轮smoke通过。独立工程验证与最终文档/冻结清单验收已完成，无未解决阻塞。审查确认66个工程文件与11个成长记录文件的全部hash和字节数准确；代码、数据、配置及CI自测试后未发生变化，所有相对文档链接有效。

## Publication boundary

本地与独立隔离环境各709项测试、五条CLI及离线wheel验证通过，最终冻结清单验收无遗留阻塞。[工程提交a274382](https://github.com/zjDing1024/pytorch-from-zero/commit/a27438230cbb2a9b515a6ca355029e5c6889ec26)已公开发布，其精确提交的[CPU checks #37808159705](https://github.com/zjDing1024/pytorch-from-zero/actions/runs/37808159705)已成功，远程709项测试、Ruff/格式和五条CLI全部通过。

工程增量完整SHA为 `a27438230cbb2a9b515a6ca355029e5c6889ec26`。发布使用基于当前远端树的单个原子提交、expected-SHA校验和非强制快进；核对全部66个公开文件Git blob与冻结清单完全一致。本次17个修改/新增文件（其中12个新增），原38个源码、测试和原始结果文件字节不变，零删除，无缓存、凭据或机器专属路径；原始Wine实验JSON未被补记覆盖。

当前补记只更新README、后续计划与本验证记录三个文档，不改变代码/数据/测试/配置/实验JSON。该文档提交也须单独检查其精确SHA的CI结果，最终工程SHA和两次CI链接将在成长仓库2026-10-09日报统一记录；不把先前绿色状态替代新提交结果。成长仓库没有工作流，不宣称其CI通过。

## Limits and interpretation

数据卡和全部结果见 [Wine协议](../docs/wine-evaluation.md)。验证CE选MLP但其测试均值略逊线性；保留既定选择，不能据测试分数重选。35条测试样本、固定单一划分、三种子训练随机性、缺少组别/时间元数据都限制外推。等更新/样本预算不是等FLOPs；没有GPU、AMP、分布式、生产部署或个人独立掌握证据。
