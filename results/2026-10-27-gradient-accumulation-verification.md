# 2026-10-27 梯度累积：验证记录与边界

本页由 AI 助理记录实际执行结果，不是学习者本人已掌握的声明。阶段为 CPU float64 仿射训练循环；旧科学结果、strict / 8-ULP 标准、旧 Wine 划分均保持不变。

## 实际新增与结果

- 原创 `accumulation.py`、`accumulation_cli.py` 与 40 项新单元测试；实现按实际 target elements 归一化、尾窗口提交、optimizer cadence 和异常清梯度。
- [9 个固定 case 原始结果](2026-10-27-gradient-accumulation.json)：每次参数和 momentum 更新均满足预声明 atol=rtol=1e-12；最大绝对差 2.220446049250313e-16。两类错误控制均按设计被检测到。
- [独立解析 oracle](../scripts/verify_accumulation_analytic.py)与[结果](2026-10-27-accumulation-analytic.json)：96 配置 × 4 epoch，最大参数/momentum 差 4.440892098500626e-16，异常清理通过。该检查与单元测试覆盖部分重叠，不能与测试数量相加。
- [干净非 editable wheel 的验证摘要](2026-10-27-accumulation-wheel.json)：离线 hash-locked 安装、清空 PYTHONPATH、源码目录外 `python -I` 运行，9 个 CLI 全部通过；stdout 与输出文件 JSON 一致。两份新模块 wheel/source SHA256 一致。本地没有运行 Docker，也没有在 wheel 环境重跑完整 pytest 套件。

## 首次全量失败与窄修复

首次主套件：918 passed / 1 failed，267.46 秒。失败为历史 `test_eighth_installed_cli_and_retention` 的目录清单假设：它把 fixed_input 视为最后一个、并断言总数为 8；新入口成为第九个，因此该断言过时。不是科学数值失败或累积算法失败。

修复保持 fixed_input 为第八个入口、原 timeout 约束不变，将总数改为 9，并新增一项精确验证全部 9 个入口顺序的测试。没有降低科学阈值、删测试、跳过失败或修改生产超时。修复后相关 focused 89 项通过（11.56 秒）。

## 最终本地验收

最终主套件 **920 项通过**（257.90 秒），其中 40 项为本次新增；加上下述既有独立套件 59 + 45 + 34，本次完整测试合计 **1,058 项通过**。这是实际重新执行后的结果，不是把历史 1,018 项原封计入。新增 CLI 在源码环境的 9 个固定 case 通过；本地完整 9 CLI 的隔离运行证据来自非 editable wheel，远端源码 9 CLI / 实际容器 9 CLI 需分别看对应提交 CI。

三套既有独立套件另行通过：首步 59 项（5.79 秒）、saved-forward 45 项（5.92 秒）、交付验证器 34 项（1.166 秒）。Ruff lint 与 97 个 Python 文件 format 检查通过；pip check 无依赖冲突。可选 NumPy 缺失警告与此前环境相同，未因此添加无关依赖。公开记录不包含内部环境路径；原始日志单独保留。

源码 CI 新增 accumulation CLI；安装 wheel 检查脚本保留原 8 个入口并添加第九个。远端提交与对应 CI 结果由发布后的精确 SHA 核验，不在本地提前宣称通过。

## 重现与下一步

```bash
python -m pytest -q
python -m pytest -q tests/test_accumulation.py
python -m pytorch_lab.accumulation_cli --output results/local-accumulation.json
python scripts/verify_accumulation_analytic.py
```

对照与分母推导见[算法契约](../docs/gradient-accumulation.md)。接下来先独立写出短窗口归一化和 optimizer 更新频率，再为未来的 clipping、随机层或 checkpoint 边界单独声明契约；本增量不意味着任意模型、GPU、AMP 或 DDP 等价。
