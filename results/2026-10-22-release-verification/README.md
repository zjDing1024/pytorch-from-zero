# 2026-10-22：四批工程成果重新执行验收

日期：2026-10-22，UTC+08:00。这是本轮实际执行记录，不是重复引用 10 月 17 日验收，也不是远端发布或精确提交 CI 成功证明。

## 验证对象与环境

恢复并整合固定输入诊断、首步定位、saved-forward 以及交付验证器。原候选 245 文件均核对字节 SHA-256；新增本记录和导航更新，原实验输入、科学结果、Python 实现、历史源码快照和 CI workflow 不改。测试缓存、安装元数据和临时构建目录不进入提交。

实际 Python 3.12.14 / Linux x86_64 / PyTorch 2.14.1+cpu、pytest 9.1.1、Ruff 0.16.10。完整版本和 PyTorch 构建配置见 [environment.json](environment.json)。使用新建 venv 共享本机既有系统包，另从官方 PyPI / PyTorch CPU index 安装依赖；包含 NumPy 2.3.5。这不是 pristine、离线或全传递依赖锁定安装。pip check 实际返回 No broken requirements found。

## 已执行结果

| 检查 | 实际结果 | 原始证据 |
|---|---|---|
| 主项目 pytest | 首轮及稳定安装后的最终复跑均 880 通过 | [首轮](main-pytest.txt)、[最终复跑](main-final-pytest.txt) |
| 首步独立 pytest | 59 通过 | [日志](first-step-pytest.txt) |
| saved-forward 独立 pytest | 45 通过 | [日志](saved-forward-pytest.txt) |
| 交付验证器 unittest | 34 通过 | [日志](validator-unittest.txt) |
| Ruff check / format --check | lint 通过，90 文件无需格式修改 | [lint](ruff-check.txt)、[format](ruff-format.txt) |
| source 主项目 CLI | 8/8 退出 0，各 stdout 与对应保存 JSON 一致 | [摘要](source-cli-summary.json) |
| 非 editable wheel | 8/8 隔离 CLI 通过，分发版本 0.1.0 | [完整报告](installed-wheel.json) |
| 原两份输入 ZIP 的静态交付验证 | 通过，28 / 41 成员，完整 hash 和 manifest 检查 | [报告](delivery-static.json) |

四套测试合计 1,018，重复执行不算新增测试用例。主项目 CLI 范围为 baseline、minibatch、checkpoint、scheduler、wine、runtime、numerical、fixed_input。它们不覆盖三个独立脚本包的完整科学捕获。

source 的完整输出分别见 [baseline](source-baseline.json)、[minibatch](source-minibatch.json)、[checkpoint](source-checkpoint.json)、[scheduler](source-scheduler.json)、[wine](source-wine.json)、[runtime](source-runtime.json)、[numerical](source-numerical.json)、[fixed_input](source-fixed_input.json)。没有覆盖任何旧结果文件。

构建 wheel 的 SHA-256 为 `853ecfff585720b8235e4973aa970329a5d6e747433fd408648de63aae6e96c0`，文件名 `pytorch_from_zero_lab-0.1.0-py3-none-any.whl`。这是本次构建物身份，不是可复现构建的宣称。安装后使用仓库已有 scripts/check_installed.py，从无源码的工作目录和 `-I` 子进程验证八条 CLI。

## 可复核命令

以下从项目根目录执行；先按项目依赖说明准备 CPU 环境。输出目录必须是新目录，不能覆盖既有科学记录。

```bash
python -m pip install --no-deps --no-build-isolation -e .
python -m pytest -q
(cd experiments/wine-first-step-2026-10-13 && python -m pytest -q)
(cd experiments/wine-saved-forward-2026-10-14 && python -m pytest -q)
(cd tools/wine-delivery && python -m unittest discover -s tests -v)
ruff check --extend-exclude experiments/wine-saved-forward-2026-10-14/docs/source-before-first-capture.py .
ruff format --check --extend-exclude experiments/wine-saved-forward-2026-10-14/docs/source-before-first-capture.py .
python tools/wine-delivery/validate_delivery.py --output /tmp/fresh-delivery-audit
python -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/fresh-wheel .
python -m pip install --no-deps --force-reinstall /tmp/fresh-wheel/*.whl
# 在另一个无源码工作目录中，使用 scripts/check_installed.py 的绝对路径：
python /absolute/path/to/scripts/check_installed.py --output /tmp/fresh-installed-result.json
```

source 八条命令及参数取自已有 scripts/check_installed.py 的 COMMANDS，仅使用当前可编辑安装，不调用其 noneditable 检查。每条命令独立保存 JSON，并核对退出码、stdout 与文件一致性。

## 不可扩大解释的边界

- numerical / fixed_input 的 all_checks_passed 是捕获和控制有效性，不把跨条件 strict hash 或 8 ULP 结果暗改成通过。
- 本轮未重新执行首步和 saved-forward 完整 A-B-B-A 科学捕获，未重新归因历史跨主机差异，原失败保留。
- 当前环境没有 Docker 命令，本轮没有运行新容器；也没有验证 GPU、分布式、其他平台或生产部署。旧容器 / CI 记录不算本轮证据。
- 本地文件、测试、wheel 和远端 commit / CI 是分开的证据。本文不预填未来 SHA，不宣称新的 GitHub 发布已经完成。
- 本轮工作由助理执行；学习者独立实现、解释和排错能力仍待本人自测。

最终主套件复跑时 wheel 已完成安装，全程不再切换安装方式；源码由 pytest 配置的 src 路径加载，隔离子进程使用相同版本的已安装 wheel。最终复跑摘要：`880 passed in 197.52s (0:03:17)`。重复运行不增加测试用例总数。
