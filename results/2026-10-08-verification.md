# 最终本地验证记录

日期：2026-10-08 UTC。执行者：AI 助理。本文是命令与结果摘要，省略机器绝对路径，不是逐字终端转录。

## 安装与环境

Python 3.12.14、PyTorch 2.14.1+cpu、pytest 9.1.1、ruff 0.16.10；Linux x86_64、CPU、float64。按 README 创建全新隔离 venv，未启用 system-site-packages，从官方 CPU 源安装 requirements.txt，安装 requirements-dev.txt，然后 `pip install --no-deps -e .`。安装成功。

## 最终代码检查

| 命令 | 实际结果 |
|---|---|
| `python -m pytest -q` | 52 passed, 1 warning in 8.09s |
| `ruff check .` | All checks passed! |
| `ruff format --check .` | 格式检查通过，无文件需要重排 |
| `python -m pip check` | No broken requirements found. |
| `python -m compileall -q src tests examples` | 退出码 0 |
| `python -m pytorch_lab --output <new-json-path>` | 退出码 0，10 项实验断言通过 |

最后一次 CLI 与已保存 `2026-10-08-cpu.json` 对比：SGD 损失历史、权重、偏置和测试 MSE 完全一致。最小二乘参考允许浮点容差；时间戳与环境记录不做逐字比较。

独立审查也在另一个新 venv 按 README 完成安装，并确认 52 项测试、lint、格式、pip check 与 CLI 通过。发现并修复的边界：极大学习率可令损失溢出；新增了有限输入溢出、训练溢出及 CLI 友好报错三个回归测试。

## 警告与未验证范围

纯净环境的 PyTorch 提示可选 NumPy 初始化失败，因为未安装 NumPy。本实验不使用 NumPy 互操作，这条警告保留披露；未通过隐藏警告来表示环境完全无提示。安装器还提示不可写缓存并自动禁用缓存，不影响本次安装。

没有验证 CUDA、AMP、分布式、真实数据质量、生产负载或跨平台兼容。检查耗时仅记录本次运行，不是性能基准。远程 CI 是另一项证据，请查看 GitHub Actions 当前运行。
