# 模块化小批次增量：本地验证记录

日期：2026-10-08（Asia/Kuala_Lumpur；UTC日期相同）。执行者：AI 助理。本文是命令和结果摘要，非逐字终端转录。原首轮结果和验证记录未覆盖。

## 环境与范围

Linux x86_64、Python 3.12.14、PyTorch 2.14.1+cpu、pytest 9.1.1、ruff 0.16.10。复用首轮已独立建立并验证的无 system-site-packages 的隔离环境；本增量没有新增依赖，不把复用环境说成再做了一次全新安装。

所有实验使用 CPU float64、单进程 DataLoader。无付费算力、API 或凭据。学习者掌握程度仍待独立自测。

## 聚合检查

| 命令 | 实际结果 |
|---|---|
| `python -m pytest -q` | 152 passed, 1 warning in 30.24s |
| `ruff check .` | All checks passed! |
| `ruff format --check .` | 22 files already formatted |
| `python -m pip check` | No broken requirements found. |
| `python -m compileall -q src tests examples` | 退出码0 |
| `python -m pytorch_lab --output <new-json-path>` | 原始实验CLI退出码0，10项实验断言通过 |
| `python -m pytorch_lab.minibatch_cli --output <new-json-path>` | 新CLI退出码0，5项实验断言通过 |

原有52项测试保留，新增100项API/CLI契约测试。新增内容包括：参数/state_dict注册、多输出解析梯度、输入快照与坏输入、跨epoch打乱重放、全局RNG隔离、尾批加权、训练/评估模式观察、梯度/损失/参数溢出、单步手写对照、仅训练数据参与拟合、float32多输出、真实工作量、CLI错误退出和证据防覆盖。

## 原始运行与重放

提前固定40轮、学习率0.05、batch size20、noise std0.05，分别运行种子42/7/123：

```bash
python -m pytorch_lab.minibatch_cli --output results/2026-10-08-minibatch-cpu.json
python -m pytorch_lab.minibatch_cli --seed 7 --output results/2026-10-08-minibatch-seed7.json
python -m pytorch_lab.minibatch_cli --seed 123 --output results/2026-10-08-minibatch-seed123.json
```

三次CLI均退出码0、5项断言通过；每次均保留全部每轮历史、环境、基线和比较协议。在新的临时输出路径重跑默认新增实验，整个 `training` 对象与已保存种子42文件逐项完全一致。重跑原始CLI的历史、权重、偏置与测试MSE也与首轮文件完全一致。时间戳/环境字符串不作逐字对比，最小二乘底层计算仅按容差理解。

## 已知警告与未验证范围

保留PyTorch关于可选NumPy未安装的1条警告；本项目不使用NumPy互操作。pip提示缓存目录不可写并禁用缓存，依赖检查成功。

没有验证多进程DataLoader、GPU、AMP、分布式、checkpoint恢复、真实数据或吞吐性能。三种子合成线性任务通过不代表统计显著性，也不代表用户已独立掌握。

本记录是本地证据。新增量远程CI须对照实际发布后的[Actions](https://github.com/zjDing1024/pytorch-from-zero/actions)；首轮提交的成功CI不替代新增代码的CI。
