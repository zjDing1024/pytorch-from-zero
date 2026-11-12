# pytorch-from-zero

**一个能从训练、评估走到导出与预测的 CPU PyTorch 项目。**

以公开许可的 UCI Wine 数据为小型真实任务，同时保留 Tensor / Autograd、优化器、数值诊断、训练恢复等可执行研究材料。现在先跑通完整路径，再按需要查阅机制实验。

[完整运行合同](docs/project-path.md) · [最新验证记录](results/2026-11-12-project-path-verification.md) · [历史记录](README.history.md) · [学习路线](https://github.com/zjDing1024/ai-learning-roadmap)

## 快速开始

已验证目标：Linux x86_64、CPython 3.12、PyTorch 2.14.1+cpu。下方使用目标平台的完整运行时 wheel 哈希锁；其他平台尚未验证。

```bash
git clone https://github.com/zjDing1024/pytorch-from-zero.git
cd pytorch-from-zero
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip download --only-binary=:all: --no-deps --require-hashes --index-url https://pypi.org/simple -r requirements/runtime-common.lock -d wheelhouse
python -m pip download --only-binary=:all: --no-deps --require-hashes --index-url https://download.pytorch.org/whl/cpu -r requirements/torch-cpu.lock -d wheelhouse
python -m pip install --no-index --find-links=wheelhouse --only-binary=:all: --require-hashes -r requirements/runtime-cpu.lock
python -m pip install --no-deps --no-build-isolation .
python -m pytorch_lab.wine_pipeline_cli --output runs/quickstart.json
```

首次安装需要下载官方依赖。之后主路径使用随包数据，可离线运行，无需 GPU、账户密钥或付费服务。NumPy 不是依赖，纯净环境可能有可选 NumPy 初始化警告。

这条命令实际执行：

1. **训练**：固定 mlp16 / seed42 / 20 epochs，107 条训练数据，训练集标准化，140 次更新。
2. **评估**：同一模型在已有 36 条 validation 上输出 CE、accuracy、macro-F1 与混淆矩阵，不重新选模型。
3. **导出**：保存特征顺序、标准化参数、类别和权重到可重载 JSON。
4. **预测**：启动独立 Python 进程，加载刚导出的模型，对照全部 36 行预测与原评估 logits。

查看 runs/quickstart.json 的 all_checks_passed 与 evaluation。详细文件在 runs/quickstart-artifacts/，包括 training.json、evaluation.json、model.json、request.json、predictions.json。

已有路径不会覆盖，重复运行换一个输出名。中断或失败可能留下部分目录，保留检查后用新路径重跑。

## 单独使用导出的模型

```bash
python -m pytorch_lab.wine_batch_inference_cli \
  --bundle runs/quickstart-artifacts/model.json \
  --request runs/quickstart-artifacts/request.json \
  --output runs/predictions-again.json
```

复制 request.json 的结构并替换 rows 即可提交自己的数值。必须保留特征顺序和原始测量单位；检查 summary 和逐行错误，不能只看进程退出码。参见[输入／产物合同](docs/wine-model-bundle.md)及[批量接口](docs/wine-batch-inference.md)。

## 工程材料导航

| 想了解什么 | 入口 |
|---|---|
| 训练、评估、导出与预测如何连起来 | [项目主路径](docs/project-path.md)、[集成实现](src/pytorch_lab/wine_pipeline_cli.py) |
| 数据来源、划分、仅训练集预处理 | [Wine 实现](src/pytorch_lab/wine.py)、[第三方声明](THIRD_PARTY_NOTICES.md) |
| 原始模型比较与误差分析 | [真实数据协议](docs/wine-evaluation.md) |
| 模型文件如何安全加载与校验 | [JSON 推理产物](docs/wine-model-bundle.md) |
| 每行错误处理与实际计时边界 | [批量预测](docs/wine-batch-inference.md) |
| 数学推导、训练恢复和数值负结果 | [历史首页完整归档](README.history.md)、[实验目录](experiments/) |
| 安装 wheel 与容器的验证 | [运行时合同](docs/runtime-reproducibility.md)、[CI 工作流](.github/workflows/tests.yml) |

## 测试

开发测试需要源码可导入的安装方式：

```bash
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pytest -q
ruff check --extend-exclude experiments/wine-saved-forward-2026-10-14/docs/source-before-first-capture.py .
ruff format --check --extend-exclude experiments/wine-saved-forward-2026-10-14/docs/source-before-first-capture.py .
(cd experiments/wine-first-step-2026-10-13 && python -m pytest -q)
(cd experiments/wine-saved-forward-2026-10-14 && python -m pytest -q)
(cd tools/wine-delivery && python -m unittest discover -s tests -v)
```

CI 分别运行源码测试、既有 26 条 CLI、新主路径，以及非 root／断网／只读容器内的安装 wheel。容器验证 CLI，不把它说成容器完整测试套件。

## 结果边界与历史

这是助理实现并验证的工程材料。已有 validation 被多次使用，本路径分数只描述固定模型，不是新的泛化证据；不会对 test 重新评分或按结果调参。通过条件针对产物与预测一致性，没有准确率门槛。

当前交付是本地 CPU 包，不包含 HTTP 服务、GPU／分布式训练、生产负载或安全审计结论。不承诺跨硬件逐位一致。原有日期、原始 JSON、失败证据和许可声明完整保留；原首页逐字节移至 [README.history.md](README.history.md)，不把旧状态写成最新结论。

数据沿用 UCI Wine CC BY 4.0 和随包来源声明。项目自身尚未选择开源许可证，不能把第三方数据许可当作本项目代码许可。
