# 从训练到预测：项目主路径

日期：2026-11-12。目标是把现有工程连成可复现交付路径，不新增模型、参数搜索或 Wine 机制实验。

## 运行前固定的合同

- 复用 `train_bundle()` 的 mlp16 / seed42 / constant / epoch20，107 条训练样本，140 次更新、2,140 次训练样本访问。
- 仅训练集拟合标准化。评估同一模型在原有 36 条 validation 上的 CE、accuracy、macro-F1 和混淆矩阵。
- 原 validation 已用于历史选择，因此新输出只用于描述该固定模型与检查流程，不是新鲜泛化证据，不重选模型或 seed，不设置准确率通过门槛。
- 不计算 35 条 test 的预测或指标。完整 CSV 的读取与完整性验证仍包含其字节。
- 导出已有 `wine-model-bundle-v1` JSON 合同，包含特征顺序、类别、预处理与模型状态。用独立 Python 进程运行既有批量预测 CLI，对照评估时 logits（rtol/atol=1e-12）、类别与行序。
- 通过条件是同一产物、完整成功行数、行序、logits 和类别的一致性。失败不更改容差；原始结果和历史负结果保持原样。
- 本地 source 全套、三份独立交付套件、Ruff、非 editable wheel 主路径及精确提交 CI 分开记录。

## 一条命令完整执行

按[项目首页](../README.md#快速开始)完成安装后，从仓库根目录运行：

```bash
python -m pytorch_lab.wine_pipeline_cli --output runs/quickstart.json
```

顺序为：训练 → validation 评估 → 导出 → 新进程预测 → 完整性核对。
不下载数据、不连接服务、不使用 GPU 或付费资源。`runs/quickstart.json` 是总报告，
同名 `runs/quickstart-artifacts/` 保存：

| 文件 | 含义 |
|---|---|
| training.json | 固定训练配置与实际工作量 |
| evaluation.json | 原 validation 指标、行 ID、对应模型 digest 与限制 |
| model.json | 可独立加载的完整推理产物 |
| request.json | 36 条公开 validation 的原始特征示例 |
| predictions.json | 独立进程按行预测、错误合同与诊断计时 |

总报告记录上述各文件 SHA-256。摘要用于完整性核对，不证明来源可信。
总报告和产物目录必须均不存在；重复运行请换名。失败可能留下部分产物，程序返回非零，
不将其当成完成报告；不会自动覆盖、删除或重新使用旧目录。不承诺多文件事务或断电持久性。

## 用导出的模型预测自己的输入

以生成的 `request.json` 为格式参考，保持 13 个特征名及顺序，替换 rows 中的数值。
字段单位及输入范围见[推理产物合同](wine-model-bundle.md)。然后运行：

```bash
python -m pytorch_lab.wine_batch_inference_cli \
  --bundle runs/quickstart-artifacts/model.json \
  --request runs/quickstart-artifacts/request.json \
  --output runs/predictions-again.json
```

批量接口会保留原行序并逐项返回错误，请检查 summary 和每个 item；CLI 退出 0 表示请求已处理，
不代表每行有效。主路径使用已知有效的公开数据，并另行验证 36 行全部成功。
推理不会训练或重新拟合标准化。`model.json` 不含 optimizer 状态，不能用于恢复训练。

## 如何读结果

`all_checks_passed=true` 表示运行路径及重载预测对照通过，不表示达到了产品质量要求。
validation 指标可读，不作为本轮性能提升论据。历史 120-epoch 双模型／三种子 benchmark 与
本路径的 20-epoch 固定单模型不同，不比较它们的分数来声称提升。
既有回归测试／旧 CLI 可能重跑历史 benchmark（包含其原 test 指标），这些输出不参与本路径的选择或调参。

此次新增集成代码调用现有原创实现，不复制上游代码。数据许可和来源继续见
[第三方声明](../THIRD_PARTY_NOTICES.md)。这是助理实现并验证的工程材料；没有 HTTP 服务、
生产负载、安全审计、跨硬件逐位一致或云成本结论。

实际验收见[11 月 12 日验证记录](../results/2026-11-12-project-path-verification.md)。
