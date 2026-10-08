# UCI Wine：可审计的真实数据分类协议

日期：2026-10-09，Asia/Kuala_Lumpur。这个增量把已有的训练机制用于一份公开授权的真实数据；目标是数据来源、无泄漏拟合、公平比较与错误分析，而不是把容易的数据集包装成先进模型成绩。保留所有原始合成实验和恢复协议，不把 Wine 分类器接入尚未验证的通用 checkpoint。

## 1. 数据卡与授权

- 数据：[UCI Wine recognition](https://archive.ics.uci.edu/dataset/109/wine)，不是 Wine Quality。178 份葡萄酒化学分析、13 个数值特征、3 个栽培品种类别，计数 59/71/48；没有已知缺失值。
- 引用：Aeberhard, S. & Forina, M. (1992). Wine [Dataset]. UCI Machine Learning Repository. [DOI 10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J)。UCI 当前许可为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。保留署名、来源、许可和格式变更说明，不暗示作者认可本项目。
- 原始来源：[wine.names](https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.names) 的 Forina/PARVUS 与 Stefan Aeberhard 记录。数据属于同一意大利地区、三类品种；没有足够的组别/年份元数据支持更广泛外推。
- 实际随包分发的文件是官方 scikit-learn [固定提交 CSV](https://github.com/scikit-learn/scikit-learn/blob/e316dbeeebfd8f38cf293d6443ce81aaa33686d3/sklearn/datasets/data/wine_data.csv)。文件 11,157 字节，Git blob 为 `6c7fe81952aa6129023730ced4581b42ecd085af`，SHA-256 为 `10e8a802908b34f86e5da8ce962f3c806694bc98450a18f61851af59f324bede`。本项目没有改动这些字节。
- scikit-learn 格式增加元数据首行，把目标移动到最后一列，并将 UCI 的 1/2/3 类别编码为 0/1/2；这里的行 ID 是去掉首行后的零基原始行号。完整来源元数据在 `src/pytorch_lab/data/wine_provenance.json`。
- 随数据保留 [scikit-learn BSD-3-Clause 原文](../src/pytorch_lab/data/LICENSE.scikit-learn)；这是上游软件通知，和 UCI 的数据许可分别记录。原创训练代码没有复制上游实现。
- 上游 `wine_data.rst` 部分署名与其自身原始来源段及 UCI 不一致，本项目使用 UCI 推荐引用，不沿用其中 Fisher/Marshall/1988 的矛盾信息。

运行不下载数据、不安装 scikit-learn/NumPy。打包时包含 CSV、划分、来源与许可文件；加载先校验固定 SHA-256，再验证维度、有限值和类别计数。哈希证明与固定文件一致，不单独证明科学数据质量。

## 2. 预先固定的实验设计

在第一次训练和测试评分之前固定以下选择；没有根据测试分数改学习率、模型或划分。

| 项目 | 协议 |
|---|---|
| 划分 | 按类别用 SHA-256 对原始行 ID 排序；盐 `wine-split-v1:20261009`；每类60%训练、20%验证（四舍五入到最近整数，0.5向上），其余测试 |
| 结果 | 107/36/35，类计数训练35/43/29、验证12/14/10、测试12/14/9 |
| 归一化 | 仅107条训练数据拟合逐列均值和总体标准差；零方差列的scale设1；同一变换用于所有集合 |
| 候选 | 线性 logits 13→3（42参数）；13→16→ReLU→3 MLP（275参数） |
| 训练 | CPU float64，SGD，lr0.05、momentum0.9、weight_decay1e-4；120轮，batch16，不丢尾批 |
| 种子 | 42/7/123同时控制局部权重初始化与局部shuffle；数据划分保持不变 |
| 选择 | 比较三个种子的最终验证集交叉熵均值；相同则依次选较少参数、字典序名称；没有early stopping或best-seed选择 |
| 最终测试 | 选择完成后，报告所有预先声明候选和基线的测试结果；测试结果不反馈至选择 |
| 基线 | 只用训练类频率的概率预测；argmax同时给出训练多数类的硬预测 |

`wine_split.json` 保存完整行 ID，并在每次运行时与算法再生成的划分核对。审计要求每行只属于一个集合、全集被覆盖、每个集合都有三类，且相同特征行不能跨集合。本次没有完全重复特征行。没有组别/时间元数据时，这些检查仍不能排除潜在的群组相关性或样本偏差。

模型只在 `fit_candidate(train_x, train_y, ...)` 接收训练数据；归一化拟合也只接收训练特征。`select_architecture` 只接收验证损失，不接收测试指标。test 张量的归一化/预测在该函数返回后才开始。数据完整性与划分审计可以读取全数据，它们不估计模型或预处理参数。

每轮训练7次更新，尾批11条；每个模型/种子共840次更新、12,840次样本访问。相同种子使用相同batch顺序。相同更新与样本预算不等于相同FLOPs、模型容量或耗时；本实验不是吞吐基准。

## 3. 损失、指标与实现边界

传递原始 `[N,3]` logits 和 `[N]` int64 类别给 `F.cross_entropy`，不先做softmax。数学上单个样本损失为 logsumexp(logits) 减真实类别logit。softmax只在报告概率时使用。

在线训练loss按样本数加权；它使用每批更新前参数，不等同于最终模型的训练loss。最终评估使用eval与no_grad，完整集合的平均交叉熵。混淆矩阵行是真实类别、列是预测类别；macro-F1先逐类算F1再等权平均，零分母约定返回0。保留每个集合的行 ID、预测、概率，及按错误预测置信度降序排列的误分类列表。

训练类先验基线的交叉熵使用非零类概率，不把带错误的one-hot多数预测说成有限交叉熵。softmax分数未经校准，不能作为可靠性保证。

## 4. 首次执行结果

原始数据：[2026-10-09-wine-cpu.json](../results/2026-10-09-wine-cpu.json)。默认协议一次完成六个训练运行；保留所有种子，不挑最好结果。

| 模型 | 验证CE均值 | 测试准确率均值 | 测试macro-F1均值 | 测试CE均值 |
|---|---:|---:|---:|---:|
| 训练类先验/多数类 | 不参与模型选择 | 0.400000 | 0.190476 | 1.083496 |
| 线性，42参数 | 0.017577 | 0.971429 | 0.970110 | 0.085390 |
| MLP，275参数（验证选中） | 0.014833 | 0.961905 | 0.959791 | 0.111148 |

两个模型在三个种子的训练和验证准确率都是100%，但这不保证测试没有错误。MLP验证CE更低，因此按预先规则保留为选中模型；不能看见线性测试更好后倒改选择。

- 线性三个种子测试均为34/35正确，准确率样本标准差0；CE为0.078373/0.093625/0.084173。
- MLP种子42/7/123分别34/35、34/35、33/35正确；准确率均值96.19%，样本标准差1.65个百分点；CE为0.115197/0.092175/0.126072。
- 每个种子都把行68的class1判为class2。线性给错误类别的概率约0.806–0.877，MLP约0.942–0.956，说明更高自信也可能对应更严重的错误损失。
- MLP种子123另外把行134的class2判为class1，错误类别概率0.662。其余五个模型对该行分类正确。
- 线性以及MLP种子42/7的测试混淆矩阵为 `[[12,0,0],[0,13,1],[0,0,9]]`；MLP种子123为 `[[12,0,0],[0,13,1],[0,1,8]]`。

这是误差定位，不是样本应被改标签或剔除的证据。没有用错误样本来重选特征或重跑参数搜索。测试集只有35条，一条样本就改变2.86个百分点。三种子仅反映初始化与打乱差异，不是划分/人群不确定度，不做显著性或MLP普遍优劣结论。未来需要新的预声明评估设计，不能在这份已看过的测试集反复调参再称其“未见”。

## 5. 复现与自测

```bash
python -m pytorch_lab.wine_cli --output results/local-wine.json
# 快速机制检查，不是120轮正式结果
python -m pytorch_lab.wine_cli --epochs 1 --output results/local-wine-smoke.json
pytest -q tests/test_wine.py tests/test_wine_cli.py
```

输出采用已有无覆盖原子保存工具，目标存在或悬空符号链接时拒绝覆盖。没有网络下载或外部密钥。CPU单线程训练后恢复调用者原有线程数。使用局部Generator，不推进全局torch随机状态。

自测：独立说明为什么拟合全体数据的均值是泄漏；不用实现重写混淆矩阵与macro-F1；解释为何验证100%准确率仍有CE差异；解释为何这里应保留验证选择而不能用测试改判。工程产物已运行不代表学习者已独立掌握。

## 6. 源码与方法参考

- scikit-learn：维护者为scikit-learn开发者与社区，提供数据/预处理/模型选择/评估接口。当前瓶颈是实验边界和数据治理，故研究其 `datasets/_base.py` 的数据约定及官方评估指导，而不是为了热门再引入大型框架。
- 固定源码：[CSV读取辅助函数与load_wine](https://github.com/scikit-learn/scikit-learn/blob/e316dbeeebfd8f38cf293d6443ce81aaa33686d3/sklearn/datasets/_base.py#L325)，主要结构为 `sklearn/datasets`、`preprocessing`、`model_selection`、`metrics`。借鉴接口职责和来源记录，自己实现轻量PyTorch协议，不复制其实现。
- [官方泄漏指导](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)、[交叉验证说明](https://scikit-learn.org/stable/modules/cross_validation.html)：划分先于拟合、测试不参与选择。
- [PyTorch CrossEntropyLoss文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)；参考阅读[固定v2.5.1 loss.py](https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/torch/nn/modules/loss.py#L1144-L1299)。该源码版本用于契约阅读，实际运行版本是2.14.1+cpu，二者没有混写。
