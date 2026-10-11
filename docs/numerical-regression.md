# 完整数值证据与预声明回归边界

2026-10-11，Asia/Kuala_Lumpur；阶段1第七增量。保留原Wine协议、历史原始JSON、原测试与严格科学hash检查，不调模型、seed、划分或测试集。

## 为什么继续这个问题

前次两个CI中source/container的科学hash映射交换：b7ab9686…与28ce527e…。每个job内部一致，不代表跨job逐位相同，也不能把类别命名为“容器误差”。此前只保留hash的路径无法定位字段，因此本次先补完整输出和运行信息，再做有限对照。

## 开跑前写下的合同

[机器可读计划](numerical-regression-plan.json)在新Wine实验前写下；这是已知历史差异后的前瞻协议，不是盲研究。它使用单独的诊断门，不修改原runtime CLI的严格hash门。

- canonical科学SHA-256仍只排除顶层created_at_utc与environment；不舍入、不截断、不把“接近”改称hash一致。
- 结构、类型、对象key集合、数组长度/顺序、数据标识、配置中的浮点数、seed、row ID、预测、混淆矩阵、选中架构、检查结果及未知路径全部精确比较。
- accuracy、precision、recall、F1、macro-F1及其汇总、类别先验来自离散计数，也必须精确一致。
- 仅明确路径上的均值/尺度、交叉熵/训练损失/交叉熵汇总、概率矩阵及错误概率允许最多8个相邻有限binary64可表示数的距离。它是刻意严格的工程诊断预算，不是累积训练误差的理论界。没有非零absolute floor，也没有套用通用assert_close默认值。
- 用符号化整数序计算ULP步数；不能用绝对误差除以单侧ulp近似。跨符号、零、subnormal、2的幂边界都需要检查。允许路径上的正负零数值相等，严格hash仍不同；其余浮点路径按位精确。
- NaN/Inf、重复JSON key、非CPU/float64元数据、错误schema或失败协议检查拒绝输入。输入大小、深度、节点数及诊断示例数有界。该验证检查报告结构和自报状态，不重新证明每个指标的真值。
- 8步失败保留失败；不按本次观测调成75或更大的阈值。小绝对差不意味着小ULP距离，特别是很小的概率；诊断失败也不等于预测质量退化。

## 使用方法

```bash
# 原严格runtime命令继续可用
python -I -m pytorch_lab.runtime_probe --repeats 3 --output results/local-runtime.json
# 同一条件完整留存，每个样本包括未删减Wine JSON
python -I -m pytorch_lab.numerical_probe --repeats 2 --output results/local-numerical.json
# 固定A-B-B-A：A unset，B ATEN_CPU_CAPABILITY=default
python -I -m pytorch_lab.numerical_probe --dispatch-study --output results/local-dispatch.json
# 比较两个完整Wine JSON；有失败仍写报告，退出1
python -I -m pytorch_lab.scientific_compare left.json right.json --output results/local-compare.json
# 若必须严格科学hash一致，明确要求精确门
python -I -m pytorch_lab.scientific_compare left.json right.json --require-exact --output results/local-exact.json
```

现有文件和断链符号链接都拒绝覆盖。捕获CLI的all_checks_passed只表示完整留存、非干预控制一致、每条件严格重复及干预确实改变已报告dispatch；跨条件的exact/approximate结果独立报告，不借捕获成功掩盖数值失败。若子进程超时/返回错误，则退出2并保存已完成样本和脱敏错误类别，报告明确失败；安装检查器也保留先前完成报告及失败子命令已写出的结构化证据。参数本身无效时不启动工作。已存在的原始产物不变。

## CPU运行信息与隐私边界

每个新样本保留完整Wine结果及其canonical hash、原计量/包文件hash，以及CPU型号/特征集合、实际affinity、kernel、binary64属性、torch git revision、ATen已报告capability、MKL/oneDNN/OpenMP启用状态和可提取的版本、deterministic算法设置、有限名单的线程/dispatch环境变量。

不dump整个环境、hostname、机器ID、序列号、用户名、原始进程maps或编译路径。torch构建配置的路径不复制，保存去除动态capability行的SHA-256及版本字段；未知环境变量值只标记redacted。版本/配置hash不等于每个原生库的实际加载映像hash，仍不能完整证明所有动态链接状态相同。

## 受控单因素的范围

同一主机、解释器、包/数据、seed和线程设置，四次新进程按A-B-B-A顺序，唯一有意修改为ATEN_CPU_CAPABILITY。该环境变量在导入/首次dispatch前设置；记录请求值和get_cpu_capability实际观测。固定120轮、每样本6次拟合，总计5,040更新/77,040样本访问。

ATen DispatchStub源码把capability缓存，并按各kernel已编译实现及fallback选择。该干预不意味着所有指令都变成scalar，也不控制全部MKL/oneDNN。若两个条件capability相同，干预无效，应失败/说明，不能据此说dispatch不影响结果。A-B-B-A仅减轻简单线性时间趋势；共享宿主和四样本不足以作性能显著性结论。更不证明历史CI主机的差异原因。

## 留存、CI与验证层次

原六条已安装wheel CLI保留，新增第七条完整数值捕获。source CI上传local-ci JSON，container CI保存并上传已安装检查器的完整JSON（其中包含原runtime与numerical报告）。Artifact保存30天，不保证永久可用；本次公开results保留正式实验与历史恢复证据供长期复查。所有Artifact上传使用always条件以尽量保留失败现场，作业失败不会被改为成功。

本地实际结果和检查命令见[验证记录](../results/2026-10-11-numerical-verification.md)。Docker与远程CI是否完成必须以本次精确提交记录为准，不能沿用昨日绿灯。

## 源码研究与来源

项目：PyTorch，PyTorch contributors / Foundation生态；核心解决张量计算、自动微分和多后端执行。研究固定安装git revision [5c4886908584029761b579af026dcfb627c84070](https://github.com/pytorch/pytorch/tree/5c4886908584029761b579af026dcfb627c84070)。

- [aten/src/ATen/native/DispatchStub.cpp](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/DispatchStub.cpp)：环境请求、硬件检测、静态缓存及kernel fallback；帮助设计新进程干预与观测有效性检查。
- [torch/testing/_comparison.py](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/testing/_comparison.py)：输入树遍历、typed Pair、容差和带路径诊断的分离；借鉴职责拆分，没有复制实现，也不依赖私有API。
- [LICENSE](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/LICENSE)：BSD-3-Clause-style条件、通知/免责声明与非背书限制；依赖组件另有许可。本次全部新增实现为原创。

官方依据：[数值准确性](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)、[随机性/复现](https://docs.pytorch.org/docs/2.14/notes/randomness.html)、[CPU capability](https://docs.pytorch.org/docs/2.14/backends.html#torch.backends.cpu.get_cpu_capability)、[构建配置](https://docs.pytorch.org/docs/2.14/config_mod.html)、[线程环境变量](https://docs.pytorch.org/docs/2.14/threading_environment_variables.html)、[Python nextafter/ulp](https://docs.python.org/3.12/library/math.html#math.nextafter)、[PEP485近零讨论](https://peps.python.org/pep-0485/#behavior-near-zero)。这些来源解释数值语义与限制，不为本项目8步阈值提供误差保证。
