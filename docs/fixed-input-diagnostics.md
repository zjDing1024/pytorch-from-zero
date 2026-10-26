# 固定 logits 的逐算子诊断

2026-10-12，Asia/Kuala_Lumpur；阶段1第八增量。问题来自前次完整 Wine 实验的末位差异，本次只调查一份固定合成输入，**零训练更新，不改变 Wine、历史严格 hash 或原8 ULP合同**。

## 预声明问题与真实负结果

[开跑前计划](fixed-input-plan.json)的 SHA-256 为 `d444bef6fbbc9a401881262cae13913038855ac1885e2a02d8fe9076ba1e4a6c`。已知前次 Wine 全流程在当时主机上对 ATen capability 干预敏感；本次是后续定位，不是盲研究。

**主机边界与事后更正：本次为AMD EPYC 9V74，前次Wine干预记录为Intel Xeon Platinum 8573C。** 原计划背景句误写“on this host”，运行后检查硬件证据发现错误，已新增[更正记录](fixed-input-plan-erratum.json)，原计划/时间/hash原样保留。前次结果没有证明今日主机的完整Wine敏感性；今日四样本内部仍保持同一主机。这个硬件差别进一步限制本次负结果对前次现象的解释。

假设：即使完全固定 logits，评估算子也可能随 capability 改变。使用手写 dyadic 数（都可精确表示为 binary64）的16×3 logits、16个类别索引；含相等 logits、正负零、小间距、共同平移及较大类别差距。fixture在任何新计算之前写定，SHA-256为 `ea3b9d24cb7466c19fced9c85413f53137238fb36e5d4268d20e81686d299f94`。没有从实验输出中筛选输入。

[最终冻结代码四样本结果](../results/2026-10-12-fixed-input-reviewed.json)：A-B-B-A的实际capability为AVX512 / DEFAULT / DEFAULT / AVX512；14个输出记录、465个值（449个float、16个预测）全部逐位相同。输入/克隆、相等logits控制、非干预运行信息、包文件hash及同条件严格重复均通过。四个科学SHA-256均为：

`b3d1cb41c109ccd68ad8ceb49bffcc16069ea6a44f858c940c1840f45e352c95`

因此，本fixture没有观察到首个不相等边界；该假设在这些输入/算子/本机构建下没有获得支持。不能推广成“评估算子都不受dispatch影响”，也没有定位Wine训练参数或历史CI的差异。**不因结果全相同就替换fixture寻找显著差异。** 前一次[审查修补前捕获](../results/2026-10-12-fixed-input-dispatch.json)也完整保留，科学结果相同；其代码hash与最终版相差JSON元数据有限性防护，不作为最终代码交付证明。

## 运算与依赖边界

输入为CPU float64、连续布局、stride(3,1)，标签为CPU int64。文件原字节和张量按固定大端编码的binary64/int64字节分别校验；不使用pickle或NumPy，不重新运行任何模型生成输入。

| 记录 | 操作/输入 | 形状 |
|---|---|---|
| identity | clone原始logits；保留正负零编码 | 16×3 |
| row_max | 每行最大值 | 16×1 |
| shifted | logits − row_max | 16×3 |
| exp | exp(shifted) | 16×3 |
| denominator | 对exp按行求和 | 16×1 |
| log_denominator | log(denominator) | 16×1 |
| decomposed_softmax | exp / denominator | 16×3 |
| decomposed_log_softmax | shifted − log_denominator | 16×3 |
| softmax / log_softmax | 分别从固定原logits调用torch API | 各16×3 |
| cross_entropy_none | 固定原logits及标签，reduction=none | 16 |
| cross_entropy_mean | 新clone logits及标签，默认mean | scalar |
| logits_gradient | mean CE对该clone logits的autograd梯度 | 16×3 |
| predictions | 固定原logits的argmax；类别平局取首项 | 16 |

数学参考：若 `p = softmax(z)`，无类别权重/label smoothing时，mean CE为 `−sum(log_softmax(z)[i,y_i])/N`，其梯度为 `(p−one_hot(y))/N`。测试用这条解析关系做独立合理性检查，但该近似检查不取代跨条件的固定8 ULP门，也不保证精确实数真值。

`first_unequal_explicit_chain_boundary`只按identity→row_max→shifted→exp→denominator→decomposed_softmax顺序报告最早不等记录。log分支、torch融合API和反向独立记录；不能把序列中相邻的独立API误当作相互依赖。中间量也可能包含已传播的上游差异。

手拆链不是融合softmax的执行跟踪，更不是ground truth。所读上游内核可能通过“先取倒数，再乘exp”归一化，本实现显式除法，两者有不同舍入位置；向量exp/log、行分块和运算顺序也可能不同。本次没有把两种算法互相作为逐位验收标准。

## 三层判定

1. **输入/捕获控制**：原fixture字节、输入张量字节、固定schema/顺序/类型/shape，克隆和相等logits控制；请求值与实际capability；线程、运行元数据、包文件hash；同条件输出严格重复。失败仍保留已完成样本。
2. **严格科学输出**：完整科学对象的canonical SHA-256及各张量字节hash，无舍入/截断；正负零不同。时间、运行环境、包hash在外层另存，不混入科学对象。
3. **独立数值诊断**：新协议 `fixed-logits-dispatch-v1` 对13个float阶段固定使用已有ULP实现和8步预算、无非零absolute floor；预测、结构、输入必须精确。8步不是数学误差界。原Wine比较器及其75/144 ULP失败记录未改动。

捕获CLI的 `all_checks_passed` / 退出0只表示第一层成立，第二/三层通过情况在 `comparisons_to_first_sample` 各自报告。测试确认即使跨条件为9 ULP失败，完整有效的捕获仍能保存该失败；同条件即便只差1 ULP，也不能通过严格重复控制。NaN/Inf、溢出JSON数、重复key、变更输入、伪造hash、stage重排或坏元数据在加入成功样本前拒绝。

## 复现与资源边界

```bash
# 两次同条件新进程，作为source/wheel/CI的第八条CLI
python -I -m pytorch_lab.fixed_input_probe --output results/local-fixed-input.json
# 预声明A-B-B-A；每个child最多60秒（可设置为(0,300]秒）
python -I -m pytorch_lab.fixed_input_probe --dispatch-study --output results/local-fixed-dispatch.json
```

A显式清除ATEN_CPU_CAPABILITY；B设置default；三种线程环境变量及torch intra/inter-op线程均为1，设置在运算前。无拟合、优化器更新、模型选参、数据下载或新增依赖。Linux CPU-only构建；同条件模式2–4次，干预模式固定4次。capability没有实际改变时，干预控制失败，不会伪称有效对照。

报告拒绝覆盖现有文件及断链symlink。参数/输出路径无效退出2；worker超时、异常或无效报告导致捕获失败，保留完整先前样本及脱敏类别并退出1。1 MiB限制在子进程输出被收集后检查，是报告验证上限，不是硬内存沙箱。另有有限数/depth/node扫描；不记录任意stderr、全量环境、hostname或secret。沿用既有白名单CPU信息。

原七条CLI仍运行；已安装wheel检查器增加第八条及完整报告留存。CI source和container artifact仍保存30天；不能把配置了CI当成已经运行。当前验证/公开状态见[验证记录](../results/2026-10-12-fixed-input-verification.md)。

## 官方源码研究与许可

项目：[PyTorch](https://github.com/pytorch/pytorch)，维护者为PyTorch contributors / Foundation生态。解决张量、自动微分与多后端执行；与当前CPU数值定位问题直接相关，不为追逐热门框架开新仓。

固定实际安装revision `5c4886908584029761b579af026dcfb627c84070`，2026-10-12阅读：

- [SoftMaxKernel.cpp](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/SoftMaxKernel.cpp)：max、vector exp、sum、倒数归一化及dispatch注册，说明为何要冻结输入并拆分观测。
- [LogSoftmaxKernelImpl.h](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cpu/LogSoftmaxKernelImpl.h)：exp归约、log与减法顺序，说明把 `log(softmax(x))` 当替代实现可能改变数值。
- [LossNLL.cpp](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/LossNLL.cpp)：无平滑class-index分支走log_softmax与NLL；本实验标签形式与该分支对应。
- [vec512_double.h](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/cpu/vec/vec512/vec512_double.h)、[vec256_double.h](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/cpu/vec/vec256/vec256_double.h)、[vec_base.h](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/cpu/vec/vec_base.h)：向量exp/log可调用SLEEF不同宽度实现，generic路径可映射标准数学函数。这是源码可能路径，未追踪本机实际执行微内核；函数名不构成本项目误差保证。
- [根LICENSE](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/LICENSE)：BSD-3-Clause-style通知/免责声明/非背书条件；依赖许可各自适用。代码、fixture和测试原创，没有复制上游实现。本仓库尚未选择开源许可证。

官方2.14文档核对：[数值准确性](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)、[cross_entropy](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.cross_entropy.html)、[log_softmax](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.log_softmax.html)。借鉴的是算子边界、依赖图和证据设计；没有把文档当作跨硬件逐位保证。

## 下一步与个人自测

先独立解释本次负结果能排除什么、不能排除什么，再另写前瞻协议：冻结真实训练输入及初始权重，比较一次forward/loss/gradient/update边界。不得将新fixture混进本次结果，也不宣称已经完成训练路径定位。持续保留历史失败和不确定性。当前所有实现/运行由助理完成，个人独立掌握仍待自测。
