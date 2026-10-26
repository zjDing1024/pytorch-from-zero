# 官方源码研究与许可

日期：2026-10-13，Asia/Kuala_Lumpur。项目：[PyTorch](https://github.com/pytorch/pytorch)，维护者为 PyTorch contributors / Foundation 生态。它解决张量、多后端执行与自动微分；本次选择它是为了理解现有 Wine 实验的记录边界，不因热门而换框架或建空仓库。

实际安装：2.14.1+cpu，revision `5c4886908584029761b579af026dcfb627c84070`。结构上，Python `torch` 提供 API/优化器，`aten/native` 实现算子，`tools/autograd` 描述求导规则。已读取固定版本的以下官方文件：

1. [Linear.cpp](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/Linear.cpp)：二维输入且有 bias 时，常规 dense linear 进入 addmm 路径；不能把独立写成 matmul 再加 bias 的两步算法当作逐位等价追踪。本探针保持原来的 F.linear。
2. [derivatives.yaml](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/tools/autograd/derivatives.yaml)：查阅 addmm 与 relu 条目，理解参数梯度的矩阵乘法结构、bias 的广播归约，以及 ReLU 梯度门控。记录梯度张量不等于追踪了底层实际执行微内核。
3. [torch/optim/sgd.py](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/optim/sgd.py)：查阅公开 SGD 参数、实现选择与 single-tensor 第一更新分支。空状态首步 momentum buffer 取加入 weight decay 的梯度；并不是先乘 momentum。显式 foreach=False、fused=False 固定路径选择，保留 lr=0.05、momentum=0.9、weight_decay=0.0001。独立审查另外验证原来默认 CPU SGD 的首步逐位等价，不能由这个结果推广所有配置。
4. [根 LICENSE](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/LICENSE)：BSD-3-Clause-style 通知、免责声明和非背书条件；依赖各自许可仍适用。未复制上游实现、未重新分发 PyTorch 源码，代码/测试原创。

官方文档核对：[数值准确性](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)、[SGD](https://docs.pytorch.org/docs/2.14/generated/torch.optim.SGD.html)、[retain_grad](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.retain_grad.html)。retain_grad 使非叶子张量梯度可被保存；不能因此假设保存了反向所需的全部前向内部状态。浮点算术次序和平台可以改变末位，8 ULP 只是本实验预声明的调查预算，不是 PyTorch 或数学误差保证。

本次借鉴的是观测边界与可验证合同：冻结上游输入，分别保留严格字节、连续量比较、捕获控制、有效干预。没有调用 PyTorch 私有比较函数。原创 ULP 排序公式由 binary64 符号/幅度构造，数值上合并 ±0，严格字节仍区分；已覆盖 8/9 步、符号零、subnormal 边界。

对应当前阶段：在 CPU 基础工程中练习可复现调查、自动微分和优化器语义。仍缺用户独立解释/改写记录、真实 GPU/AMP/DDP 实验；不把自动化产物数量当作岗位熟练度。
