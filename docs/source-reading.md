# 阶段 1 源码阅读：PyTorch 的梯度与张量布局

## 项目与阅读边界

- 项目：**PyTorch**，官方仓库 [pytorch/pytorch](https://github.com/pytorch/pytorch)。它提供 CPU/GPU 张量计算，以及支撑神经网络训练的自动微分能力。
- 作者与维护：官方 [CITATION.cff][citation] 将软件作者记为 **PyTorch Team**；代码由项目维护者和社区共同贡献。本版本 [CODEOWNERS][owners] 在 autograd 路径列出 `@albanD`、`@soulitzer`，但该文件明确仅用于变更通知，不能据此把两人写成唯一作者或唯一维护者。
- 固定阅读版本：**v2.5.1**，commit **a8d6afb511a69687bbb2b7e88a3cf67917e1697e**。标签到提交的关系已通过 [GitHub 官方接口][tag] 核对；选择固定版本是为了让源码行号可追溯，不表示它是最新版本。
- 配套实验运行版本：**2.14.1+cpu**，与上述源码阅读版本不同。已在实验虚拟环境中实际导入 PyTorch 核实版本，并补查安装包的两个 Python 入口：`backward` 仍向引擎传递 `accumulate_grad=True`，`grad` 的相应调用传递 `False`。这只补核这一处接口行为，不代表已验证两个版本的全部实现相同；数值实验结果应以配套实验记录为准。
- 阅读日期：2026-10-08。本文是原创资料研究稿，没有复制上游实现代码。已阅读下列入口及相关片段；尚未完整研究调度、线程、CUDA、编译器或全部反向算子。

## 1. `backward` 与 `grad` 的差别在哪里？

这次追踪的局部调用链是：`Tensor.backward` → `torch.autograd.backward` → `_engine_run_backward` → C++ 的 `run_backward` 绑定与执行引擎。Python 入口负责规范化参数、检查上游梯度的形状，并把工作交给底层；不能把读懂这一层等同于读懂整个自动微分引擎。

在 [`torch/autograd/__init__.py`][autograd] 中，`backward` 调用引擎时打开 `accumulate_grad`，普通 `grad` 分支则关闭它。对应的使用区别是：

- `backward()` 默认把梯度累加到参与计算且需要梯度的叶子张量 `.grad` 中；指定 `inputs` 后，累加目标受该参数限制。
- `torch.autograd.grad()` 返回指定输入的梯度元组，不把这些梯度累加进其 `.grad`。
- 非单元素输出通常需要显式传入与输出形状匹配的上游梯度。这是在计算向量－雅可比积，而非自动返回完整雅可比矩阵。
- 省略上游梯度的便利规则适用于需要梯度的实数单元素输出；`_make_grads` 会检查元素数量和浮点类型，不能把规则无限推广到任意张量。

[`graph.py`][graph] 明确将调用交给 C++；[`python_engine.cpp`][binding] 解析参数；[`engine.cpp`][engine] 的执行入口接收这些参数。这次只确认了接口边界。进一步看到 [`AccumulateGrad::apply`][accumulation] 读取叶子张量的梯度槽并调用累加辅助逻辑，没有据此声称完整掌握梯度布局、引用计数或并发实现。

## 2. 梯度累加不是“保留计算图”

假设损失是各元素平方之和，输入为 `[1, 2, 3]`，单次梯度应为 `[2, 4, 6]`。在同一叶子输入上重新做一次前向再反向，若未清理 `.grad`，预期会得到 `[4, 8, 12]`。这来自跨次反向的累加语义。

`retain_graph` 控制反向后是否保留用于再次反向的图资源；`create_graph` 控制是否为梯度计算构建图，以支持高阶求导。二者都不是“开启梯度累加”的开关。一般训练步会重新前向；累积多个微批次也通常不需要为了累加而设置 `retain_graph=True`。这些参数的定义可在 [Python 入口与文档字符串][autograd] 中核对。

阶段 1 应先用小例子区分：重置 `.grad`、重复前向、复用同一次前向、返回梯度、保存梯度。不要遇到重复反向错误就机械加入 `retain_graph=True`。

## 3. `view`、转置与 stride

shape 描述各维长度，stride 描述沿各维移动一步在存储中跨过多少个元素。对于普通稠密张量，转置可以只交换 shape 与 stride，继续共享原数据；这能在 [`TensorShape.cpp` 的 transpose 分支][transpose] 中直接看到。

`view` 需要目标形状与现有存储布局兼容。其 [`view_impl`][view] 会尝试计算新 stride，失败时抛出错误。因此，“元素总数相同就一定能 view”是不完整的判断；“所有非连续张量都不能 view”也过于绝对。

[官方视图说明][views-doc] 还强调：视图共享底层数据；`reshape` 可能返回视图，也可能复制；`contiguous` 在输入已经连续时返回自身，否则复制成连续布局。实验应同时观察 shape、stride、连续性和共享关系，不能仅凭输出数值判断有没有复制。

## 4. 广播为什么不等于复制？

[广播说明][broadcast-doc] 的核心是从尾部对齐维度，兼容维度须相等或有一侧为 1。对于普通稠密张量的 `expand`，[`TensorShape.cpp`][expand] 先计算扩展布局，再创建共享存储的视图；[`ExpandUtils.cpp`][expand-geometry] 将真正扩展的单例维 stride 设为 0。多个逻辑位置因此能指向同一个存储位置，不需要先复制输入数据。这里并不意味着随后的加法等运算也不分配输出。

反向时，这些重复使用的位置对原输入的贡献需要相加。读取 [`tools/autograd/derivatives.yaml` 的 expand 规则][expand-grad] 可以确认，它把上游梯度求和还原到输入形状。例如长度为 3 的偏置被用于两行数据、最后整体求和时，偏置梯度的预期值是 `[2, 2, 2]`。这也是检查广播形状错误的一个直观入口。

## 为什么适合阶段 1？下一步学什么？

这条路线只需要 Python、基本张量运算和链式法则，就能把“API 行为 → 可复现实验 → 官方源码证据”连起来。它优先解决训练中常见的形状、梯度累加和内存共享误解，无需先编译完整框架。

建议用自己写的最小实验验证四件事：

1. 比较 `backward` 和 `grad` 的返回值与 `.grad` 变化。
2. 用两次独立前向演示梯度累加，再加入重置进行对照。
3. 检查一个转置张量的 stride，比较 `view` 与 `reshape` 的行为。
4. 对广播偏置手算梯度，并验证 `expand` 的零 stride。

**验证状态：本文完成的是固定版本官方文档与源码核对，以及当前安装包版本和入口标志的补核；上述数值实验是建议与数学预期，不是本文已执行的测试记录。** 实验执行后，应另行记录运行版本、命令、断言与结果。后续再研究叶子与非叶子张量、原地修改限制、`gradcheck` 和高阶梯度。

当前安装包补核记录：`torch.__version__` 为 `2.14.1+cpu`，`torch.version.git_version` 自报 `5c4886908584029761b579af026dcfb627c84070`。通过 Python 的 `inspect.getsourcelines` 读取本地 `torch/autograd/__init__.py`，`backward` 的累加标志位于 402 行，`grad` 的对应标志位于 587、601 行。这些行号仅适用于本次安装包，不与下方固定版本链接的行号混用；没有另外核验该自报提交的远程源码。

## 固定版本来源

以下链接均指向官方仓库固定提交；路径与行号对应本次实际阅读范围。

- `torch/_tensor.py`：[`Tensor.backward`，525–583 行][tensor-backward]
- `torch/autograd/__init__.py`：[`_make_grads`、`backward`、`grad`，88–519 行][autograd]
- `torch/autograd/graph.py`：[`_engine_run_backward`，816–831 行][graph]
- `torch/csrc/autograd/python_engine.cpp`：[Python/C++ 绑定入口，183–248 行][binding]
- `torch/csrc/autograd/engine.cpp`：[执行入口片段，1188–1245 行][engine]
- `torch/csrc/autograd/functions/accumulate_grad.cpp`：[梯度槽与累加调用，24–66 行][accumulation]
- `aten/src/ATen/native/TensorShape.cpp`：[expand][expand]、[transpose][transpose]、[view_impl][view]
- `aten/src/ATen/ExpandUtils.cpp`：[扩展布局计算，62–114 行][expand-geometry]
- `tools/autograd/derivatives.yaml`：[expand 的反向规则，664–666 行][expand-grad]
- `docs/source/tensor_view.rst`：[视图文档源码][views-doc]
- `docs/source/notes/broadcasting.rst`：[广播文档源码][broadcast-doc]

[citation]: https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/CITATION.cff
[owners]: https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/CODEOWNERS
[tag]: https://api.github.com/repos/pytorch/pytorch/git/ref/tags/v2.5.1
[tensor-backward]: https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/torch/_tensor.py#L525-L583
[autograd]: https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/torch/autograd/__init__.py#L88-L519
[graph]: https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/torch/autograd/graph.py#L816-L831
[binding]: https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/torch/csrc/autograd/python_engine.cpp#L183-L248
[engine]: https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/torch/csrc/autograd/engine.cpp#L1188-L1245
[accumulation]: https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/torch/csrc/autograd/functions/accumulate_grad.cpp#L24-L66
[expand]: https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/aten/src/ATen/native/TensorShape.cpp#L1138-L1153
[transpose]: https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/aten/src/ATen/native/TensorShape.cpp#L3114-L3147
[view]: https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/aten/src/ATen/native/TensorShape.cpp#L3363-L3375
[expand-geometry]: https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/aten/src/ATen/ExpandUtils.cpp#L62-L114
[expand-grad]: https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/tools/autograd/derivatives.yaml#L664-L666
[views-doc]: https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/docs/source/tensor_view.rst
[broadcast-doc]: https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/docs/source/notes/broadcasting.rst
