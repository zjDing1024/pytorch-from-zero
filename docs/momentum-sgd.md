# 从公式写出 Momentum SGD，并逐步对照 PyTorch

## 目标、预期与失败条件

这次增量把优化器的状态转换从训练循环中单独取出。目标不是复刻整个 `Optimizer`，而是独立实现一个能手算、能检查状态、能明确失败的更新函数。代码由 AI 助理原创实现并自动验证；学习者自测仍待完成，不代表用户已经掌握优化器源码。

预期：在相同初值、梯度、超参数和学习率序列下，手写更新与当前安装的 `torch.optim.SGD` 在**每一步、每个参数及其 momentum buffer**上的最大绝对误差都不超过 `1e-12`；buffer 是否存在也必须一致。只比较最终损失不足以检验这些语义。

失败条件包括参数误差超标、buffer 误差超标或存在性不同，以及输入非法、形状隐式广播、数值溢出、意外修改调用者张量或全局随机状态。运行失败不能被解释为“训练大致正常”。

## 1. 状态转换与原始梯度

记参数为 θ，当前损失梯度为 g，学习率为 α，动量系数为 μ，dampening 为 τ，耦合 weight decay 为 λ。以下运算逐元素进行，同一组标量设置用于所有参数。

先得到下降形式的方向：

```text
q = g + λθ             # minimize
q = -g + λθ            # maximize
```

最大化时只翻转损失梯度，衰减项仍把参数拉向零。这里实现的是 SGD 的**耦合**衰减：λθ 进入 q，也进入后续动量。不能把它当成 AdamW 一类的解耦衰减。

当 μ > 0 时，每个参数单独管理 buffer：

```text
尚无 buffer： b_new = q
已有 buffer： b_new = μb_old + (1 - τ)q
普通动量：    d = b_new
Nesterov：    d = q + μb_new
最终更新：    θ_new = θ - αd
```

当 μ = 0 时，d = q，不建立 buffer；如果调用者带入了旧 buffer，返回其不变副本。Nesterov 只接受 μ > 0 且 τ = 0。

这些语义按 [PyTorch 2.14 SGD 文档][sgd-doc]及安装包的 Python 单张量路径核对。手写函数只使用张量算术，不调用官方优化器；官方优化器只在独立的参考实验和测试中充当对照。

## 2. 三个容易写错的边界

### 第一次更新以“该参数第一次有梯度”为准

某个 bias 在全局第一步没有梯度，并不表示它第二步应进入“已有 buffer”分支。没有 buffer 时直接用 q 初始化，τ 不缩小该次贡献。例如 μ=0.8、τ=0.5，前两次有效 q 为 3、4，则 buffer 为 3、4.4，而不是 1.5、3.2。

### `None` 与全零张量不同

`None` 表示跳过这个参数，θ 和 buffer 都不更新，也不施加衰减。全零张量仍是一项有效梯度：旧动量和 λθ 都可能使参数继续移动。前者还可能推迟 buffer 的第一次创建。该区别在官方 `SGD._init_group` 收集参数时就已产生，并非更新公式的特判。

### buffer 不包含学习率

α 只用于最后的参数更新。改变 α 不应额外缩放或重置 buffer；α=0 时，若梯度存在，buffer 仍能继续演化。手算例子：θ=2、μ=0.8，第一步 g=3、α=0.1 得 θ=1.7、b=3；第二步 g=1、α=0 得 θ=1.7、b=3.4；第三步 g=0、α=0.01 得 θ=1.6728、b=2.72。这里不讨论把学习率存入“速度”的另一套定义。

## 3. 狭窄而明确的 API 契约

`src/pytorch_lab/momentum.py` 提供：

```python
new_parameters, new_buffers = momentum_sgd_step(
    parameters,
    gradients,
    buffers,
    learning_rate=0.05,
    momentum=0.8,
    dampening=0.0,
    weight_decay=0.0,
    nesterov=False,
    maximize=False,
)
```

- 三个输入集合都是等长且非空的 list/tuple，位置一一对应。梯度和 buffer 可以是 `None`。
- 每个参数、有效梯度、有效 buffer 必须是非空、有限、CPU `float64` 稠密 strided 张量。梯度与 buffer 必须和对应参数完整 shape 相等，禁止广播。
- 支持标量和转置等非连续输入。不同参数不得共享底层 storage；即使是同一 storage 的不相交切片，也属于此实现明确排除的情况。返回值只保留数值，不保留调用者的别名关系。
- α、μ、λ 为有限非负 Python int/float；τ 限于 `[0, 1]`。bool 不冒充数值，Tensor 超参数不接受。`nesterov`、`maximize` 必须是真正的 bool。
- 函数返回两个新列表，其张量不与输入共享存储，不保留 Autograd 图。它不读取或写入 `.grad`，也不更新模型对象；调用者须显式使用返回值。
- 已跳过的参数和 buffer 也做输入检查。中间方向、buffer、更新后的参数若出现 NaN/Inf，抛出 `ValueError`；调用者输入保持不变。
- 不提供参数组、closure、可微优化器步、CUDA、低精度、complex、sparse、foreach、fused 或 AMP。这里的严格输入检查不是 PyTorch 全部合法输入范围的声明。

函数式接口有意让状态可见：忘记接回 `new_buffers` 会丢失动量，忘记接回 `new_parameters` 会丢失更新。它适合小规模教学和数值对照，不声称性能或可替代生产优化器。

## 4. 参考实验和测试覆盖

`run_momentum_reference_trace()` 不需要参数，返回可以直接序列化的字典。它固定两个参数、六步梯度及学习率 `[0.2, 0.2, 0.05, 0, 0.025, 0.025]`，逐步运行六组条件：普通 SGD、普通动量、dampening+decay、Nesterov+decay、maximize+decay、maximize+Nesterov+decay。

梯度序列包含延迟首次更新、两个参数分别缺席，以及显式零梯度。每条参数记录有参数最大绝对误差、buffer 最大绝对误差，以及双方 buffer 存在性。没有 buffer 时其误差记为 `null`，不是把“未建立状态”伪装成一个全零 buffer。

参考端明确设置 `foreach=False, fused=False`，沿普通非可微单张量路径执行。最大误差只检验这条已执行路径，不扩展到所有后端。手写公式的乘加分解与参考端的 `add(alpha=...)` 运算次序可能引起末位舍入，因此使用绝对容差而不承诺逐位一致。

`tests/test_momentum.py` 另外包含手算结果、独立从各自当前参数求出的二次损失梯度连续十二步对照、标量/非连续输入、dampening=1、暂停/恢复动量、输入不变与图隔离、形状和 dtype/device/layout 拒绝、四条溢出路径，以及严格 JSON 与 RNG 检查。

在安装项目依赖的环境中，从项目根目录执行：

```bash
python -m pytest -q tests/test_momentum.py
ruff check src/pytorch_lab/momentum.py tests/test_momentum.py
ruff format --check src/pytorch_lab/momentum.py tests/test_momentum.py
PYTHONPATH=src python -c 'import json; from pytorch_lab.momentum import run_momentum_reference_trace; print(json.dumps(run_momentum_reference_trace(), indent=2, allow_nan=False))'
```

2026-10-08 本地实测：上述参考轨迹参数最大差 `1.1102230246251565e-16`，buffer 最大差 `2.220446049250313e-16`，所有逐步检查通过。此处是运行摘要；集成实验保存完整逐步 JSON。自动测试通过仍不等于学习者已能脱离答案解释或重写这些分支。

## 5. 阅读来源、版本与边界

- 官方文档：[PyTorch 2.14 `torch.optim.SGD`][sgd-doc]，阅读算法、参数约束与动量说明。
- 固定源码入口：[v2.14.1 `torch/optim/sgd.py`][sgd-source]。
- 实际检查的是本地安装包 `2.14.1+cpu`：`SGD.__init__`（29–73 行）、`SGD._init_group`（84–103 行）、`_single_tensor_sgd`（322–379 行）。只阅读与本实验有关的 Python 边界，没有完整研究 foreach/fused、CUDA 内核、分布式或可微优化器。
- 本地 `torch.version.git_version` 自报 `5c4886908584029761b579af026dcfb627c84070`；没有把它当作另行核验过的远程标签映射。
- 本地 `torch/optim/sgd.py` SHA-256：`e6c647249cee41c99cd29da4cb3fe74c94d899e775c9f952308fcafdcae43725`。
- 此处代码、手算案例、测试和文字均为本增量原创；没有复制上游优化器实现，也没有改变现有手写回归基线。

[sgd-doc]: https://docs.pytorch.org/docs/2.14/generated/torch.optim.SGD.html
[sgd-source]: https://github.com/pytorch/pytorch/blob/v2.14.1/torch/optim/sgd.py
