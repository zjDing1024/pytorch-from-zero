# 特征缩放、条件数与优化器坐标

## 预声明问题与公平预算

[运行前协议](feature-conditioning-plan.json)固定五行 `x=[-1,-.5,0,.5,1]`，
`y=2*x+.5`，CPU float64、零初始预测、每格 12 次全批量更新。
三个尺度 `z=s*x, s∈{.25,1,4}` 与三种更新规则组成 9 格、108 次更新。
不改变数据目标，不搜索更好曲线，不把缩放当作增加模型表达能力。

原坐标权重 `a=s*theta`，预测始终是 `a*x+b`。
损失精确写作 `L=.5*(a-2)^2+(b-.5)^2`，
在 `(theta,b)` 中 Hessian 是 `diag(s²,2)`。
条件数为 32、2、8；条件数是坐标相关量，不是数据固有难度。
对于标量学习率，严格稳定区间 `0<lr<2/lambda_max`。

- **shared**：三尺度都用标量 `.15`。相同学习率不代表相同函数空间更新。
- **curvature**：用标量 `.5/lambda_max`，保证各特征方向收缩，但不保证有限预算内最快。
- **mapped**：`lr_theta=.15/s², lr_bias=.15`。链式法则给出
  `grad_theta=s*grad_a`，因而 `delta_a=-.15*grad_a`；这是有意改变优化器度量的机制对照，
  不是声称普通标量 SGD 对重参数化不变。对角步长不适用单一标量稳定阈值。

## 观测与解释

[完整 JSON](../results/2026-11-02-feature-conditioning.json)保留每步坐标梯度、映射梯度、
参数、预测、损失、预测更新 RMS 和闭式 oracle。比较使用原问题的 `(a,b)` 与预测空间，
不比较跨尺度 `theta` 范数来判断更好。初始 MSE 均为 2.25。

|尺度|条件数|shared 最终 MSE|curvature 最终 MSE|mapped 最终 MSE|
|---:|---:|---:|---:|---:|
|.25|32|1.59538626|1.37051506|.04051333|
|1|2|.04051333|.00200680|.04051333|
|4|8|6428.39944873|.05311921|.04051333|

尺度 4 的 shared 权重误差乘子为 `1-.15*16=-1.4`，交替放大；所有数值仍有限，
发散是预期科学结果而不是测试失败。尺度 .25 的权重乘子接近 1，固定预算内下降慢。
三尺度 mapped 预测与原坐标轨迹在预声明 `atol=rtol=1e-12` 内一致。
curvature 各步损失下降，但尺度 4 的最终损失反而高于 mapped；安全不等于最优。

闭式 oracle 使用 `e_a(k)=-2*(1-lr_theta*s²)^k`、
`e_b(k)=-.5*(1-2*lr_bias)^k`，没有调用训练循环。独立测试另外从原始样本
计算 autograd Hessian 与闭式数值，防止实现与 oracle 同源错误。

## 运行

```bash
python -m pip install --no-deps -e .
python -m pytorch_lab.feature_conditioning_cli --output results/my-feature-conditioning.json
python -m pytest tests/test_feature_conditioning.py tests/test_feature_conditioning_cli.py tests/test_feature_conditioning_independent.py -q
```

CLI 不覆盖已存在输出，非有限数不能序列化，科学检查失败仍保留有限证据并返回非零。
source CI 与安装 wheel 检查追加第 17 条 CLI，原 16 条命令和历史 LR/stress 证据不变。
本实验不声称标准化对所有数据都更快、提高泛化、提高墙钟吞吐量或证明个人独立掌握。
它提供可面试解释的曲率、链式法则、学习率稳定性与预条件机制证据。

## 开源研究与许可

研究官方 [PyTorch SGD 固定源码](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/optim/sgd.py)：
其参数组支持各自学习率，本实验通过公开 API 实现 diagonal-rate control。
[固定版本 LICENSE](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/LICENSE)
为 BSD 风格三条款。上述推导、实验及测试为原创，没有复制上游实现。
