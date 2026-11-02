# 特征缩放增量验证记录

日期：2026-11-02（UTC+08:00）。工程基线
`16c2a0e92a5a0b65fcec2acd55fd31f06c9b3d92` 的 325 个 Git blob 在修改前逐一匹配。
现有 learning-rate 主实验、stress 失败原始记录、历史阈值和依赖锁保持原样。

## 科学证据

运行前冻结 [计划](../docs/feature-conditioning-plan.json)，9 格各 12 步共 108 步，
CPU float64、PyTorch 2.14.1+cpu；未改变 1e-12 绝对／相对 oracle 容差。
[原始 JSON](2026-11-02-feature-conditioning.json)与 CLI stdout 字节一致，SHA-256：
`802e5b3c5f1234af55bcfb6804f90a33bb6e93cd2f02a37b199128d781b18779`。
9 项检查均通过，包括保留 scale4/shared 发散的预期结果。
最终损失与机制解释见[实验文档](../docs/feature-conditioning.md)。

初次核心聚焦测试 39 passed；CLI 接线聚焦 133 passed，后新增证据保留测试后
CLI 文件 13 passed；这些相交集合不得简单相加。
[独立审查](2026-11-02-feature-conditioning-independent-review.md)33 passed，
包含 216 步独立闭式复核、3 次 autograd Hessian、非二进制尺度 1.3 对照、
no-op 和非有限优化器干预。独立测试包含在最终项目总数中，不重复计数。
本次聚焦运行未出现失败或修改容差；既有缺失可选 NumPy 警告不影响 tensor/scalar 测试。

## 回归与交付

最终 source 主套件 **1668 passed，393.46 秒**；三个独立交付套件分别
59 passed（5.29 秒）、45 passed（5.70 秒）、34 unittest OK（1.209 秒）。
合计 **1806 项通过**，不重复计算聚焦与独立复跑。Ruff lint、155 文件格式、
compileall 和 pip check 均通过；依旧仅有可选 NumPy 缺失警告。
全新 venv 从已有官方索引 wheelhouse **离线 hash-lock 安装 10 个运行依赖 wheel**，
安装非 editable 项目 wheel；从无关临时目录以 `python -I` 实际完成 **17 条 CLI，全部 exit 0、科学检查通过**。
[wheel 摘要](2026-11-02-conditioning-installed-wheel-summary.json)记录 wheel SHA-256、
安装证据、全部命令结果、源文件／wheel／已安装包字节一致性及锁文件散列。
完整检查 JSON 保存在本地审计材料中，公开摘要保留其 SHA-256，避免发布本地路径等运行细节。
远程发布和远程 CI 由后续精确 commit 验证，本记录不预先宣称成功。
本地未执行 Docker、GPU/AMP/DDP 或 wheel 环境内全套 pytest。
既有 16 条安装 CLI tuple 保持一致，仅追加第 17 条 feature conditioning。
助理实现与自动化验证是项目工程证据，不证明用户独立掌握。
