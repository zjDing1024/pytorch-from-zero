# 2026-10-28 累积窗口恢复验证

日期按 UTC+08:00；本次自动运行由 AI 助理实现、执行和复核，个人掌握情况未评估。
基础工程提交：`db976faca4dbcc399923cd0d6fe137d39bef2cbc`。

## 实际结果

- 主套件 **966 passed**，244.20 s（此前 920 + 本次 46）。
- 既有独立套件 **59 passed**（4.62 s）、**45 passed**（4.58 s）、
  **34 unittest tests OK**（0.890 s）；合计 **1,104 项**，不把下列矩阵重复计入单元测试。
- 新 CLI：3 seeds × 7 cuts，**21/21 新进程恢复**的每个剩余安全点完整状态 digest 精确一致。
  原始 [JSON](2026-10-28-accumulation-recovery.json)保留全部 cases、cursor/pending counts、
  final hashes 与负对照。第二次完整 JSON 与第一次相等。
- malformed controls：缺失梯度、元素数不一致、cursor 不一致，**3/3 拒绝**。
  合法形状错误归零 pending gradient 后下一更新参数最大绝对漂移
  **0.005892547857277126**；归零 momentum 后为 **0.0014589620113064899**。
  这是预期的错误对照，不改阈值、不把结构校验当真实性认证。
- [独立审查结果](2026-10-28-accumulation-recovery-independent-review.json)：
  **36 配置 / 412 安全点**逐点精确重放，另与原有无状态累积实现对照；
  **25 malformed snapshots** 及 **4 文件/预期配置错误**拒绝。
  同时验证快照无别名、RNG 不被污染、异常失效、不覆盖保存、独立进程短窗口恢复。
- 干净非 editable wheel 的 **10/10 CLI** 均 exit 0 / all_checks_passed；独立环境、
  `-I` 和源码树外临时 cwd，pip check 通过。
  [已安装检查摘要](2026-10-28-installed-wheel-summary.json)列出全部 10 条 CLI 的退出码与检查结果、
  非 editable/隔离上下文、wheel hash 及完整原始结果 SHA256。原始 839,715 字节 JSON
  已原样保存在私有运行证据，不在仓库重复发布旧科学输出；摘要明确标注不是原始科学报告。
  用安装 wheel 的解释器执行 `python scripts/check_installed.py` 可重新生成完整输出。
  wheel `pytorch_from_zero_lab-0.1.0-py3-none-any.whl` SHA256：
  `d9ff45be037adc2ec366d2c9e8e6afa9e6223666131f91c89ea51604f1f2a287`。
  本地没有把 wheel 检查冒充实际 Docker/远程 Actions；容器约束由精确提交 CI 另行验证。
- Ruff check 通过；format check **103 files already formatted**。
  workflow YAML 经解析检查 source budget 15 min、container budget 20 min、两项新增步骤存在。
- Python 3.12、Torch **2.14.1+cpu**；可选 NumPy 未安装的初始化警告保留为环境限制，
  不影响这些纯 Torch CPU 测试；不发布含本地路径的原始环境日志。

## 复现入口

```bash
python -m pytest -q
(cd experiments/wine-first-step-2026-10-13 && python -m pytest -q)
(cd experiments/wine-saved-forward-2026-10-14 && python -m pytest -q)
(cd tools/wine-delivery && python -m unittest discover -s tests -v)
python -I -m pytorch_lab.accumulation_recovery_cli --output results/local-recovery.json
python validation/verify_accumulation_recovery.py
ruff check --extend-exclude experiments/wine-saved-forward-2026-10-14/docs/source-before-first-capture.py .
ruff format --check --extend-exclude experiments/wine-saved-forward-2026-10-14/docs/source-before-first-capture.py .
```

新 CLI 一次计时复跑用 **68.580 s**，当时有并发本地检查，不作为性能 benchmark。
因此 source job 总预算从 10 调至 15 min，容器 job 仍为 20 min；新 worker 60 s、
installed-check 每 CLI 300 s 超时不变。旧 9 条 CLI 保持原顺序，恢复作为第 10 条，
独立审查 JSON 也由 Actions artifact 保留。旧 Wine/固定输入 fixture、科学门与诊断阈值未改。
开发时新增文件的初次 Ruff 报格式/import 问题，统一格式后复查通过；没有科学失败被删掉或阈值放宽。

## 能力边界与发布证据

验证的是原创工程实现的特定恢复合同，不是一般 DataLoader、GPU/AMP/DDP 恢复，也不是
任意指令中断或断电恢复。训练无 live RNG，固定 fixture 的局部 seed 与完整数据指纹绑定。
无下载/复制上游实现，无付费资源，无私有仓库内容或凭据。

本地测试不替代远端证据；应在 [Actions](https://github.com/zjDing1024/pytorch-from-zero/actions)
核对承载本文件的精确 commit 对应运行，日报记录该 commit/run 链接。
本人需独立解释 pending sum/实际分母、复现恢复与错误注入；尚不能标为已掌握技能。
