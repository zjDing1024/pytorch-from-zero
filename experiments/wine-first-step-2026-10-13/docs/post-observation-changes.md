# 初次观测后的工程加固披露

原计划和 fixture 一直保持原字节。首次冻结所用源代码保存为 `context/probe-initial.py`；其 SHA256 同时存在于 `docs/freeze.json` 和首次四样本的运行信息中。首次完整报告 `results/abba-initial.json` 不覆盖。

第一次观察结果后，仅作以下工程变更：

- 将冻结 fixture 文件 hash 硬编码到加载契约，拒绝“同时改 fixture 和 manifest”绕过校验
- 对完整 sample 增加有限 JSON/深度/节点检查；将 JSON parser 的 RecursionError 转为可保存的 ValueError
- 加强 runtime 精确 schema 与类型检查：bool 不能充当线程数/affinity 的整数，校验 package hashes、CPU/flags、白名单环境值及未知/缺失字段；未知环境值不能作为有效控制
- 增加相关失败路径测试，并做格式整理

没有改变模型运算、fixture/种子/批次、阶段列表、SGD 参数、运行顺序或 8-ULP 数值预算。完整模型操作代码与初版的功能相同。

加固后四次新进程报告保存为 `results/abba-reviewed.json`，每个条件的科学 hash 与初版完全一致。加固后重跑是对最终交付代码的验证，不是另找输入或挑选显著结果。首次和最终捕获合计 8 个单步 worker；另有单元测试/审查用于数学和契约核对，不是完整训练拟合。

这份披露不能把本地时间戳提升成外部注册证明。预声明是可检查的本地先后顺序：计划→输入冻结→首次模型计算。它不是盲实验，也未事前在公共注册平台登记。
