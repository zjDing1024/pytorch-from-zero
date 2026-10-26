# 2026-10-12 固定输入增量验证记录

日期：2026-10-12（Asia/Kuala_Lumpur）。本次由助理实现/运行，个人独立掌握仍待自测。当前是本地准备阶段；公开提交及其精确SHA的CI尚未执行。

## 输入、计划与主机边界

工程基线 `aa56d5e3ae5ffcee428d1f4c996a04497df7bb7e`，成长基线 `a5cc07e1931b8ad2f118c3d83018b02323d3ba4b`。开工时逐个核对两个远端tree全部blob，与本地完全一致。没有覆盖旧源码/测试/科学结果。

- 预声明计划 SHA-256：`d444bef6fbbc9a401881262cae13913038855ac1885e2a02d8fe9076ba1e4a6c`
- 固定fixture SHA-256：`ea3b9d24cb7466c19fced9c85413f53137238fb36e5d4268d20e81686d299f94`
- 当前最终探针源码 SHA-256：`481d1231f6210d334c9f6eb8b6fbb818964820550a131ee86362116538e6e5b9`
- 当前环境：Linux、AMD EPYC 9V74、Python3.12.14/glibc2.41、torch2.14.1+cpu；运行记录包含实际git revision、CPU特征、线程、capability及白名单控制

**前次Wine干预使用Intel Xeon Platinum 8573C，今日使用AMD EPYC 9V74。** 计划背景误写“on this host”，在实验后核对主机证据发现，已添加[事后更正](../docs/fixed-input-plan-erratum.json)，原计划/hash完整保留。前次观察不证明当前主机的Wine敏感性。本次四个样本内部仍为同机受控比较，输入/操作/8步标准未改动；本次负结果不能直接解释前次Wine现象。

## 实验与可复核原始材料

1. [首次四样本捕获](2026-10-12-fixed-input-dispatch.json)：A-B-B-A、14阶段全相同、捕获控制通过；后续补元数据有限性防护，故不是最终代码身份的交付证据。
2. [最终代码四样本捕获](2026-10-12-fixed-input-reviewed.json)：实际AVX512/DEFAULT/DEFAULT/AVX512；输入精确、非干预控制一致、每条件严格重复。14阶段/465个值逐位相同，严格与8步门均通过，没有首个不等的显式链边界。文件SHA-256：`07b3a46d284915c592b1d45d634dd4c6b51a01a5ddd07996c819a483b80aeec2`。
3. [source八条CLI](2026-10-12-fixed-input-source-smoke.json)：baseline、minibatch、checkpoint、scheduler、wine、runtime、numerical、fixed_input，全部退出0、文件/标准输出相同、实验检查通过；完整runtime/numerical/fixed报告保留。
4. [已安装wheel八条CLI](2026-10-12-fixed-input-installed.json)：同八条全部退出0，noneditable/distribution位置检查、外部临时目录和`-I`均真实执行。runtime3 + numerical2 + fixed2共7个保留样本的25项代码/数据/许可hash全部与最终源码一致；source smoke的7个保留样本也逐一核对。

所有正式固定输入样本的科学hash为 `b3d1cb41c109ccd68ad8ceb49bffcc16069ea6a44f858c940c1840f45e352c95`。这支持仅该fixture/本机构建下的负结果；未进行真实训练输入/初始权重的一步定位，不是融合kernel跟踪，也不是所有评估算子均不敏感的结论。原Wine协议、历史hash失败和8 ULP失败保留，不用本次成功覆盖它们。

## 工程检查

```bash
python -m pytest -q
ruff check <all 41 publishable Python files>
ruff format --check <all 41 publishable Python files>
python -m compileall -q src tests scripts
python -m pip check
python -I -m pytorch_lab.fixed_input_probe --dispatch-study --output results/local-fixed-dispatch.json
python -m pip wheel --no-index --no-deps --no-build-isolation . -w <new-wheel-directory>
python -m pip install --no-index --no-deps --force-reinstall <new-project-wheel>
python -I scripts/check_installed.py --output results/local-fixed-installed.json
```

- 完整pytest：**880 passed，225.17秒，0失败/错误/跳过，退出0**。不是把局部通过拼接；[完整stdout](2026-10-12-fixed-input-pytest.txt)与[机器可读证据](2026-10-12-fixed-input-test-evidence.json)保留。原始JUnit在本地保存；公开摘要去除runner hostname，保留其文件hash和计数。
- 新增49项聚焦通过，4.31秒。初次40 passed / 1 failed来自新测试误将输出总数写成513，实际465；只改计数，输入和数值门未动。该失败日志本地保留，机器摘要公开记录。
- 独立审查复跑新测试、原比较器和安装留存测试，共77 passed/6.57秒，不额外加到880计数。另用不依赖本比较器的stdlib代码重算四样本科学与张量hash、25项包hash、非干预控制，全通过。
- Ruff、41个拟发布Python文件格式、compileall通过；source及wheel解释器pip check通过。通用`ruff format --check .`还扫描了本地构建副本，共62文件通过；拟发布数量以41为准。
- 项目wheel在已有隔离依赖环境内无索引/不解析依赖构建，再重新安装为noneditable。wheel SHA-256：`4566d27fa000b4eb2834bac20a649323e6edacad89e853dadbbce2897ed5d004`。该hash标识本次产物，不承诺重新构建字节相同。运行依赖/锁文件/Docker基础镜像没有更新。
- 固定输入CLI真实超时测试退出1并保留失败报告；已有文件/断链symlink/非法repeats拒绝覆盖或启动。坏输入、9步失败、1步同条件漂移、无效capability干预均有负对照。

## 审查修复与边界

审查发现坏worker字段容器、缺capability和重复JSON key可能绕过部分验证或导致报告丢失。现采用精确的必需字段/类型、唯一key解码，在样本接纳前验证。随后发现JSON数 `1e999` 可解析成Infinity而不经过parse_constant，新增对整个worker对象的有限数/深度/节点扫描。回归测试证明第二个worker失败仍保留第一个完整科学输出，且最终报告可用 `allow_nan=False` 序列化；不泄露任意stderr。

1 MiB上限是收集后的report检查，不是硬内存限制；进程被外部强杀或磁盘写入失败仍不保证证据持久化。捕获all_checks/退出状态只指控制有效性，跨条件strict与8步结果另报；手拆链不作为真值。更多语义见[协议](../docs/fixed-input-diagnostics.md)。

## 公开与CI状态

当前仅完成本地实现、独立代码/科学审查、完整880测试及source/wheel八CLI。本次公开commit、exact-SHA source/container CI、artifact留存/下载核验尚未执行，不引用昨日CI当作本次通过。

工作流已增加source第八CLI；已安装检查器保留旧七条并增加fixed_input完整报告。CI artifact仍配置30天；本机无Docker/Podman，不能宣称本机实际容器运行。后续发布必须对实际工程SHA重新核验source完整测试、八条source与container CLI、artifact内容与最终包hash。成长仓库无工作流，不声称CI通过。

建议工程commit：`feat: add fixed-input dispatch diagnostics with staged evidence`。下一步是单独预声明固定真实训练输入/初始权重的一次forward/loss/gradient/update实验及个人独立自测。
