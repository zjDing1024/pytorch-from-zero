# CPU运行时、离线wheel与资源计量

2026-10-10；阶段1的第六增量。目标是把“当前目录能跑”推进为“明确安装产物、依赖字节和测量范围后仍能跑”。没有重新训练新候选、调参或重复实现Wine评估。

## 三种不同的复现

1. **安装重复性**：对CPython 3.12 / Linux x86_64使用相同10个运行依赖wheel及SHA-256。完整闭包包括torch、filelock、typing_extensions、setuptools、sympy、networkx、jinja2、fsspec、mpmath和markupsafe。
2. **数值重复性**：相同数据、代码、seed与线程设置下，科学输出一致。本次六次正式新进程测量及前一日Wine JSON除时间戳/环境字段外完全一致。
3. **构建字节重复性**：不据此宣称成立。自产wheel记录哈希，但ZIP时间戳、构建工具链及基础系统仍可能影响重建字节。pip/bootstrap工具不在运行依赖锁内；pip实际版本单独记录。

锁文件只覆盖本目标平台。torch与MarkupSafe含CPython/ABI/架构标签，不能把它宣传为Windows/macOS/ARM或任意Python版本的通用锁。CPU机器无需GPU依赖；旧requirements.txt保留便捷开发路径，严格复现使用本页的新锁。

## 获取与安装分离

从仓库根目录运行，先创建不带system-site-packages的环境：

```bash
python3.12 -m venv .venv-runtime
. .venv-runtime/bin/activate
mkdir -p wheelhouse
python -m pip download --only-binary=:all: --no-deps --require-hashes \
  --index-url https://pypi.org/simple \
  -r requirements/runtime-common.lock -d wheelhouse
python -m pip download --only-binary=:all: --no-deps --require-hashes \
  --index-url https://download.pytorch.org/whl/cpu \
  -r requirements/torch-cpu.lock -d wheelhouse
python -m pip install --no-index --find-links=wheelhouse --only-binary=:all: \
  --require-hashes -r requirements/runtime-cpu.lock
python -m pip wheel --no-index --no-deps --no-build-isolation . -w dist
python -m pip install --no-index --no-deps dist/pytorch_from_zero_lab-0.1.0-py3-none-any.whl
python -m pip check
python -I scripts/check_installed.py --output results/local-installed.json
```

获取阶段需要网络。两个来源分别获取，避免混用extra-index造成来源歧义。运行锁全为name/version/hash与本地include，没有远程URL。后续本地安装不需要索引，也没有源码依赖构建。`--no-index`本身不是网络防火墙；若需求文件含直接HTTP URL仍可能联网。本地验证实际执行的是本地wheel安装，没有声称在本机设置了网络隔离；Docker配置把安装RUN与最终运行也断网。

自产wheel不在第三方锁内，安装使用`--no-deps`，避免解析器偷偷更换已锁依赖。发布证据记录此次wheel字节哈希。`scripts/check_installed.py`拒绝editable安装，核对导入位置等于distribution记录位置，再于临时工作目录以`python -I`启动原五条CLI和新runtime CLI。该脚本全部是原创；没有调用pip内部API。

## 新进程资源协议

```bash
python -I -m pytorch_lab.runtime_probe \
  --epochs 120 --repeats 3 --timeout 300 \
  --output results/local-runtime.json
```

- 每次样本为新Python解释器；`-I`排除当前目录、PYTHONPATH和用户site干扰。editable `.pth`仍可生效，因此`-I`不等于安装隔离，wheel检查另行拒绝editable。
- 固定Wine原协议：六次拟合、每次120轮、batch16、三种子和两个预声明架构。每个测量样本合计5,040次参数更新、77,040次训练样本访问。
- torch intra-op / inter-op各1线程；OMP/MKL/OpenBLAS环境上限为1；CPU-only torch构建才通过。
- workload wall：从数据加载/预处理开始到六次训练及评估结束，`perf_counter`差值；包含首次运行延迟，不做warmup，不含初始import/thread setup及报告序列化；工作期间的PyTorch懒加载/初始化仍在计时内。
- workload process CPU：同一段`process_time`差值，含本进程各线程CPU时间，不是整个机器占用率。
- end-to-end wall：父进程从启动子解释器到退出，含导入、线程设置、计算和JSON序列化。
- RSS：Linux `getrusage(RUSAGE_SELF).ru_maxrss`，单位KiB，再除1024得到MiB。本进程启动以来的高水位，包含解释器、库和allocator；不是张量占用、当前RSS、训练增量或容器整体内存。
- 新进程不等于冷文件缓存；共享宿主调度、频率、页共享与资源上限都会影响计量。不作速度提升、能耗、GPU、FLOP或统计显著性结论。

每个样本保存完整代码/数据/许可证文件SHA-256及运行版本，科学结果的canonical JSON SHA-256排除且只排除`created_at_utc`与`environment`。跨样本比较输出、环境和包文件；任何变化使总检查失败。超时/异常退出为2；重复性检查失败保留失败证据并退出1；现有输出文件/断链符号链接拒绝覆盖。

## 实际观测

[源码环境](../results/2026-10-10-runtime-source-cpu.json)与[干净wheel环境](../results/2026-10-10-runtime-wheel-cpu.json)各3次120轮。两者都为CPython3.12.14、torch2.14.1+cpu、Linux x86_64、glibc2.41；source开发环境可见额外系统包，wheel环境仅10个运行依赖、项目与pip。source有NumPy2.3.5，最小wheel环境没有；后者有一条可选NumPy警告但实验不使用NumPy。

| 环境 | workload wall中位数（范围）秒 | process CPU中位数秒 | 高水位RSS中位数MiB | end-to-end wall中位数秒 |
|---|---:|---:|---:|---:|
| 现有源码开发环境 | 3.3467（3.0463–3.5866） | 3.3323 | 296.0664 | 5.3279 |
| 新建最小wheel环境 | 3.2098（3.1511–3.4588） | 3.1933 | 321.3633 | 5.1659 |

不是性能对照试验：环境额外包不同、共享宿主当时也有测试活动，没有隔离负载、随机交替顺序或足够重复。尤其不能把RSS差异归因于wheel安装方式。目的在于实测、定义单位/边界并证明安装后数值输出未漂移。

六个科学结果均为SHA-256 `b7ab96864bf8265cc5ec54c7318cb65bc78cbb0404174ef868a803ce3eb75fc1`，与前一日完整Wine结果一致。不据此保证所有平台、版本或硬件逐位一致。[产物与安装验证](../results/2026-10-10-runtime-artifacts.json)记录依赖及wheel哈希、负对照和环境边界。

## CPU容器与最小矩阵

[Dockerfile](../Dockerfile)分为wheel获取/构建和非root运行两阶段；基础镜像为Python Official Image 3.12.14-slim-bookworm，固定OCI index digest。已向官方registry按该digest取回manifest，核实含linux/amd64子manifest；**这不等于镜像层已拉取或容器已执行**。只支持`linux/amd64`，其它架构的wheel哈希不匹配即失败。

```bash
docker build --platform linux/amd64 --progress plain -t pytorch-lab:cpu .
docker run --rm --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges --pids-limit 128 --memory 1g --cpus 2 \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  pytorch-lab:cpu /checks/check_installed.py
```

最终UID/GID为10001:10001，无源代码工作目录或测试依赖；包代码与数据自然保存在site-packages。Docker运行命令不挂载主机目录，默认stdout输出报告。不要把已有私有文件放进镜像；`.dockerignore`采用输入白名单。镜像基础系统仍有自己的许可证，Python依赖许可证保留在安装元数据中。

| 路径 | 检查范围 | 本次发布前状态 |
|---|---|---|
| 现有源码开发环境 | 完整测试、静态检查、6条CLI、3次资源样本 | 实际执行，见验证记录 |
| 干净hash-locked wheel venv | pip check、6条CLI、3次正式资源样本、安装负对照 | 实际执行 |
| GitHub CPU source job | 同一runtime锁、完整测试与6条CLI | 待本次精确提交CI |
| GitHub CPU container job | Docker build、断网/只读/非root的已安装wheel 6条CLI | 本机无Docker/Podman，未执行；待精确提交CI |

容器job不是完整pytest；全量pytest保留在source job。没有声称跨Python次版本或跨操作系统矩阵。若容器CI失败，必须修复并重新检查精确提交，不能使用旧绿色状态代替。

## 来源与设计理解

研究PyPA维护的[pip25.1.1固定源码](https://github.com/pypa/pip/tree/01857ef79f59a98db592bacb6e7b48f354528c80)（MIT）：`src/pip/_internal/utils/hashes.py`的allow-list/缺失/不匹配分支，`commands/wheel.py`的获取与构建分层，以及`tests/functional/test_hash.py`的真实CLI负对照。pip解决包发现、依赖解析、构建及安装问题，内部API不是稳定接口。借鉴“边界可验证”和“失败也要测试”，没有复制源码。

辅助阅读Docker Community维护的[Python Official Image生成Dockerfile](https://github.com/docker-library/python/blob/688a0b86bb44289df16a363e9f41d90514c1a5f9/3.12/slim-bookworm/Dockerfile)及apply-templates（仓库MIT）；学习版本/来源校验和构建工具清理。固定镜像digest需主动刷新才能获得安全更新；本例不声称是最新或已完成安全扫描。

官方语义：[pip安全安装](https://pip.pypa.io/en/stable/topics/secure-installs/)、[重复安装](https://pip.pypa.io/en/stable/topics/repeatable-installs/)、[Docker multi-stage](https://docs.docker.com/build/building/multi-stage/)、[RUN --network=none](https://docs.docker.com/reference/dockerfile/#run---networknone)、[Python resource](https://docs.python.org/3.12/library/resource.html)、[Linux getrusage](https://man7.org/linux/man-pages/man2/getrusage.2.html)、[Python计时](https://docs.python.org/3.12/library/time.html#time.perf_counter)、[PyTorch复现边界](https://docs.pytorch.org/docs/main/notes/randomness.html)。

## 下一步

先闭合容器精确提交CI和用户独立自测。后续可练习受控单因素profiling，显式声明负载/线程/重复顺序后定位真实瓶颈；或加入依赖/基础镜像更新的兼容性回归。没有个人独立解释证据时不自动升级能力，也不为加仓库数量跳到大型模型/Agent。
