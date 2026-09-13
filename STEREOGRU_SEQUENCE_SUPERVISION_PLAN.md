# GRU 初始／逐轮监督：独立受控验证（2026-09-13）

## 1. 目标与边界

验证「匹配体出初值 → raw correlation + GEV 查表 → 三层 GRU 修正 8 次 →
初值及每轮均获得直接深度监督」能否改善深度估计。直觉是待验证假设，不是结果。

旧 C0/C1/G1 的模型、训练器、配置、结果目录及来源哈希全部保留。
新协议用新注册模块和独立入口，不修改核心训练程序，不用历史结果替代新 baseline。
GT metric 相机、冻结 clean ConvNeXt、原有 30 train / 16 dev、固定 crop/支持域、
seed 0/1/2、AdamW 1e-3、weight decay 0、clip 1、FP32/关闭 TF32、1000 步保持一致。
数据清单、缓存、训练顺序、共同**初始**参数从已完成三种子对照只读绑定；绝不继承训练后的权重。

## 2. 预注册对照矩阵（每行三个种子）

| arm | 模型 | 训练输出 | loss | 前置条件 |
|---|---|---|---|---|
| B0 | 无 GRU | q0 | 同一 sequence objective 的单输出特例 | 新 baseline，先完成三种子 |
| F1 | 有界 raw+GEV GRU，8 次 | q0…q8 | 仅 q8，权重总和 1 | 三个 B0 完成；三个独立训练门控通过 |
| S1 | **与 F1 完全相同** | q0…q8 | 初始及全部 8 轮，归一化权重 | 三个 B0/F1 完成；三个独立训练门控通过 |

- **S1 vs B0**：新监督配方下加入 recurrent refinement 的整体效果（披露新增参数/计算）。
- **S1 vs F1**：同网络、同所有初值（包括新增 GRU）、同 readout、同预算，唯一变量是监督权重。
- **F1 vs B0**：新入口内重新测量 final-only GRU 效果，不能靠旧 G1 数字作因果控制。
- 不换成加法更新、不扩训练集、不追加 seed、不选择最佳中途 checkpoint。

## 3. Loss 精确定义

所有预测是 $q=(1/d-1/80)/(1/3-1/80)$，每张预测用同一有效 GT 支持域上的绝对 L1。
S1 原始权重为初值 $a_0=1$，第 $i=1\ldots8$ 轮
$a_i=0.9^{15(8-i)/7}$；有效权重 $w_i=a_i/\sum_{j=0}^{8}a_j$。
F1 权重为 `[0,0,0,0,0,0,0,0,1]`，B0 为 `[1]`。

相对权重依据 [IGEV-MVS 固定版本训练代码](https://github.com/gangweiX/IGEV/blob/e02312b8f9615c52346bd04c08f8d219c64c404d/IGEV-MVS/train_mvs.py)。
**总和归一化是本实验显式差异**，不冒称原版 unnormalised sequence loss。
仍保留 G1 的有界更新、骨干/聚合/上采样，故不是完整官方 IGEV-MVS 复现，也不是 RGB-only 榜单。

## 4. 必须由测试证明的实现契约

1. q0 使用未 detach 的 softargmin index 和**初始** fine hidden，不能用最终 hidden 回填。
2. qi 在当轮更新后、下次坐标 detach 前读出；坐标 detach，raw/GEV/hidden 不 detach。
3. S1 九个输出都有非零直接 loss 梯度；每轮 index update 输出有直接梯度；q0→softargmin 直接梯度存在。
4. raw lookup 读取 G8 raw 的 group mean，不是 softmax；GEV lookup 读取 logits；两个查表分支均可微。
5. F1/S1 state_dict/参数量/初始化逐字节相同；F1 最终输出、梯度和 BN 更新与旧 G1 对齐。
6. 0 轮与 C1/B0 像素级相等；显式 eval sequence 的最后一张与正常 eval 相等；训练最后一张与同模式旧 G1 相等。
7. loss 长度/有限性/权重/支持域严格检查；零权重路径不反传；总 loss 不因预测数隐式放大。
8. 新 B0 从共同初值完成后回放旧 C1 final 指标（固定 rtol 1e-4/atol 1e-6），验证新接口没有改变 baseline。

## 5. 执行顺序与门控

准备新目录 → 三个 B0 正式各 1000 步 → 三个 F1 门控 → F1 重置后三个正式 run →
三个 S1 门控 → S1 重置后三个正式 run → 汇总全部末步配对结果。

门控每 seed 固定两个训练 clip（Scene01/02 各一个），200 步，不使用 dev 调参；
初末 weighted sequence loss 和 final index-L1 均须降至初值的 0.5 以内，关键模块和直接输出梯度须通过。
门控权重不保存、不进入正式初值；共 1200 额外门控更新，另计 9000 正式更新。
不执行通过率搜参，不放宽门槛，不自动重跑失败门控。任何缺 baseline/改变缓存/短预算/哈希不符都拒绝后续阶段。

保存共同源证书、初值、所有训练 clip 顺序、每项 loss、每轮更新诊断、初末指标、最后一轮和各轮完整 train/dev 指标。
主比较固定 final1000、三个 seed 全计，报告原值/均值/样本标准差/配对差。每轮指标是机制读出，不能用其挑最优轮数。

## 6. 验证状态

- 改动前回归：50 项现有 corrected-head/workflow/matched-control/repeats 测试通过。
- 2026-09-13 本窗口最终相关回归：**163 passed / 305.68 s**；范围为全部
	`test_stereogru_*.py`、权重恢复、decoder/backbone/objective 注册表测试。
- 小尺寸 CPU 合成端到端：先生成三个种子的历史 C0/C1 证书，再执行新的三个 B0、
	F1 三个门控及三个正式 run、S1 三个门控及三个正式 run，最后重算配对汇总。
	fixture 正式各 4 步、门控各 60 步，**不是**真实 1000/200 步实验；验证历史源目录字节不变、
	缓存不重提取、共同初值/空优化器、门控权重不保存、前置缺失/篡改/失败拒绝后续阶段。
- 正式头设置 CPU smoke：4 帧 × 256²，ConvNeXt 通道 96/192/384/768，G8/D32、
	match96/hidden128、8 轮。使用**随机缓存特征＋合成相机和深度**；验证新 F1 与旧 G1
	train/eval 输出、全部参数梯度、BN buffer 和一次 AdamW 更新逐元素相同。
	实际 runner 的验证函数确认 q0…q8 的直接 loss 梯度和 8 次 index update 梯度全为正且有限。
- 单独截获三层 GRU 的全部 8 轮 hidden，证明仅末轮 loss 仍穿过全部 recurrent hidden；
	坐标 detach 不被误扩展到 hidden/volume，raw 和 GEV 的直接查表梯度均验证。
- fresh-process CLI 测试故意屏蔽 xFormers 并触发注册发现：依赖提示进入 stderr，
	stdout 仅保留一条 `SEQUENCE_NEXT` 控制记录。shell 测试使用假解释器/假设备查询，
	验证完整动作解析、污染输出/非法 arm/seed 拒绝和子命令失败停止；没有发起 GPU 作业。
- 独立模型/流程审阅后修复：控制输出污染、生产 200 步门控绕过、源码指纹遗漏自动发现模块、
	非 prepare 命令内重复完整验证。复审未发现这些阻塞仍存在。
- 新增/改动文件的编辑器诊断、shell 语法检查和 whitespace 检查通过。
- 阿里云 GPU 真实精度实验尚未启动；本窗口无法直接访问阿里云终端。

## 7. 运行交付与禁止事项

- 独立入口：[scripts/run_stereogru_sequence_experiment.sh](scripts/run_stereogru_sequence_experiment.sh)。
	Python 入口不走核心训练程序，不能把本配置直接交给核心训练程序来代替它。
- `REPETITIONS` 指向已完成三种子 C0/C1 的 runs 根目录；已知远端为
	`/mnt/data/PROJECT_CHEN/code/PP-DPT/stereogru_seed_repeats_FCEyWuB3/runs`。
	`RUN_ROOT` 默认 `/mnt/data/PROJECT_CHEN/code/PP-DPT`，`PYTHON` 默认 `/usr/local/bin/python`，`GPU` 默认 5。
	入口每阶段检查显存/compute 进程，忙则停止，不杀其他任务、不自动选择别人的 GPU。
- `SEQUENCE_RUN` 未设置时创建全新目录；设置时只接受已准备的实验，并只能进入**尚未开始**的阶段。
	部分训练/失败门控绝不自动重试，不从断点恢复，也不重用旧 G1 目录。
- 发布时只提交本实验的新增模型、loss、配置、runner/shell、三份测试和本文档。
	用户现有的核心训练程序、综述页面、图片和旧 GPU smoke 改动不得夹带。
	远端使用发布提交的 `model loss dataset config scripts` 完整只读源码快照；不要在训练工作树
	上 pull/reset/stash 或切分支，不触碰 checkpoint 软链/真实目录。
- `source_inventory()` 对上述五棵本地代码/配置树重新枚举并绑定指纹，新增或删除自动发现模块
	也会使准备证书失效。第三方包仅记录既有 runtime 版本，不宣称完整供应链锁定；
	普通哈希检测不一致修改，不防御证书与全部文件一起被恶意重写。
- 一次完整运行包含 **9000 正式更新＋1200 丢弃的门控更新**；只有全通过才输出
	`SEQUENCE_FINAL_STEP=1000` 和所有种子/场景的末步对照。中途输出不是最终方法结论。

### 仍待真实设备验证的边界

1. **CPU 正式尺寸通过不等于 CUDA 通过**：真实缓存、GPU 显存/运行时间、GPU 内核数值、
	 200 步训练门控和实际精度必须在阿里云验证，不能由本地合成测试代替。
2. 本版保留严格 B0 验收：新 B0 的 final1000 指标还需匹配旧 C1 的固定容差
	 `rtol=1e-4, atol=1e-6`。这是**重新训练轨迹的一致性要求**，不只是旧 checkpoint 回放。
	 CUDA backward 可能存在非确定性，历史种子也不全在同一 GPU；因此该要求未被保证能通过。
	 若失败，将保留新 B0 的 checkpoint、实际/期望/diff 和 failure 记录，禁止后续方法；
	 先核查设备/运行时/训练数值，不在失败后放宽门槛或把失败 B0 当有效基线。
3. 仍需读输入、权重与缓存做完整指纹验证，远端共享存储耗时未测。
	 入口会先打印 `VERIFY` 状态，每个 CLI 阶段仍有 3600 秒上限；不将验证静默时间当作训练进度。
4. 本版**本地实现和验证已完成，阿里云真实实验尚未运行**。
	启动须使用包含本实验九个新增文件的固定发布提交；发布状态以远端提交核验为准，
	不使用旧提交或训练工作树的混合代码。长任务用 `setsid nohup` 隔离终端会话，
	为本次启动创建独立日志；看到后台 PID 不等于准备成功或训练已经开始。