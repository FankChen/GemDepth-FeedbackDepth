# StereoGRU：depth 轴信息的匹配训练对照

日期：2026-09-10。状态：**v2 的 seeds 0/1/2、共六个 C0/C1 run 全部完成，均取固定 final1000。** 完整体的开发集 AbsRel/RMSE/δ1 均值更好，但前两项逐 seed 会翻转，index-L1 三组更差；这是有限正信号，不是稳定全面提升。本轮对照收束，不继续增加种子。最终统计与下一阶段边界见第 11 节。

**当前执行版本为 v2：30 train / 16 dev。** v1 的阿里云准备阶段确认 Scene02 仅有 15 个满足运动范围与 8 帧起点间隔的窗口，无法满足每训练场景 16 个；当时任何 arm 均未训练，随后显式将 Scene01/02 各改为 15 个，保持场景平衡、dev=16、运动/间隔/预算不变。v2 配额现已通过，三个 seed 的 C0/C1 均完成。旧配置和失败输出保留，不绕过配额检查。详见第 7–11 节。

## 1. 这一步只回答一个问题

在校准几何与绝对 index 监督已经成立的前提下，**真实 depth 轴变化是否比仅保留逐像素/group 均值提供可迁移的增益**？

依据：[已完成 pilot](results/stereogru/20260910_calibrated_pilot.json)证明可学习；[只读干预](results/stereogru/20260910_pilot_readout.json)显示清空体会退化，但拉平 depth 的影响小，反转 depth 在未拟合场景上的 AbsRel/RMSE 方向不一致。现有结果不足以说“已学会可靠三角化”，也不足以说“几何/GRU 不 work”。

只读干预属于固定模型的输入分布改变，不是重新训练的控制组。下一步必须比较**各自从同一初始状态训练**的模型，避免把失配 BN/输入分布造成的损害当成几何贡献。

## 2. 最小矩阵与执行依赖

记原始融合体为 $C(n,g,d,h,w)$，其中 $d$ 是已校准的 inverse-depth bin：

$$
\bar C(n,g,d,h,w)=\frac{1}{D}\sum_{k=1}^{D}C(n,g,k,h,w).
$$

| ID | 训练与评测送入 aggregation 的体 | 保持一致 | 待回答问题 |
|---|---|---|---|
| C0：flat baseline | $\bar C$ 沿 depth 重复 | 原始 plane sweep、matcher、view weighting、guides、3D aggregation、上采样、监督和参数数量 | 保留空间/group 均值而不保留逐 bin 变化时，能做到多少？ |
| C1：full volume | 原始 $C$ | 与 C0 相同 | 真正的逐 bin 变化是否带来开发集增益？ |
| G1：条件项 | full volume + 正确 raw/GEV lookup 的 GRU | 另定义与完成的 C1 的预算/初始化匹配协议 | 在已验证的 geometry 上，迭代是否进一步有益？ |

**C0 完整跑完并核对 artifact 后，才允许 C1；G1 不在当前执行清单。** 当前旧的两 clip full-volume pilot 不是这张新表的 C1，不能直接复用其最终权重补表。

C0 不是纯单帧、不是无相机/无几何 baseline：均值来自带 GT 相机、depth bounds 和有效性的真实体，仍可携带多视角/遮挡/空间信息。它只控制**显式 depth 轴变化**。C0 仍保留相同的全部体构建计算，不把减少计算或替换网络容量混进贡献。

## 3. 固定的首轮开发协议

这是**新 split、新预算**，与旧历史主表分开，所有数字只在本矩阵内比较。

| 项目 | 预注册内容 |
|---|---|
| 场景 | Scene01/02 用于训练；Scene18 固定为开发场景。它是原 training pool 的开发划分，不是未使用过的官方测试集 |
| 最终测试 | Scene06/20 以及四集 RGB-only benchmark 不用于这轮挑配置；GT-camera 实验不进入 RGB-only 榜 |
| 训练样本 | 当前 v2 预先固定 30 个 clip，Scene01/02 各 15 个；原 v1 的 16+16 配额未通过准备检查。每段连续 4 帧，同场景起点至少间隔 8 帧，禁止帧重叠 |
| 开发样本 | Scene18 预先固定 16 个 clip；同样不重叠，不按当前模型误差选择，不逐轮更换清单 |
| 运动条件 | 沿用 GT 最大首帧相对平移 .5–5m；共同适用且完整报告。它是机制子集，不是完整数据集分布 |
| 数量不足 | 停止并重新登记共同协议，不从开发集回填、不自动放宽阈值、不随机重试坏数据 |
| 相机/图像 | GT metric K/T，逐帧 K、float64 pose 重基、同一确定 crop256/中心映射；两臂同一 manifest、crop、顺序 |
| backbone | 已验证官方 342 tensors，冻结且 eval，无 GEM/LoRA；缓存相同 features，记录 file/state 指纹 |
| 输出与目标 | 相同 3–80m、32 inverse-depth bins，绝对 normalized-index L1；不加 SSI、.005 floor、额外 CE/正则或新的相机目标 |
| 优化 | 每组 1000 updates，batch=1 clip、AdamW lr=.001、wd=0、grad clip=1、FP32，与 pilot 相同优化器形式 |
| 初始化 | C0/C1 加载同一个独立保存的全新 head 初值，公共 state keys、数值/BN buffers、参数量和哈希完全相同；优化器均重新初始化 |
| seed | 首轮 seed=0；若有开发信号，再按相同预算补 paired seeds 1/2。每个 seed 的 C0 完成后才启动对应 C1 |
| checkpoint | 主比较固定最后第 1000 步，不从 train/dev 曲线挑最优。中间每 250 步的结果只记录，不混表 |
| 成本 | 单 H20 顺序运行，每 arm 内部检查 3600 秒预算，shell 同时限制训练进程 3600 秒、TERM 后 30 秒仍未退出则强制结束；超限不产生完成证据，不能以半程结果对比另一组全程 |

训练 scene 覆盖、间隔和数量需要在生成 manifest 时计数验证；v1 已回传间隔可用数 48/15/41，v2 的完整清单/图像/监督检查仍由准备阶段执行。1000 步/30 clips 是定额开发机制实验，不叫完整多域训练，不从先前 9.68s 的缓存两 clip pilot 推断真实吞吐。

本次不同时更换深度范围、相关归一化、hourglass 容量、RGB stem、teacher 或 loss，以免发现增益后仍不知道是哪一项生效。

## 4. 实现边界与验收

- C0 已实现为新增注册类 [DPTHeadCalibratedFlatVolumeConvNeXt](model/dpt_calibrated_flat_volume_convnext.py)，只实现体到 depth-mean 体的转换；C1 复用 [calibrated head](model/dpt_calibrated_volume_only_convnext.py)。差异通过新 config/注册模块表达，**不修改旧 head、旧 pilot 或核心训练文件来堆实验分支**。
- 训练和 eval 都一致使用各自 raw/full 或 flat 输入，不在评测时才突然切成 flat。两臂各自学习的 BN 统计是训练结果，不手工搬用另一臂的统计。
- 首先验证 C0/C1 参数/state keys/初值相同、flat 保留逐像素/group 均值且 depth std 为零、matcher 梯度正常、共享 K/T/mask 不变。非法相机直接报错，不通过数值填充隐藏。
- 保存 manifest、clean backbone 指纹、共同 initial head 哈希、各自最终 head 哈希、完整训练配置/源码、已完成步数及输入文件 hash。不能再次遗漏最终 head 的独立文件指纹。
- 预定义场景级开发报告 **AbsRel、δ1、RMSE**，无 affine fit；固定 GT 支持与 bounds，同时保留每 clip 值。index-L1、raw std、shuffle sensitivity 仅用于排错，不成为新排名指标。
- 同一个开发场景的相邻 clip/百万像素不是独立统计样本；单 seed 或单场景小差异不能宣布普遍提升。后续 paired seeds 仍不能替代跨场景/跨数据集验证。

## 5. 结果如何改变下一步

| C0/C1 开发观察 | 下一步 | 禁止的解释 |
|---|---|---|
| C1 跨 seed 在预定标准指标上有稳定、无明显相反退化的收益 | 继续完成该校准协议的完整 volume-only baseline，再设计 G1 | 直接称 RGB-only SOTA 或把多视角收益归功于尚未运行的 GRU |
| C1 仅训练集更好，Scene18 无稳定收益 | 先处理覆盖/过拟合和匹配监督，另登记单因素验证 | 对两 clip 继续多训就算修复完成 |
| C0 不输，或 AbsRel 与 RMSE/δ1 持续冲突 | 如实判定当前配方没有证实 depth 轴增益；检查特征/监督契约，不预设胜者 | “平体更好，所以 StereoGRU 无效”或“加 GRU 必然救回来” |
| 两臂都无法正常学习 | 回到新实现/数据契约排错，暂停方法组 | 扩预算、换评测器或跳过失败样本来形成正结果 |

**本轮 readout 与三种子 C0/C1 均已收束。** 不再要求用户反复运行同类 zero/flat/reverse 诊断或追加种子。以下保留已完成实验的执行契约，所有结果仍按相同协议解释。

## 6. 实现与运行契约

当前新实验入口：[scripts/run_stereogru_matched_30train.sh](scripts/run_stereogru_matched_30train.sh)；v2 固定配置：[config/stereogru/matched_c0_c1_30train.yaml](config/stereogru/matched_c0_c1_30train.yaml)。训练/验证仍用原 [scripts/stereogru_matched_controls.py](scripts/stereogru_matched_controls.py) 与 [dataset/stereogru_matched.py](dataset/stereogru_matched.py)。[原 launcher](scripts/run_stereogru_matched_controls.sh)默认新建使用 [v1 配置](config/stereogru/matched_c0_c1.yaml)，不要再次用其空 `MATCHED_RUN` 默认路径创建 v2；v2 wrapper 显式指定配置后再传已准备好的实验目录给它。

### 6.1 准备一次，两个 arm 共享

- 默认新建独立实验目录，先在 CPU 检查三场景 quota、输入文件、相机、有效深度 mask；不依赖旧 checkpoint 软链布局，不扫描测试场景。
- 候选按起始帧排序，先贪心取最大间隔集合，再按时间均匀抽取固定配额；这样不会因随机抽取失误而把可行配额误判为不足。每场景打印原始帧、连续窗口、运动有效窗口、满足间隔的数量与最终选择。配额不够保存失败报告并终止。
- 保存共同 config、manifest（含每 clip RGB/depth/标定路径、完整变换与相机、有效像素及 mask SHA256）、训练 clip 顺序和**独立全新 initial head**。共同初值与各 arm 的 state keys/参数数量逐项验证。
- 验证 clean backbone 的全部 342 个 native tensor 指纹，不采用旧 LoRA 或 pilot 最终 head。阿里云可直接复用此前已验证恢复的权重，不再下载/搜索/重复恢复。
- C0 首次运行构建共享 CPU 特征缓存；C1 只能加载相同缓存，不能重新生成、换样本或覆盖。缓存生成成本单列，不能用 C0/C1 的含缓存总耗时直接宣称速度差异。

### 6.2 每次只启动一个 arm

- 默认 `ARM=C0`；无旧实验路径时先准备新目录，只运行 C0。完成后打印 `COMPLETED C0` 和实验绝对路径，不自动启动 C1/GRU。
- 后续 `ARM=C1` 必须提供同一个 `MATCHED_RUN`。C1 的初值仍来自共同 initial head，优化器为空状态，绝不加载 C0 的最终权重或 BN 统计。
- 准备好的 config、源码、runtime、原输入、共同初值与顺序必须与指纹一致；此协议显式要求 C0=flat、C1=full 的有序配置，不能把 full 改名为第一组来绕过 baseline。
- C0 完成记录须匹配整个预算、相同 initial/manifest/order/cache、独立最终 head 文件/状态指纹以及逐 clip 指标/支持域；最终记录在 artifact 校验后才写出。缺失、部分训练、篡改或短预算都会在 C1 创建目录之前被拒绝。
- 每 100 步打印 loss/梯度/有效 raw 统计与粗略剩余时间，每 250 步分别评测固定 train/dev；主比较只用最后第 1000 步。场景和每 clip 的 AbsRel/δ1/RMSE 全部保存，不挑 best-dev checkpoint。
- 所有失败/超时保留证据，但不恢复覆盖原目录、不自动调整训练预算。共享 manifest/实验不满足要求时，先看失败原因，不用旧 pilot 充作已完成的对照。

### 6.3 安全与验证

GPU 在准备前和真正起训前各检查一次（降低竞态，但不是调度器锁）；内部时限也覆盖逐 clip 读取/缓存，shell 进程时限处理阻塞 I/O 或 CUDA。所有输出在新目录；旧 train/model/loss/config、旧 pilot 和 checkpoint 都保持不动。

本地相关回归 **107 passed（15.62s）**。包含合成完整 C0→C1、共同初值/顺序/缓存、开发隔离、配额失败、篡改/短跑/跳过 baseline、缓存中超时、最终指标聚合，以及旧 pilot/readout/恢复/注册表回归。另以真实官方 backbone 和合成 RGB/GT 完成两头前后向：C0 的 raw depth-std=0、两头 matcher 均有有限非零梯度；**未做真实数据优化，不构成方法分数**。

真实 v2 配额已通过、seed0 两臂均已完成；重复 seed 改变初始化和训练顺序，但 manifest/crop 不变，每个 seed 内 C0/C1 共享初值与缓存，不能使用另一臂的最终权重。

## 7. v1 配额失败后的显式 v2 修订

原始证据：[preflight record](results/stereogru/20260910_matched_quota_preflight.json)。Scene01/02/18 的运动合格窗口分别为 375/120/323，满足起点间隔后的容量为 **48/15/41**。Scene02 的 `selected=0` 是配额失败时拒绝部分清单，**不是没有有效数据**。失败发生在准备阶段、生成完整 manifest/初始化/优化器之前；不应重跑旧 readout 或启动 C1。

新 [v2 配置](config/stereogru/matched_c0_c1_30train.yaml)与 v1 的解析差异严格只有两项：`protocol` 改为 `vkitti_calibrated_depth_axis_control_v2_30train`，`dataset.splits.train.clips_per_scene` 从 16 改为 15。开发配额仍 16、start gap 仍 8、motion 仍 .5–5m、两臂仍各 1000 updates。参数/目标/初始化策略/优化器和 C0 完成门槛不改。

这是依据可用数据量、在任何 C0/C1 训练前登记的共同配额修订，不根据模型结果挑样本，也不是运行时自动放宽限制。新目录重新生成共同 manifest、crop 分配、训练顺序和初值契约，两个 arm 必须同用 v2；旧失败目录不覆盖、不作 C0 结果。原始 trainer、dataset、v1 配置和完成验证器逐字节保留。

新增薄 wrapper 只在 CPU 用 `--config` 显式准备 v2，再将 `MATCHED_RUN` 传给原启动器单跑 C0；原 GPU 占用检查、超时和 C1 完成依赖仍执行。v2 配额若因数据变化再次不足，仍直接报错，不降到别的数量。后续 C1 使用同一已完成的 v2 实验目录，不重新准备 v1。

v2 相关验证合计 **111 passed（15.23s）**；新增测试确认解析后只改变上述两项、可用 15 个的边界仍拒绝配额 16、开发集不变且 C0/C1 仍各 1000 updates。shell 语法通过；旧 trainer/dataset/model/v1 配置与 `d35ce77` 完全一致。随后用户完成的 C0 见下节，未因实测修改训练器。

## 8. v2 C0 完成记录

记录来源：[20260910_matched30_C0.json](results/stereogru/20260910_matched30_C0.json)，用户提供的阿里云 stdout。实验标识 `stereogru_matched30_cJPdMuhW`，源快照 `d4f3388`，GPU 4；配额为 Scene01=15、Scene02=15、Scene18=16，46 个 clip 的共享缓存已完成。head 参数为 **1,234,505**，共同初值状态指纹为 `935d88d2f8e591872798c06c8805871bec857d033e6e3a02a25607969bbd99b0`。

| C0 最后第 1000 步 | train（30 clips） | dev / Scene18（16 clips） |
|---|---:|---:|
| 有效像素 | 5884702 | 3591702 |
| index-L1 | .01682694 | .04113862 |
| AbsRel | .11015101 | **.23792445** |
| RMSE | 7.46385419 | **9.81047361** |
| δ1 | .86170226 | **.62542661** |

终端已输出 `COMPLETED C0`，且没有启动下一组；C1 入口仍会重新核验完成记录、预算和所有指纹后才起训。本机只归档用户返回值，没有取得并独立复算远程最终 checkpoint。

主比较严格使用 final1000，不把 step750 较低的 dev AbsRel `.23561281` 挑出来。所有打印训练步的 raw depth-std 为 0，符合 C0；聚合后 logits 可以有 depth 变化，因为网络仍可从空间/group 均值、guides 和 depth 位置相关的计算生成非均匀读出。这不代表平体操作失效，也不证明逐 bin 匹配有益。

日志中的 **21.55 秒**是循环起点到最后一步打印的时间，尚未计入最终评测、保存及完成验证，并排除了前期准备和缓存，不能当整次训练墙钟时间或预言 C1 耗时。

**C1 不再重建实验、不重新恢复 backbone、不加载 C0 最终权重。** 以原 `d4f3388` 快照执行原 launcher，设 `ARM=C1`、`MATCHED_RUN` 为已完成 C0 的 experiment 目录；它复用全部共享 artifact、从共同初值和空优化器开始 1000 步。完成后调用现有 `compare`，自动校验两臂并打印 final-step 的逐场景指标表；不改 loss/预算或加 GRU，也不根据 C0 分数提前判断胜负。

## 9. seed0 首轮结果：AbsRel 变差，RMSE/δ1 变好

C1 已在同一 `stereogru_matched30_cJPdMuhW/experiment` 完成并通过原比较入口校验。用户回传原值见 [C1 evidence](results/stereogru/20260910_matched30_C1.json)，源快照 `d4f3388`，GPU 4，1000 步，下一组未启动。两臂 train/dev 支持分别始终为 5884702 / 3591702 像素。

| Scene18 开发集，final1000 | C0 flat | C1 full | C1 相对 C0 |
|---|---:|---:|---:|
| index-L1 | .04113862 | .04244677 | +3.18%（更差） |
| AbsRel ↓ | .23792445 | .25194988 | **+5.89%（更差）** |
| RMSE ↓ | 9.81047361 | 8.80519551 | **−10.25%（更好）** |
| δ1 ↑ | .62542661 | .63985013 | **+1.44 个百分点（更好）** |

训练集整体 C1 的 AbsRel 仅改善约 1.11%，RMSE 改善约 13.20%，δ1 增加约 1.43 个百分点。逐场景仍不一致：Scene01 训练 AbsRel `.120133→.111419`，Scene02 为 `.097927→.105882`。这不是“只有训练集全部变好”的简单结论，也不能从总体误差分配倒推出远近景或特定物体原因。

开发曲线在相同步数下：

| step | C0 AbsRel | C1 AbsRel | C0 RMSE | C1 RMSE | C0 δ1 | C1 δ1 |
|---:|---:|---:|---:|---:|---:|---:|
| 250 | .338420 | .263011 | 10.8635 | 9.1068 | .560578 | .617973 |
| 500 | .327492 | .241382 | 10.3482 | 8.5338 | .597610 | .663298 |
| 750 | .235613 | .217690 | 9.8883 | 8.2817 | .606046 | .683885 |
| **1000（主比较）** | **.237924** | **.251950** | **9.8105** | **8.8052** | **.625427** | **.639850** |

较早检查点说明完整体存在有利信号，但 **不能在看完曲线后将主比较改成 750 步**。C1 的最后 250 步开发指标退化与优化随机性/过拟合/BN 等解释都相容，当前仅一个 seed、一个开发场景，不能确定根因或把差值称作“噪声内”。也不能据此宣布 StereoGRU 失败：这一对根本没有 GRU。

**当前结论：通过了第一组可比的训练对照，但尚无跨 seed 一致的 accuracy 提升。** 暂不修改 loss、学习率、训练预算、评测点或模型；先完成原计划列出的 seed1/seed2 配对重复。它是一次有限的重复性核验，不是不断跑 seed 直到出现赢家。

## 10. 配对重复入口：固定 seed0/1/2，只有 seed 变化

新增 [scripts/stereogru_matched_repeats.py](scripts/stereogru_matched_repeats.py) 与 [scripts/run_stereogru_matched_repeats.sh](scripts/run_stereogru_matched_repeats.sh)。源 seed0 实验只读，必须先核验两臂完成；旧 trainer、数据/模型实现、v2 配置均保持不变。

- 新目录准备 seed1/2：复制 seed0 的已解析 config，**只修改顶层 seed**。crop_seed、manifest、相机、mask、bounds、loss、预算、优化器、runtime、源代码和 clean backbone 都校验一致；manifest 文件内容及指纹必须与 seed0 完全相同。
- 不同 seed 的全新 head 初值必须不同，训练顺序由对应 seed 固定生成；每个 seed 内的 C0/C1 仍共享同一初值、顺序和缓存，各自优化器从空状态开始。
- 一条入口会明确顺序运行 **seed1/C0 → seed1/C1 → seed2/C0 → seed2/C1**，共四个 1000 步 run。每次调用原单 arm launcher，GPU 忙/超时/配额/指纹或前置完成验证失败都停止；不是并行启动或放松 baseline 门槛。无 GRU 自动后续。
- 每个新 seed 单独构建相同输入的冻结特征缓存，并在该 seed 两臂间共享；不会覆盖 seed0 缓存或权重，也不伪称复用了其含 seed0 契约的缓存文件。
- 三个 paired seeds 全部完成后，只读取各自 **final1000**，按场景输出 C0/C1 的原始值、均值/样本标准差，以及每 seed 的 `C1−C0` 差值和其均值/样本标准差。不得遗漏不利 seed、用 best-dev 值替代，或将标准差当作独立场景的统计显著性。
- 重复后若取舍保持，应明确报告指标取舍并停止宣布“整体提点”；若方向随 seed 翻转，应报告不稳定。若有一致信号，也仍只属于此 GT-camera 开发子集，不代表 RGB-only/SOTA。

该入口只追加预登记的两个 seed，不改变 seed0 的任何结果或训练代码。用户现已完成真实 seed1/2：最初 GPU4 占用导致准备后停下，随后使用同一已准备目录改为 GPU5，四组全部完成。没有重新挑清单或改变参数，也没有自动运行 GRU。

本地相关验证 **119 passed（23.89s）**，新增测试覆盖只改 seed、manifest/crop/预算或 runtime 改动拒绝、三种训练顺序、未完成任一 pair 拒绝汇总、样本标准差/配对差，以及合成 seed0→seed1/2 的完整执行和原实验文件不变。shell 语法检查通过；旧 trainer/model/data/config 与 seed0 快照仍逐字节一致。未在本地执行真实数据重复训练。

## 11. 三种子最终结论：有用的初步信号，稳定性仍有限

完整数值见 [three-seed evidence](results/stereogru/20260910_matched30_three_seeds.json)。来源为用户回传的四个重复 run 日志和最终汇总；复算了开发集全部均值/样本标准差与 12 行配对差统计，和输出的精度一致。六个 run 均完成 1000 步，开发集每次 3591702 个有效像素；不选 seed，不替换为更早 checkpoint。

### 11.1 Scene18 开发集，跨 seed 均值 ± 样本标准差

| 指标 | C0 flat | C1 full | 均值变化 | 逐 seed 方向 |
|---|---:|---:|---:|---|
| AbsRel ↓ | .256214 ± .021895 | **.233526 ± .016117** | **−8.86%** | 2/3 更好；seed0 更差 |
| RMSE ↓ | 9.691719 ± .139123 | **9.400443 ± .656135** | **−3.01%** | 2/3 更好；seed1 更差 |
| δ1 ↑ | .626705 ± .005754 | **.636903 ± .003354** | **+1.02 个百分点** | 3/3 更好；seed1 仅 +.0264pp |
| index-L1 ↓ | **.041804 ± .000722** | .043037 ± .000612 | **+2.95%（更差）** | 3/3 更差 |

“−8.86% / −3.01%”是**三个 seed 指标均值的相对变化**，不是每 seed 相对百分比的平均；δ1 用绝对百分点。不能把它们写成完整测试集、跨域或 RGB-only 提点。

| seed | C0 AbsRel | C1 AbsRel | C0 RMSE | C1 RMSE | C0 δ1 | C1 δ1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | .237924 | .251950 | 9.810474 | 8.805196 | .625427 | .639850 |
| 1 | .280476 | .226592 | 9.726032 | 10.103989 | .632990 | .633254 |
| 2 | .250243 | .222036 | 9.538650 | 9.292145 | .621697 | .637604 |

AbsRel 配对差 `C1−C0` 为 `.014025 / −.053884 / −.028207`，均值 `−.022688`、样本标准差 `.034289`；RMSE 为 `−1.005278 / .377957 / −.246505`，均值 `−.291275`、样本标准差 `.692704`。波动及方向翻转需要如实保留；**不能仅看均值宣布稳定优势，也不能仅凭“标准差大于均值”就断言无效或噪声内**。这里只有三个 seed，且共享同一开发场景，不作统计显著性或独立场景置信结论。

### 11.2 index-L1 变差不是应被删除的反例

index-L1 对 normalized inverse-depth/index 计量，AbsRel、RMSE 和 δ1 对还原后的 metric depth 计量；它们对深度区间和误差尾部的权重不相同，因此可以出现前者更差而后几项均值更好。**这只说明指标并非单调对应，不足以定位具体近景/远景、动态物体、BN 或优化原因。** 不根据此次结果换掉目标或重算评测。

训练 Scene02 的 index-L1 和 AbsRel 也三组都比 C0 差；Scene01 的若干指标会翻转。完整体不是在所有场景、训练目标或种子上一致更强。更准确的表述是：**在这个匹配训练的开发子集里，保留 depth 轴信息呈现标准深度指标的平均收益，但结果异质，尚未达到稳健方法结论。**

### 11.3 本轮停止条件与下一阶段

1. **结束 C0/C1 诊断循环。** 保存全部六组、当前均值与不利方向；不追加 seeds3/4，不调整到750步，不重新挑 crop 或样本。当前没有未完成的本轮训练，也没有后台自动任务。
2. **把 C1 作为修正后 GRU 的可比较基线，而不是宣称已战胜平体。** 它已在同协议完成三种子预算，可支撑下一步小规模 GRU 实现验证；这属于探索是否能进一步提取几何信息，不是统计优势已经证明后的全面推广。
3. **下一项工程工作是新注册的 calibrated GRU head。** 恢复 raw correlation + regularised GEV 两路 lookup，保持 GT metric 相机与绝对 index 坐标，逐轮检验 lookup/梯度、有限输出与图像依赖；先合成合同测试和训练场景小样本，不复活旧坏相机 checkpoint，不静默改历史 head。
4. **增益对照的直接 baseline 为 C1。** 尽量保留同 manifest/crop/cache、三个 seed、1000 updates、优化器和最后一步 evaluator；公共 matcher/GEV/上采样参数从各 seed 的共同初值映射，而非用 method 额外预训。若第一轮仅比较 GRU 加入，可保持现有最终预测的 absolute-index L1，不同时新增迭代辅助 loss。新增循环参数/计算须报告，不能将结果直接称为纯迭代机制的贡献。
5. **任何契约变化都重新定义匹配 baseline。** 如果为了 GRU 改深度范围、loss 权重/中间监督、共同相机单位、训练预算或初始化，不能借用现有 C1 当完全匹配基线，必须先跑相应新 baseline。GRU 单测/小样本通过前不直接开完整 run；RGB-only 预测 K/T 的修复与标准跨域评测仍是更后面的任务。

上述 GRU 下一阶段仅明确工程/实验边界，**当前尚未实现新 GRU、尚未运行 G1，不能给出已可运行的 G1 指令或称其有收益**。本次交付是完成统计核验、证据归档和结论收束，不要求用户再跑旧命令。