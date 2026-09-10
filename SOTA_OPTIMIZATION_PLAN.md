# PP-DPT：面向标准视频深度 benchmark 的优化方案

审阅日期：2026-09-09。代码依据：`feat/registry-mixdata@eb2c9ac`。状态：**研究设计，未启动新实验、未验证预期增益**。

本次核对了当前实现、已有实验记录、跨会话记忆、DenseGRU / DA3-Metric-Repro 的本地结果，以及下列论文原文和官方资源。历史文档有相互矛盾的版本，以下不继承其中未经验证的因果结论。已有综述的工作区修改保持不动。

**2026-09-09 师兄反馈后的优先级修订：先完成 [StereoGRU 实现审计](STEREOGRU_IMPLEMENTATION_AUDIT.md)，再决定结构路线。** 已证实双体 lookup 接线偏离、相机双归一化/无 focal 监督、裁剪主点表示不兼容，以及 SSI 与物理 index 的契约冲突。旧负结果不是对 StereoGRU 方法的有效否定。下文 RGB 旁路是候选设计，不因旧 costvol 失败而自动优先。

**2026-09-10 实测更新：两版 costvol 各 8×4 抽查帧均出现非有限 K、预测相机支持为 0、实际 raw 体为零；GT K 恢复约 95% 支持。** 当前执行优先级是校准几何/索引修复 → 小样本 → 完成 volume-only 基线 → GRU 对照，再决定 SOTA 结构路线。不是继续等待“方法是否不适合”的抽象解释，也不是先启动下文 RGB/flow arm。死 ReLU 还使“仅打开 focal loss”不成为充分修复，详见 [实测与修复顺序](STEREOGRU_IMPLEMENTATION_AUDIT.md)。

精确回传已确认首 clip 四帧 **fy=inf、GEV 全部严格 0**，六个执行源码 hash 与诊断快照吻合，取证结束。已新增[隔离的校准 volume-only pilot](config/stereogru/calibrated_volume_only.yaml)：clean frozen backbone、GT 相机、训练场景 2×4 帧、绝对 index 监督、200 步，无 GRU/no auto method。它是实现 gate，不是正式 baseline 完成或修复提点结果；详见 [入口与验收](STEREOGRU_IMPLEMENTATION_AUDIT.md)。

## 0. 决策摘要

**不再把“ConvNeXt-S + VKITTI 10k + 不断换 decoder”当作冲击准确率 SOTA 的主线。** 它仍适合便宜筛选，但不能替代强预训练模型、多域训练和长视频验证。

当前先完成上述 StereoGRU 修复验证。之后依据对照证据选择 SOTA 主线；下列先验/RGB 与蒸馏仍是候选设计，不是已经用旧负结果决定的转向：

1. **候选主线：完整预训练 GemDepth 的先验保持型残差增强。** 保留原 backbone、GEM、ASTT、DPT 及长视频拼接，增加小型 RGB 细节旁路；先验证空间旁路，再验证可靠的二维对应关系是否带来额外收益。保留强先验是一条待测假设，不用坏相机下的随机匹配头失败来证明其优越。
2. **扩展：有监督回放约束下的真实视频蒸馏。** 先验证长上下文教师确实优于短上下文学生，再做固定教师蒸馏；EMA、更多数据和动态区域处理均不在第一版同时加入。
3. **保留但限额：当前注册式 loss 矩阵。** 完成已经运行的 `full` baseline，按依赖关系筛选；其胜者还必须在主线的强底座、较长训练 clip 上复验。
4. **StereoGRU 先排错、不盲重跑：**冻结旧 checkpoint/配置，按官方 IGEV-MVS 核对 camera、raw/GEV lookup、index supervision，再做校准 oracle 和匹配对照。不将“暂停重复错误实现”误写成“放弃纯匹配/GRU 方法”。继续堆迭代数、未经校准的 warp 和完整重训 ViGeo/PXDepth 级模型仍不优先。

这是一条提高成功概率的路线，**不是必达 SOTA 的保证**。配方改进、方法增益和 SOTA 声明分别验收，不能互相替代。

## 1. 先固定要超过的任务和数字

### 1.1 主任务

- **RGB-only、离线、zero-shot relative video depth**；推理不输入 GT 深度、GT 内外参、双目右图或测试标签。
- 主 benchmark：Sintel 官方视频清单；Bonn / ScanNet / KITTI 的官方 500 帧清单。论文把 Sintel 标作 50 帧，本地 evaluator 上限是 100；须以清单实际帧数为准，不虚构 Sintel500。
- 主指标：**AbsRel、δ1**；同时保留已有 **RMSE**。分别报告四个数据集，不发明跨域加权总分。
- 对齐：**整条序列一组 inverse-depth scale + shift**，不是每帧独立拟合，也不是 depth-space affine。
- 训练中的几何/对应诊断、置信度统计、校准漂移只用于排错，不用作“新榜单”。

当前评测实现在 [evaluation/eval/eval.py](evaluation/eval/eval.py#L81-L270)。它已对缺预测和非有限值报错；后续接入外部模型也必须保留完整性检查，不能静默跳过失败序列。

协议基准固定为 GemDepth 官方代码 `652865b0ed20e727784a6b77314da1dca2f14e36`，交叉核对 VDA `4f5ae23172ba60fd7bc11ef671cca678842c7072`。开始主线训练前，补齐官方数据包 revision/hash、场景及逐帧路径、采样方式、GT 类型、mask/crop/depth cap、输出转换与聚合方式，并验证本地数值实现与官方实现的一致性。**该数据快照核验目前未完成**；如果最终无法对应，只能称“固定本地协议复测”，不能凭帧数相同就与论文横比。

### 1.2 经原表核实的参考线

以下是 **GemDepth v4 Table 1 的论文报告值，不是本次复跑结果**。列序尤其不能再写错。

| 方法 | Sintel：AbsRel / δ1 | Bonn500 | ScanNet500 | KITTI500 |
|---|---:|---:|---:|---:|
| VDA-L | .295 / .644 | .071 / .959 | .089 / .926 | .083 / .944 |
| GemDepth-DAv2 | .188 / .812 | .055 / .970 | .069 / .959 | .077 / .950 |
| GemDepth-VDA | **.157 / .827** | **.051 / .978** | **.066 / .967** | **.071 / .955** |

来源：[GemDepth v4](https://arxiv.org/html/2605.10525v4)。Table 6 的 Bonn Stage-2 与主表有差异，Table 9 还包含不同 pose-integration 版本；不要从不同表拼一个“最优 checkpoint”。

本地 Stage-1 历史长协议记录为 Sintel .3165 / Bonn .0695 / ScanNet .0769 / KITTI .0862。推理 dropout 修复后，目前记录明确补测的 KITTI 为 .086110；不能把另外三集旧值自动标成已重测。参见 [Stage-1 结果](assets/gemdepth_stage1_repro_dashboard.html#L103-L118)及[修复复测](assets/gemdepth_stage1_repro_dashboard.html#L278-L288)。

这说明：**从现有 Stage-1 的 Sintel .3165 追到 .157，需要约 50% 的相对误差下降。寄望一个小 loss 改动独自补齐这段差距，证据不足。** 优先接入完整公开预训练模型，而不是反复从弱起点重新训练。

### 1.3 还必须补哪些强参考

| 优先级 | 参考模型 | 目的与限制 |
|---|---|---|
| 必须 | 官方 VDA-L、官方 GemDepth | 用同一评测清单复跑；官方执行路径与 PP-DPT 加载路径做预测一致性验证 |
| 必须 | DA3-LARGE-1.1、资源允许时 GIANT-1.1 | 更新后的权重不能拿旧论文数字代替；不得输入 GT camera |
| 必须 | DVD 发布权重 | 与本任务接近的近期竞争者；v1.0/v1.1、300/500 帧不能混标 |
| 最终 SOTA 核查 | ViGeo、PPVD，以及届时可用的 ICDepth | 记录输出域、原生推理、数据暴露与完整成本；先确认可运行，不假装已有同协议数字 |

外部强模型做两类说明：**同一评测协议下的官方推荐推理结果**，以及有必要时的**匹配分辨率/上下文预算结果**。前者比较可达到的性能，后者比较预算；不要因官方原生输入不同而冒称完全等预算。

内部 baseline/method 则严格固定输入分辨率、clip 长度、窗口、重叠、拼接、精度和 future-frame 权限。模型输出转换后进入同一 evaluator，不能为某一 arm 改对齐域。

### 1.4 TAE 单独锁定协议

TAE 是已有论文指标，可以使用，但不能自写相似量后仍沿用其名字。GemDepth Table 2 报告 VDA .57 / GemDepth-VDA .47，描述为 ScanNet 前 20 条、每条 110 帧；当前 VDA 官方启动脚本是 `[10:180]`，长序列实际最多 170 帧。

先锁定场景清单、切片、深度/内参预处理、尺度处理及最终 ×100 的单位，再对所有模型统一复测。未解决论文/脚本差异前，**不拿 .47 作直接复现结论，也不把它称作 500 帧 TAE**。本仓目前没有 TAE evaluator。

## 2. 从过去实验真正能继承什么

| 已有经历 | 可继承的证据 | 不能推出的结论 |
|---|---|---|
| 同配方混训 temporal vs multiscale | temporal 在四集中的三集 AbsRel 更好；足以降低继续堆 decoder 的优先级 | 所有多尺度结构都无效，或差距纯由 backbone 决定 |
| PR#8 后的 coarse supervision | `[1,1,1,1]` vs `[0,0,0,1]` 的 KITTI .1004 vs .1016，只是单 seed 微小信号 | “粗尺度无贡献”；loss ×4 等于 AdamW LR ×4；head 参数 ×4 |
| errmap / cost-volume 失败 | [实测审计](STEREOGRU_IMPLEMENTATION_AUDIT.md)确认 costvol 两版抽查帧 K 非有限、空体；相机/双体查表/index 契约需重建。errmap 未包含在这次实测中 | 把该失败当有效 StereoGRU 对照；泛化为历史所有训练步/errmap 都为空体；只打开 focal 权重就能解决 |
| Stage-2-lite | 该次继续训练方案没有显示可靠的明显收益，且不是论文 Stage-2 数据配方 | 严格证明 Stage-2 收益全部来自真实数据，或冻结 GEM 永远无效 |
| DenseGRU 原始结果 | 同一 652 样本文件记录 calibrated D0 .2757 → final .0804；尺度约定和强初值非常重要 | 这是 RGB-only zero-shot SOTA，或已隔离出“时序/GRU”的因果贡献 |
| DA3-Metric-Repro | KITTI-only 适配曾伴随 NYU 大幅退化；跨域回归必须监测，实际输入焦距必须同步 | KITTI 定域 fine-tune 超过零样本论文数字，就等于全面超过基础模型 |

多尺度参数和监督修正见 [已核对看板](assets/vkitti_decoder_arms_dashboard.html#L139-L155)。DenseGRU 是 **已知 GT pose 的另一任务**，其评测样本不全、checkpoint 选择与单帧可训练对照仍需审计；这里仅继承工程经验，不搬分数进入视频相对深度主表。

还要记住：早期被称为“DA3 / VDA-V3”的部分代码实际使用 VDA。后续每个 run 必须记录模型类、权重来源/hash、匹配键和输出语义，不能按目录名称认模型。

## 3. 当前方案的具体问题：先修事实，再谈增益

### 3.1 不是“再补一个 per-frame SSI”

[VideoDepthLoss](loss/videoloss.py#L584-L603)已经把 `B×T` 展平，逐帧 robust-normalize 后计算 SSI / spatial gradient；temporal 项另做 clip-shared affine fit。新增 [joint objective](loss/objective_joint_align.py#L20-L51)补的是**clip 对齐后的回归项**。

另外，当前 `sequence_weight=1` 不能直接理解为与 SSI 等强：sequence 项是目标 inverse-depth 单位下的 L1，原 frame SSI 是 robust-normalized 空间。跨域时近景、有效像素数和尺度会改变实际权重。先记录分域 loss / gradient norm；如需按 clip 的 GT inverse-depth 尺度归一化，应注册为**另一个 objective**，与原 joint 单独对照，不能静默改掉已有实验。

CARVE 的固定 inverse-GT-depth weighting、多任务 geometry consistency 与这里的 inverse-depth SSI 不是同一个东西。因此这些配置是“受消融启发的实验”，不是 CARVE 复现。

### 3.2 内参覆盖值得查，但现有解释错了

- [训练调用](train.py#L500-L508)是 `model(image)`；GT `IntM` 进入 loss，**不是直接输入深度 forward**。
- [数据管线](dataset/dataset_mix.py#L333-L395)只保留首帧 `K_clip`，并用 `h_new/H` 同时缩放 `fx/fy/cx/cy`。对于逐帧变内参、或 multiple-of rounding 造成的横纵缩放不完全相等，有潜在错误。
- 当前实际调用的是 resize + square crop；定义了 `RandomScale` / flip 不代表它们参与了 `_getitem_inner` 的增广。
- 原图的 `fx/W` 不能代表训练输入。以 1242×375、fx=725 的 VKITTI 为例，短边缩放后裁成正方形，输出的 `fx/W≈725/375≈1.93`，**不是原始的 .58**。这只是由代码推出的例子，完整分布仍需实测。

正确更新应逐帧使用 `sx=W'/W`、`sy=H'/H`，再减裁剪偏移；投影验证同时覆盖变焦、非中心裁剪和非等比 rounding。

整图等比 resize 不改变 FOV。不同大小的 crop 可以改变归一化焦距，但裁剪不能凭空生成原图之外的更宽视场。clip 内 RGB / depth / mask / K / flow 必须使用相同空间变换。

**FoundationGeo 证明的是其米制几何设定下的焦距覆盖收益，不证明 Sintel 误差有“一半”由焦距造成。** 先修相机标签契约，再做独立 FOV 实验；不要与 uniform sampling、多分辨率同时改而分别宣称贡献。

### 3.3 数据有效性比“凑够几个数据集”重要

- [PointOdyssey loader](dataset/loaders/pointodyssey.py#L34-L73)递归找序列，不自动限制官方 train split，也未显式筛选论文要求的背景深度子集。
- [Dynamic Replica loader](dataset/loaders/dynamic_replica.py#L46-L71)会递归读取给定根目录中的 annotation；根路径包含哪些 split，就可能吃到哪些 split。
- [通用窗口函数](dataset/loaders/base.py#L42-L54)的 train 覆盖所有窗口，val 取尾部；这不是独立 held-out。**不能把这个问题误扣到已经启用 scene split 的当前 VKITTI-only baseline 上。**
- [训练 mask](train.py#L491-L507)按 depth range 重建，没有合并 loader mask。现有 loader 大多先把无效深度置零，但新伪标签 confidence / sky / background-valid mask 不能沿用这种假设。
- [GPU smoke 原日志](jobs/newdata_gpu_smoke.12655888.stdout#L23-L56)确实验证了六源 finite 前后向；它不证明 split、背景 GT、pose 精度或零样本泛化正确，也不代表新增 IRS 已可用。

新协议先冻结 scene/trajectory manifest，计数实际有效样本、抽样概率和跳过样本；开发集按场景留出，不按相邻帧切。官方四个测试集不作反复调权重的验证集。

### 3.4 几何一致性不能绕过 gauge

GEM 的 translation 先经过 scene normalization，又在 camera loss 内按最大相对平移归一化，最终约为 clip-baseline 单位，**不是只除平均场景深度**。cost-volume 用固定米制 bins、其他深度头输出 affine-ambiguous inverse depth；直接与该平移刚性投影，没有共同尺度保证。旧诊断仅乘 scene scale 的还原也已修正，详见 [STEREOGRU_IMPLEMENTATION_AUDIT.md](STEREOGRU_IMPLEMENTATION_AUDIT.md)。

只有先校验相机、深度和坐标系，再通过 [camera 诊断](scripts/diagnose_gem_camera.py)及[cost-volume oracle](scripts/diagnose_cost_volume_oracle.py)，才能继续花训练预算。这里的 oracle 指标只是 gate，不替代 benchmark。

CARVE 的 depth / camera / independent pointmap 一致性需要独立预测路径；同一深度反投影后再与自身比较，是恒等式。也不能用“pose loss 与 depth loss 同时差”当作 PAGE 式梯度冲突的证据。

## 4. 主线方法：保留完整先验，只学习有证据的修正

### 4.1 起点必须强且可复现

首选官方完整 GemDepth checkpoint，先在官方执行路径复测，再验证 PP-DPT 加载后的输出。若不能匹配，保留官方版本作外部参照，先解决加载/推理差异，不能用 `strict=False` 吞掉问题。

本地 Stage-1 checkpoint 是备选的开发起点，不伪称完整两阶段强基线。VDA-L 可作为第二底座检验方法通用性，但必须单独完成其 baseline。

**第一轮冻结整个已有预测器，不止 encoder。** 避免把已有 DPT/GEM/ASTT 重置，也避免新分支训练把空间先验一起拉坏。冻结不等于 eval：训练时旧预测器保持 eval，新分支才 train，防止 pose dropout 再次改变教师/初始预测。

### 4.2 最小结构

第一版只依赖原预测器已经返回的每帧 inverse depth `p0` 和 RGB。**当前 forward 不暴露 DPT 中间特征，不能假装现有 decoder registry 已解决这一点。** 旁路自行编码上下文；内部特征 tap 留作以后独立扩展，不是第一轮前置条件。

1. **RGB 细节流**：原输入分辨率的 16–32 通道轻量卷积，保留真实像素信号；不是将低分辨率特征插值后冒称“高分辨率分支”。
2. **空间残差**：融合 RGB 与规范化 `p0` 的旁路特征，先做逐帧修正。注意 `p0` 本身已经含时序信息，这不是“完全单帧模型”。
3. **可选时序残差**：用冻结光流产生 `t±1` 的对应位置，在约 1/4 分辨率对齐特征；融合 target、aligned source、差异和有效性，不做全分辨率全局 attention。
4. **可靠性门控**：越界/遮挡/forward–backward 不一致时关闭时序修正，退回空间分支；不把坏 flow 传成 NaN。训练仍对所有有效 GT 像素监督，不能靠 mask 掉困难区域“提分”。

为适应不同 checkpoint 的 inverse-depth gauge，可在固定的有限预测支持域上，以同一 clip 的 `p0` 计算 detached 中位中心 `μC` 和绝对偏差尺度 `sC`，旁路只读 `(p0−μC)/sC`，不另外混入未经规范化的 `p0`：

$$
p_t=p_{0,t}+s_C\left(r_{\mathrm{rgb},t}+c_t r_{\mathrm{temp},t}\right).
$$

残差分支读入规范化的 `p0`；两个输出投影均零初始化，**新模型初始输出须等于原模型输出**，再经过双方相同的后处理。`c` 不要与残差末层同时零初始化，以免乘积使梯度全断。

修正位置固定在原模型输出一个窗口的全分辨率 `p0` 之后、原有窗口间仿射拼接之前；不新增会改变零残差输出的 ReLU/数值下界。必须检验原始窗口输出及完整长序列的零残差一致性。

在支持域不变、`sC>0`、没有 clipping 的条件下，该参数化可对 `p0'=a·p0+b, a>0` 保持仿射等变；这不是绝对米制恢复。尺度退化或非有限时返回原预测，记录触发率。原有 clipping、不同窗口的统计量和拼接会破坏理想等变，须用同一帧出现在不同窗口位置的诊断验证，不能从公式宣称已经解决长视频漂移。

这只是结构设计假设；零初始化保证起点不被破坏，**不保证训练后必然不退化**。低通道分支也不自动意味着低延迟，实测全分辨率带宽和 flow 成本。

### 4.3 为什么先选二维对应，而不是重新押 GEM rigid warp

- 二维 flow 对齐特征不需要把 relative depth 换算成米，也不依赖 GT pose。
- 它能表达部分动态物体运动，减轻纯刚性场景假设的限制；遮挡、出画、弱纹理和 flow 错误仍然是失败条件。
- **禁止强制对应点的两帧深度相等**：相机/物体运动会改变 camera-z。第一版保留现有监督损失，只改特征融合；若以后增加 warp loss，应匹配预测与 GT 的深度变化，而非把它们硬压成常数。
- 第一版一次融合、一次输出，不加 GRU。只有校正量本身在标准指标上有收益，才考虑迭代。

光流先选来源清楚、未在目标 benchmark 图像上微调的权重。例如 torchvision RAFT `C_T_V2` 只声明 FlyingChairs + FlyingThings3D 训练；**不能直接选 DEFAULT**，它对应包含 Sintel/KITTI 微调数据的 `C_T_SKHT_V2`。同样要审计 CoTracker / SEA-RAFT / teacher 的训练数据，不能让旁路引入目标集泄漏。

### 4.4 主线最小对照矩阵

公共条件：同一完整预训练预测器、相同冻结策略/初始化、固定训练与开发 manifest、loss、seed、有效训练帧数、优化步数、crop、clip、推理窗口及 evaluator。

| ID | 分支 | 与谁比较 | 唯一待回答问题 |
|---|---|---|---|
| E0 | 原始完整 GemDepth，不训练 | 官方结果复测 | 强起点是否成立；它不是训练预算匹配的 method baseline |
| B0 | 冻结 E0 + 深度上下文空间残差 | E0，同时单列训练适配收益 | 便宜的普通 refiner 已经能改善多少？ |
| B1 | B0 + 原分辨率 RGB 细节输入 | B0 | 新像素信息是否有用，而不是多加参数？ |
| T0 | B1 + 未对齐的邻帧融合 | B1 | 普通时序融合的基线有多强？ |
| T1 | 同一融合器，改用 flow 对齐 | T0 | 改善是否来自正确对应关系？ |
| T2（条件项） | T1 + 可靠性门控 | T1 | 是否能减少遮挡/错误对应造成的退化？ |

**依赖顺序：E0 → B0 → B1 → T0 → T1 → T2。每个直接 baseline 完成后，才启动对应 method。** B1/T0 没有开发集信号时，不把所有 arm 都跑满。

RGB 支路失败只停止该支路，**不等于 flow 无效**；若独立对应诊断通过，可以在 B0 上另建匹配的 T0/T1，不能借用 B1 结果当直接 baseline。

实现容量控制：B0/B1 使用同样的 stem 和输出结构，B0 用重复的规范化深度替代 RGB 通道。这匹配名义参数/计算，但不保证有效容量相同；若 B1 有收益，补同预算的 depth-only 容量对照。T0/T1 使用相同的冻结 flow、源特征、数值有效性 mask 与可训练融合参数，只改变采样坐标；学习型置信门控属于后续 T2，不能提前混入 T1。两者都计算 flow 的受控实验用于匹配成本；另报告删去无用计算后的部署成本，避免装作 flow 免费。

必要诊断包括：固定源特征只扰动对应坐标，以及另做错配/打乱邻帧；二者区分坐标有效性与邻帧语义。若均不影响结果，模型可能没有利用对应关系。逐帧 residual 的提升也必须优于同预算的普通适配，而不是只和未训练 E0 比。

本主线要检验的具体机制是：**共享 clip gauge 的修正参数化，加上对应可靠性选择，是否比普通逐帧/未对齐时序 refiner 更能在长序列中纠错而不破坏原先验。** 这不是新颖性已被确认的结论；若只有普通 RGB 适配收益，就按配方改进报告，不能靠给现有模块改名形成方法贡献。

## 5. 第二阶段：真实视频蒸馏，不直接照搬“+36.5%”

### 5.1 先验证教师优势

SelfEvo 的 36.5% 是其 KITTI scale-only AbsRel `.074→.047`，不是四集平均，也不是当前 inverse-depth-affine500 协议。

先在独立有 GT 的开发场景，用同一 checkpoint 比较：教师 32 帧上下文 vs 学生 8/16 帧上下文，**只对共同的目标帧打分，使用相同 GT 支持域和标准指标**。不能拿更短序列更容易拟合的指标当教师优势。

本模型的时间位置表示默认围绕 32 帧构造；不未经检查就改到 64/128。若 32 帧并未优于 16 帧，停止“更多上下文必然更强”的假设，改选经同协议验证的外部固定教师，或不做这一支。

为避免训练/部署目标漂移，本扩展的第一轮**学生固定训练与推理 16 帧、教师 32 帧**，先完成新的 16 帧 baseline；它与主线 32 帧结果分开标识。共同目标帧的绝对时间 ID、位置编码、边界与 crop 要固定或做独立位置对照；不能把索引变化全归因于上下文。若以后恢复学生 32 帧推理，需新 baseline 和新的教师优势验证，不能自动转移 32→16 的结果。

### 5.2 数据和训练建议

- 先做 10k–20k 帧的小型合法公开训练子集，质量通过后才扩到 50k–100k；不一开始搭建百万帧数据引擎。
- 候选池：RealEstate10K train 的室内、BDD100K train 的室外，按许可/可获得性选取；动态补充可考虑 DROID 或 DAVIS train。**这些不是已在本机就绪的数据声明。** 场景/轨迹去重和所有目标评测清单隔离是前置条件。
- 首轮固定为有监督 synthetic replay + 小比例 real pseudo（例如按有效帧 75%/25%，只是预注册起点，不声称最优）；不可把可变长度 clip 的数量当帧比例。
- 先用固定教师离线缓存，再考虑 EMA。缓存记录 teacher hash、输入帧、分辨率、输出域和 mask，计入生成成本。
- 伪标签保留 inverse-depth 语义，不伪装成米制 GT 后套 80m cap；无 pose 样本显式标记，不拿 identity pose 充当真监督。
- 置信筛选须先在开发 GT 上验证与误差的关系；教师相互一致并不保证都正确。过滤不能对动态对象一概删除。

### 5.3 把数据增益与蒸馏机制分开

| 对照 | 共同条件 | 回答的问题 |
|---|---|---|
| 合成回放继续训练 vs 加 real pseudo 的普通固定教师版本 | 同一起点和优化预算；数据变化明确标出 | 增加该伪标签数据是否有用？这是 recipe 比较，不是纯架构比较 |
| 普通教师版本 vs 长上下文教师版本 | **相同真实视频、相同学生看到的帧、相同监督帧和 loss** | 额外教师上下文是否带来有效监督？同时报告额外 teacher FLOPs |
| 固定长上下文教师 vs EMA 教师 | 数据、上下文、更新数、student 初始化相同 | 在线教师更新是否必要？ |

所有蒸馏在对应时间索引和空间变换下比较。无 GT 的 depth 蒸馏可在共同预测有效域中，对整个 student clip 拟合一组 student→teacher affine 参数，再用 teacher 的 detached clip 尺度归一化 L1；teacher、支持域和置信度均停止梯度，退化拟合必须有保护。这里的对齐目标是 teacher，不是测试 GT，也不是逐帧独立拟合；教师错误仍会被继承，所以保留有监督回放和开发集回归检查。开发集发现漂移时回滚/停止，不靠测试集标签筛 teacher。

75% 合成 + 25% 真实会减少合成曝光，因此只能回答该混合 recipe 是否有效；若要隔离“纯增加真实样本”的作用，另做保持合成曝光的比较，并公开增加的步数/算力。等更新数、等曝光量和等计算量不是同一公平口径。

## 6. 当前六个配置怎样处理

最后收到的日志只证明 `vkitti_temporal_carve_full` 已运行超过 step 2800/10000，**没有收到完成和评测证据**。

| 顺序 | 已有配置 | 解释边界 |
|---|---|---|
| 1 | [full](config/vkitti/vkitti_temporal_carve_full.yaml) | 先跑完并评测，保留正在产生的基线价值 |
| 2 | [no_sg](config/vkitti/vkitti_temporal_carve_no_sg.yaml)、[no_tg](config/vkitti/vkitti_temporal_carve_no_tg.yaml) | 分别相对 full 删除一个项；不因 CARVE 就预设胜负 |
| 3 | [simple](config/vkitti/vkitti_temporal_carve_simple.yaml) | 两项同时删除的交互对照，不等于另一个单变量结果 |
| 4 | [joint](config/vkitti/vkitti_temporal_carve_joint.yaml) | 必须与 simple 比；目前不能隔离“full 上加 sequence 项”的作用 |
| 条件项 | [pose_aux](config/vkitti/vkitti_temporal_pose_aux.yaml) | 与 full 比的是新增 camera 辅助任务，包含 FoV；不是 pose features 注入 ASTT |

不再给这条筛选线追加大规模头结构矩阵。若 pose_aux 有收益，再拆 R/T 与 FoV；移除 GEM 部署前必须做输出一致性验证。不同模型族/训练长度的 loss 胜者不能直接迁移为已证实最优。

## 7. 接口与工程约束

已经有 [decoder registry](model/decoder_registry.py)、[objective registry](loss/objective_registry.py)、[freeze policies](model/freeze_policies.py)和[mix policies](dataset/mix_policies.py)。新增实验应通过注册模块和配置表达，**不在核心训练循环堆 `if experiment == ...`**。

但“loss 已注册”不等于教师训练、可变分辨率数据、RGB/flow payload 已全部可插拔。实施前须明确一次性的通用输入/输出契约和可选监督 mask；不能为了赶实验把 teacher forward 偷塞进 loss，或把真实数据强伪装成现有 metric-depth loader。接口不足先独立设计通用扩展，保留现有训练入口行为。

第一版建议采用**新增的注册式 refiner-stage 入口/包装器**，消费现有预测器的 `p0`，不改核心训练循环、不伪报 decoder 的几何依赖来索取 RGB。它统一管理冻结 predictor 的 eval 生命周期、可训练 residual、数据 mask 和公共优化步骤；后续 arm 只新增注册模块/config。任何需要改既有通用接口的部分单独审阅并做行为一致性测试，不宣称“加一个 decoder 文件即可完成教师训练”。

实现验收至少包含：

- checkpoint 共享键完整加载，新增键显式声明；零残差输出与旧模型一致；分支关闭也一致。
- 新分支有梯度、冻结先验无梯度；旧预测器在训练过程仍保持 eval；参数分组覆盖且不重复。
- 全无对应/全遮挡/全无效 GT、NaN flow、T=1、变长 clip 均不产生伪监督或非有限值。
- 相机逐帧 K、resize/crop 后投影、flow 坐标同步、mask 不被 range mask 覆盖。
- baseline/method 同一 clip 的原始输出和完整评测结果保存；不只保存挑选过的可视化。

**原 SOTA 方案仍未启动方法训练。2026-09-10 仅新增隔离的校准 volume-only pilot 入口（见上文），不修改旧训练链路或旧 checkpoint、不覆盖已有综述修改。真实数据 pilot 尚待用户执行，不能标成完成的正式 baseline。**

## 8. 执行顺序、预算与止损

### 第一轮顺序

1. 核实 full 的完成/评测状态（当前没有新证据）；StereoGRU 首轮取证已确认非有限 K/空体，接下来是校准索引小样本与 volume-only → GRU 匹配验证，不重复错误配置或先转向 RGB/flow。
2. 固定开发/测试清单；复跑官方 VDA、GemDepth，确认可用的强起点和本地预测一致性。
3. 把相机标签、split/mask 等必要修正冻结为新协议；**新协议的 baseline 先完整跑完**。旧结果保留但不冒充新对照。
4. 主线先做 B0/B1。RGB 分支不优于 B0，停止扩展像素支路，不搬用 PXDepth 结论硬解释。
5. 在通过的空间 baseline 上做 T0/T1；T1 有信号再做 T2。RGB 未通过但对应诊断通过时，可在 B0 上开这一对。匹配 oracle 不工作或扰动对应也不影响结果，先回到对应质量，不追加 GRU。
6. 再做教师优势检查和小规模 real pseudo 对照。冻结最终选择后，完成三组 paired seeds 和全部标准测试。

### 预算口径

- 先实测 100–200 个更新的显存/吞吐，再确定每个 arm 的固定预算。冻结 backbone 不代表 teacher/flow/全分辨率分支没有成本。
- 建议 pilot 2k steps、筛选 5k steps；每一级有自己的同预算 baseline。不得把 method5k 与 baseline20k 当同预算实验，也不得用长 run 的中间 checkpoint 冒充相同 schedule 的独立短 run。
- 第一轮先批准**协议复测 + B0/B1 两个完整 arm**的预算；有信号才解锁下一对。以实际 H20 GPU-hours 记账，不拿 H100/H200 卡数换算承诺耗时。
- 尚无用户指定总预算时，本计划建议首阶段**512 H20 GPUh 上限**（等价于八卡合计占用 64 小时，不是承诺完工时间，也不是作业授权）；训练/teacher/flow 总计最多约 342 GPUh，至少约 170 GPUh 留给协议复测、重复种子及回归。所有 pilot、伪标签生成和失败 run 均入账；profile 预计会超支就缩小候选数/暂停，而不是省略直接 baseline。后续阶段需要另定预算。
- 历史量级仅供参照：Stage-1 20k 在 8×H20 约 37h55m（约 303 GPUh）；单卡四集长协议约 4h，ScanNet 最慢。新方法耗时必须重新 profile。
- 保留至少约三分之一的研究预算给强参考复测、多 seed 和最终全测试；不把预算全部花在单 seed 架构搜索。

### 预注册 go/no-go，而不是新指标

建议在**开发集**预注册这样的资源门槛：至少两个开发域 AbsRel 相对降低约 2%，其他域不能出现超出测量噪声的明显退化；δ1、RMSE 不能持续反向恶化。这些是预算决策阈值，不是“自定义评分”。具体容忍度应以新 baseline 的重复实验估计，而不是照搬旧 E2a 的 .0006 跨模型当万能噪声带。

最终用至少三组 paired seeds，报告每个数据集的均值/标准差与按**序列**重采样的差值区间。像素和相邻帧不是独立样本，不能用百万像素制造虚假的显著性。

如果只有 TAE 改善、AbsRel / δ1 退化，应如实标注为一致性—准确率取舍，不能回答为“准确率超 SOTA”。如果只在某一数据集赢，就只声明那一数据集。

### 数值目标：目标，不是预测

相对已核实 GemDepth 论文主表，约 5% AbsRel 降低对应：

| Sintel | Bonn500 | ScanNet500 | KITTI500 |
|---:|---:|---:|---:|
| ≤ .149 | ≤ .0485 | ≤ .0627 | ≤ .0675 |

这些只作规划标尺。**真正的超越线是最终同协议复测表里更强的有效参考，而不是永远固定在这四个旧数字。** 还需 δ1 / RMSE / TAE 的完整结果，以及推理时间、显存、额外预训练/teacher 成本。不能靠换对齐、减少帧数、丢失败序列、用测试集 adaptation 或 GT camera 越线。

## 9. 文献如何改变方案，而不是只提供名称

| 工作 | 已核实的可借鉴内容 | 本计划的取舍 |
|---|---|---|
| [CARVE v1](https://arxiv.org/html/2604.21713v1) | 梯度项/固定权重/对齐策略的受控消融；同数据 finetuned baseline 很重要 | 保留 loss 筛选；不照搬其 scale-only geometry 数字或 independent pointmap loss |
| [SelfEvo v1](https://arxiv.org/html/2604.08532v1) / [官方代码](https://github.com/Self-Evo/SelfEvo) | 更丰富上下文的教师监督帧子集；EMA；主要训练数据也包括合成视频 | 教师优势先过 gate，先 fixed teacher，再 EMA；不是必然的真实域收益 |
| [FoundationGeo v3](https://arxiv.org/html/2607.11588v3) | 米制几何中的实际相机覆盖、预测 scale/ray field | 先修 K 和审计实际 FOV，不把 scale field 与 GT 局部仿射 loss 当成相反结论 |
| [PXDepth v2](https://arxiv.org/html/2608.16984v2) / [官方资源](https://github.com/yuanzhy29/PXDepth) | 保留真实 RGB 像素流、用强全局先验条件化；训练代码尚未完整公开 | 借鉴旁路，不复刻 700k-step 全流程；图像边界优势不能代替视频准确率 |
| [Tracktention v1](https://arxiv.org/html/2503.19904v1) | 冻结基础深度模型、轨迹引导残差融合 | 借鉴对应关系与保留先验；官方实现尚未可直接复现，先用成熟 flow 做最小实验 |
| [VeloDepth v1](https://arxiv.org/html/2512.10725v1) | 特征传播、关键帧刷新，有准确率—速度取舍 | 目前目标是准确率，不将快速传播当主线 |
| [PAGE-4D v8](https://arxiv.org/html/2510.17568v8) | 动态区域对 pose queries 与 geometry queries 的不同作用 | 第一轮冻结旧几何分支；只有实际共享梯度冲突证据后才做 attention masking |
| [DVD v1](https://arxiv.org/html/2603.12250v1) / [官方代码](https://github.com/EnVision-Research/DVD) | 单步生成先验回归，图像/视频共同训练 | 新强对照；学习保护空间先验，不把 latent temporal differences 误叫 optical flow |
| [ViGeo v3](https://arxiv.org/html/2605.30060v3) | 数据精化和视频几何；论文训练规模远超当前主线预算 | 用公开权重比较/探索 teacher，暂不全流程复训 |
| [ICDepth v1](https://arxiv.org/html/2607.01677v1) / [PPD v2](https://arxiv.org/html/2510.07316v2) | 干净 RGB 条件与像素空间结构保留 | 背景依据和候选参考；不是可直接搬用的 affine500 SOTA 数字 |

辅助协议来源：[VDA 官方评测](https://github.com/DepthAnything/Video-Depth-Anything/tree/4f5ae23172ba60fd7bc11ef671cca678842c7072/benchmark/eval)、[DA3 官方发布](https://github.com/ByteDance-Seed/Depth-Anything-3)、[RAFT 权重训练来源](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.optical_flow.raft_large.html)。

截至此次核查，没有从这些论文中得到“所有最新公开模型已经统一在 affine500 下复测”的完整榜单。因此这里明确区分论文参考、资源可用和本地实测，不将摘要的 SOTA 声明当事实。

## 10. GitHub 范围与交付边界

`fank53` 的公开用户页/API 返回 404，GitHub 用户/仓库搜索无匹配。**不能声称已经审阅该账号。** 需要完整 URL 或可访问仓库内容才能补充。

已另行核对当前工程对应的 [FankChen 公开账号](https://github.com/FankChen)，并读取工作区内 DenseGRU-v1.0、DA3-Metric-Repro 的原始结果和文档；这不是擅自认定 `fank53=FankChen`。

本方案优先回答的是：**先站到可验证的强起点上，再证明新信息能带来可重复的标准指标收益。** raw-RGB、flow、EMA 本身都不是新发明；如果最终只得到普通配方收益，就按配方收益报告。新方法贡献必须来自干净的同底座对照、失效机制分析和跨数据集/底座复验。