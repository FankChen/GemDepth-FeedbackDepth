# StereoGRU：depth 轴信息的匹配训练对照

日期：2026-09-10。状态：**方案已固定，尚未实现/启动此训练对照；不是新结果。** 本轮只归档只读检查，不继续堆同类诊断或自动开启 GRU。

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
| 训练样本 | 预先固定 32 个 clip，Scene01/02 各 16 个，每段连续 4 帧；同场景起点至少间隔 8 帧，禁止帧重叠 |
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
| 成本 | 单 H20 顺序运行，每 arm 预设最多 1 GPUh 的安全上限（提议，不是已授权作业）；超限停止，不能以半程结果对比另一组全程 |

训练 scene 覆盖、间隔和数量需要在生成 manifest 时计数验证；**目前没有宣称数据已满足这些数量**。1000 步/32 clips 仍是定额开发机制实验，不叫完整多域训练，不从先前 9.68s 的缓存两 clip pilot 推断真实吞吐。

本次不同时更换深度范围、相关归一化、hourglass 容量、RGB stem、teacher 或 loss，以免发现增益后仍不知道是哪一项生效。

## 4. 实现边界与验收

- C0 应是新增注册类，例如 `DPTHeadCalibratedFlatVolumeConvNeXt`，只实现体到 depth-mean 体的转换；C1 复用 [calibrated head](model/dpt_calibrated_volume_only_convnext.py)。差异通过新 config/注册模块表达，**不修改旧 head、旧 pilot 或核心训练文件来堆实验分支**。
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

**本轮 readout 到此收束。** 已获得足够信息决定有意义的下一种实验，不再要求用户反复运行同类 zero/flat/reverse 诊断；本文件只是这一步的可执行规格，尚无新训练任务、结果或自动启动指令。