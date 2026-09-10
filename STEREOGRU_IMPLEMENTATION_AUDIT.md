# StereoGRU：实现审计、历史结果重判与下一步取证

初审：2026-09-09。真实 checkpoint 回传更新：2026-09-10。审计基线：`feat/registry-mixdata@eb2c9ac`；首次阿里云诊断快照：`b8e2dd3`。

**修正此前判断：先证明实现和几何/监督契约成立，再评价 StereoGRU；当前负结果不能否定这类方法。** 多篇论文验证了校准条件下的迭代匹配，但也不保证任意视频移植自动有效。本次不重训、不替换历史 head、不改变历史训练配置。

**已收到实测：两版 costvol 各 8 clips × 4 frames 的预测 K 全部包含非有限项，预测 K 的三种模式均无有效支持，实际 raw volume 为零。换 GT K 后支持恢复到约 95%。** 这已经定位了抽查推理路径的致命几何输入故障，不再只是待验证假设；完整数值和结论边界见第 8 节。不能据此声称历史每一步训练均为空体，或全部误差都由单一问题解释。

## 1. 究竟在审计哪个实验

主对象是 [DPTHeadCostVolumeConvNeXt](model/dpt_cost_volume_convnext.py) 及 [cost_volume primitives](model/util/cost_volume.py)，对应 [costvol](config/vkitti/vkitti_costvol.yaml) / [costvol_d3](config/vkitti/vkitti_costvol_d3.yaml)。这是 plane-sweep + geometry volume + 三层 ConvGRU 的视频移植，不是一个名叫 `StereoGRU` 的类。

另一个 [多尺度迭代头](model/dpt_multiscale_iter_convnext.py) 没有 cost-volume lookup；它的 6600/10000 步中断结果不能混进本结论。DenseGRU-v1.0 使用 GT pose 的结果属于另一输入协议，也不是该 head 的直接对照。

已重新核对 2026-08-28 用户提供的原始终端输出，而不只看历史总结：

| run | 初值 loss（末次记录） | 最后 GRU loss | VKITTI AbsRel / RMSE / δ1 | 证据边界 |
|---|---:|---:|---:|---|
| costvol | 2.7081（step 9800） | .5220 | .4297 / 17.8809 / .5207 | 268 帧、**1 条序列** |
| costvol_d3 | 2.7281 | .5262 | .4171 / 17.4013 / .5233 | 同为 268 帧、1 条序列 |
| ms_gem | — | — | .1134 / 6.754 / .8657 | 相机辅助任务的相关参照；不是同容量/同索引监督的 GRU 因果对照 |
| temporal | — | — | .1192 / 7.257 / .8579 | 无 GEM 的相关参照 |

**`.0981` 是 temporal 的 KITTI zero-shot 数字，不是该 VKITTI 对照的 `.1192`。** costvol 两版没有已提供的 KITTI 结果。

[旧 VKITTI evaluator](evaluation/inference/eval_vkitti_dense.py) 对每帧单独 inverse-depth affine fit 后汇总；不是计划中整条序列一组 affine 的四集主协议。原输出只有 1 个 held-out scene，需核对实际数据覆盖，不得写成完整两场景/标准四集结果。

原先“depth_min 调整无效，所以深度采样已排除”“全部改善来自 GRU”“亚像素就是根因”“结构没有空间”均撤回。这里最多说明：**只改近端范围没有消除严重退化**；其余阻塞同时存在，不能据此排除采样与它们的交互。`.0126` 的差值也不能借用另一模型的 `.0006` 噪声带叫作“噪声内”。

## 2. 对照官方源码：不是只有 views/stride/bin 数改变

核对 [IGEV-MVS forward](https://github.com/gangweiX/IGEV/blob/e02312b8f9615c52346bd04c08f8d219c64c404d/IGEV-MVS/core/igev_mvs.py)、[lookup](https://github.com/gangweiX/IGEV/blob/e02312b8f9615c52346bd04c08f8d219c64c404d/IGEV-MVS/core/corr.py)、[update](https://github.com/gangweiX/IGEV/blob/e02312b8f9615c52346bd04c08f8d219c64c404d/IGEV-MVS/core/update.py) 和 [training loss](https://github.com/gangweiX/IGEV/blob/e02312b8f9615c52346bd04c08f8d219c64c404d/IGEV-MVS/train_mvs.py)。

| 环节 | 官方 IGEV-MVS | 当前移植 | 判断 |
|---|---|---|---|
| GRU 读取的两路体 | regularised geometry + **raw correlation** | logits + **softmax(同一 logits)** | **明确接线偏离**：raw matching 的直接查表路径丢失 |
| 平移与深度采样 | 已校准、同单位的 projection/depth bounds | 预测的归一化平移 + 固定米制 .5–80 / 3–80 | **契约不一致**，没有尺度恢复模块 |
| intrinsics | 输入已知 K | GEM 的焦距通道无有效目标监督；主点固定中心 | **几何输入未被正确约束** |
| depth-index supervision | 同一 bounds 下的绝对 normalized inverse-depth L1 | 逐帧 SSI + 对齐后的 temporal loss | **索引坐标没有被锚定** |
| 初值权重 | 独立权重 1；迭代部分用 adjusted gamma | 初值和迭代一起做归一化 gamma=.9 平均 | 当前初值约 .0703，末轮 .1632；不是官方 loss |
| RGB / matching / upsample | RGB stems、可训练 feature 网络、图像引导上采样 | backbone 基础权重冻结、LoRA + matcher；无 RGB stem，hidden-only mask | 设计差异，不可再称“唯一改变 views” |
| volume / aggregation | 64 hypotheses，G=1，stride4，特定 3D hourglass | 32 hypotheses，G=8，stride8，精简 hourglass | 非等价变体；不自动是 bug，也不保证无影响 |

**不能直接给旧 checkpoint 改接 raw correlation 后拿即时分数当修复收益。** 已训练 motion encoder 会接收到不同分布；修复须注册独立版本、保留原版并按新契约训练匹配对照。

## 3. 优先级最高的五个问题

### P0-A：焦距输出缺乏监督，非有限值被变成“安全的空体”

[旧配置继承](config/vkitti/vkitti_ms_gem.yaml)中 `use_astt=false`、`video_path=null`，相机模块随机初始化；[camera focal weight](loss/videoloss.py) 默认为 0。[cost-volume warp](model/dpt_cost_volume_convnext.py) 又 detach K/T，因此深度 loss 也不能校正相机输出。

[CameraHead](model/tools/camera.py) 下一轮 pose 输入 detach；因此焦距输出行不能通过下一轮 R/T 的梯度“顺便学好”。合成反向实测：`pose_branch.fc2` 的 R/T 行梯度范数 `.303886`，FoV 行为 **0**。共享 trunk/weight decay 可以改变数值，但不能等同于学到了正确焦距。

FoV 激活是 ReLU，0 会在 `f=(W/2)/tan(FoV/2)` 中变成 inf；[warp](model/util/warp.py) 将非有限投影移到图外。这避免 NaN，却可能使某些帧没有几何证据。**finite loss / 无 NaN 不是代价体有效性证明。** 2026-09-10 实测已确认两版 checkpoint 抽查帧全部有非有限 K、实际 raw 体全零；具体是哪个焦距轴、inf 还是 NaN，待现有完整 JSON 核对，不把合成 FoV=0 的构造冒充真实 logits。

新增合成反向验证：故意令 FoV preactivation 为负，即便 `weight_focal=1`，FoV loss≈`.24915`，该输出行梯度仍为 **0**。因此不能把“打开 focal 权重”当作充分修复；新版本需要可训练且严格正值的焦距/有界 FoV 参数化和初始化。共享 R/T 更新可能偶然改变其符号，但不是恢复机制的保证。

### P0-B：平移有两次归一化；之前的诊断还原也错了

[Cameraloss](loss/videoloss.py) 先构造 `E_t E_0^{-1}`，用第一相机坐标中的平均有效点距离 $A$ 归一化；`compute_camera_loss.forward` 再按最大平移范数归一化。记 $L=\max_t\|t_t^{rel}\|$：

$$
t_t^{sup}=\frac{t_t^{rel}/A}{L/A+10^{-6}}
=\frac{t_t^{rel}}{L+10^{-6}A}.
$$

所以它约为“最大 clip baseline 归一化”，**不是简单 scene-depth gauge**。固定米制 plane sweep 没有自动获得相同单位。

已经修正 [camera diagnostic](scripts/diagnose_gem_camera.py)：返回有效米制单位 `L+1e-6*A`。用真实 loss 的目标编码调用捕获验证：0.8m 最大平移、约 10.3067m 场景距离时，loss 的末帧 tx≈`.999987`；旧诊断≈`.077619`。正确恢复单位≈`.800010m`，不是 `10.3067m`。

该恢复使用 GT，仅用于定位 bug，**不能作为 RGB-only 部署修复或 benchmark 输入**。上线版本需要预测共同尺度，或采用自洽的归一化深度/采样契约。

### P0-C：裁剪后的主点不是中心，当前相机表示不能表达

[实际数据管线](dataset/dataset_mix.py)将 1242×375 resize 为 1484×448，再随机裁 448×448。其标签 `cx≈741.291−left_margin`，可能范围约 `[-294.709,741.291]`；[pose decoding](model/tools/pose_enc.py)却始终输出 `cx=224`。

即使焦距和 R/T 全正确，非中心 crop 的 rays 也不一定正确。主点在画面外并非非法：它是 off-axis crop 的几何结果。纯横移、相同 K 的简化测试可能消掉该误差，必须覆盖旋转、前进和非中心投影。

注意：现有 FoV **作为焦距编码**仍能 round-trip `fx/fy`；不要又误判成“必须把 FoV 标签改成非对称视角公式”。真正缺失的是 `cx/cy` 表示/同步 crop 契约，单换 FoV 公式并不能修好投影。横纵 resize 比例的小差异另行记录，不将其夸大为此次全部退化根因。

### P0-D：给查表索引使用仿射自由的 loss

当前 hypothesis 是固定 depth bounds 上的 bin index：

$$
q=i/(D-1),\qquad Z^{-1}=Z_{max}^{-1}+q(Z_{min}^{-1}-Z_{max}^{-1}).
$$

因此 `q` 不只是任意相对深度图，它同时决定下次读哪一段 cost volume。SSI 可以把 `q` 与 `.5q+.25` 都拟合成相同 GT。合成例中两者 loss 均小于 `1e-6`，但索引平均已差约 **3.024 bins**。

这不是“SSI 普遍错误”，而是 **SSI-only 没约束物理查表坐标**。修复时需要同一 K/T/bounds 单位下的 index supervision，或把可辨识的内部 index 与外部相对深度输出分开；不能仅继续调 gamma。

此外 legacy loss 的 `clamp(min=.005)` 用在 normalized index 后，若 index 有物理含义，会将 `.5–80m` 模式中约 **44.57–80m** 的区间压到相同下界；`3–80m` 则约 **70.90–80m**。这是该表示契约下的潜在远端截断，不是已经实测的受影响像素比例。

### P0-E：GRU 第二路 lookup 接错了；计算图可达不等于坏相机下有有效梯度

[forward](model/dpt_cost_volume_convnext.py)当前调用 `VolumeIndexer(logits, softmax(logits))`。官方第二路保留 raw correlation，让迭代仍能直接查局部匹配证据。这里把两路都建立在正则化 logits 上。

不过 raw volume **仍通过 3D aggregation 和 hidden initialization 连接输出**；使用有限有效相机的小型模型，最终 loss 能反传到 matcher，gradient norm≈`.00850`。这只证明计算图没有永久断开，不能替代坏相机状态下的梯度验证。新增 inf-K 测试中，valid mask 将匹配梯度实际压成 0，但 GRU 仍有非零梯度。必须区分“实现图可达”“当前输入提供证据”和“历史训练期间是否学到”，不能只由初值/末轮 loss 倒推。

## 4. 哪些实现已通过，不能错怪

[专门的回归测试](test/test_stereogru_contracts.py)加上[原 smoke](test/test_cost_volume_head.py)和[诊断测试](test/test_diagnostics.py)验证了：

- `E_src @ inv(E_ref)` 的 world-to-camera 相对方向、旋转/非中心 K/非单位参考相机的投影。
- inverse-depth 轴 lookup、batch/H/W 展平与双层 pooling、亚 bin 插值。
- 三层 GRU 的 gate 形式、跨尺度更新顺序与官方一致；每轮 detach index 也来自官方，不是误切梯度。
- normalized inverse-depth index 的 convex upsample **不应该额外乘 stride**；不能照搬像素 disparity 的 stride 缩放。
- 有效相机下 matcher 到最终输出有有限非零梯度；非有限相机下的空体/零 matcher 梯度另有回归，历史相机 detach 行为保留。
- pose dropout 改的是 ASTT 用的 embeddings，不改供 cost-volume 的原始 `extrinsic/intrinsic`。`use_astt=false` 的这条 arm 不能再归咎于那次 eval dropout bug。

初审 focused suite 为 32 passed；回传后的补测为 **36 passed（10.83s）**，包含死 ReLU、空体梯度、焦距非有限覆盖率及原 objective/八路 CLI 回归。未跑耗时全套；真实 checkpoint 诊断由用户在阿里云完成，本机没有这些权重。

## 5. 诊断自身已修的错误

原 [oracle](scripts/diagnose_cost_volume_oracle.py)有三处会误导结果的逻辑，现已修复并加测试：

1. 原先 rank=`1+count(score>true_score)`，全相等的 D-bin 体被算作 top1=100%。现在采用随机平局打破的期望命中率：32-bin 平体应为 **1/32**，并报告 unique winner、chance、margin。
2. true bin 改为**逆深度距离**最近，而非米制距离；GT 超出 sample bounds 单列，不伪装为边界 bin 命中。
3. `score_std` 改为每个像素沿 depth 轴的变化；不把不同像素的纹理差异当深度分辨力。

2026-09-10 又修正 camera 汇总：旧 `_mean` 会删除非有限 focal error，可能在全部相机帧都坏时仍显示有限“平均焦距误差”。camera diagnostic v3 对任何 focal 失败将总体误差标为未定义，另列 `finite_only` / 完整有效帧均值、轴/帧覆盖率及 fx/fy 失败数。**不用重跑旧 GPU 诊断来确认已发现的空体故障**；旧 `.6663` 必须解释为有限项条件均值。

同时精确修正双归一化、为 Albumentations 单独设种子、跨索引均匀选 clip、记录路径，跳过 dataset 的随机坏样本重试；比较 mode-specific / 八路共同 / 与 GT 两两共同支持域。空交集直接报告，不能换另一批像素“改善”。in-bounds mask 仍不等于真正的遮挡/静态背景 mask，因此这些是排错量，不是新 benchmark。

新增 `--trace-head` 不改变 eval 状态或参数，记录 raw correlation、GEV entropy、每次 lookup 的越界/零值比例、delta、最终下界截断比例。`--descriptor backbone` 绕过已训练 matcher，但沿用同 checkpoint 的 backbone/LoRA；可定位 projection 层问题，**不冒充未被错误训练影响的干净预训练特征**。不同 descriptor 通道数下原始相关幅值不同，不能直接比较其 margin 数量级。

## 6. 阿里云首先需要什么

[诊断入口](scripts/audit_stereogru_aliyun.sh)顺序执行，不并行抢显存：

1. costvol / costvol_d3：相同 8 个确定 clip，原 matcher + 完整 GRU trace。
2. costvol_d3：相同 clip，projection 前 backbone 特征再做 oracle。
3. ms_gem：相机辅助任务参照的 K/R/T 诊断（不是把其相机冒充 costvol 的相机）。

八个 camera mode：

| mode | K | T | 问题 |
|---|---|---|---|
| gt_metric | GT | GT 米制 | 校准几何下特征是否有匹配证据 |
| gt_normalized | GT | 两次归一化 GT，depth/bins 同步缩放 | 应与 gt_metric 数值一致 |
| pred_raw | GEM | GEM 原始单位 | 精确对应旧 sweep 输入 |
| pred_rescaled | GEM | GEM × 正确 GT metric unit | 尺度 alone 能解释多少 |
| gtK_predT_raw | GT | GEM 原始 | K 修正与 T 尺度的影响分开 |
| gtK_predT_scaled | GT | GEM × 正确 unit | 还剩 R/T 精度问题吗 |
| predK_gtT | GEM | GT 米制 | 单独检验 K |
| gtF_center_gtT | GT focal、主点强设中心 | GT 米制 | 单独检验不可表达的 crop 主点 |

**返回终端汇总、配置摘要、provenance 即可；必要时再要完整 JSON。** 脚本校验所有三个权重，严格加载，检查指定 GPU 是否占用；忙则退出。不安装依赖、不训练、不改 checkpoint、不覆盖旧预测。

推荐从一个只含代码的 `git archive` 临时快照运行，权重用阿里云原路径；不 pull/merge/reset/stash 当前训练工作树。这也规避了已知的“Git 跟踪 checkpoint 软链但阿里云是真目录”的风险。

## 7. 根据回传结果分流，而不是继续猜

| 观察 | 下一步 | 不能推出 |
|---|---|---|
| 非有限 K / raw 体大量为零，GT camera 正常 | 先修 camera focal/主点/共同尺度契约 | GRU 不适合视频 |
| GT camera 下 matcher 弱、backbone 明显好 | 查 matcher 是否被错误几何训练破坏；固定校准输入做小样本学习测试 | backbone 本身无对应能力 |
| GT camera 下两者都弱 | 测静态可见纹理区域、真实 depth bin/采样坐标、相机路径和帧间运动；需要时再比较 RGB/干净初始特征 | 继续加 GRU 一定能补，或架构已被否定 |
| raw 有明显证据但 GEV 初值坏 | 查 aggregation、初值绝对 index 监督和权重，恢复双体 lookup | volume 与最终 loss 完全断开 |
| index 经常逃出 `[0,D-1]`、lookup 多为零 | 查不受约束的更新/gauge；记录分轮曲线 | 无条件硬 clamp index 就是最终修法 |
| 修正 GT oracle 后小样本仍学不会 | 按同一数据/初始化分开测 volume-only 与 volume+GRU，再定位更新/上采样 | 直接开新 10k 大扫参 |

下一阶段验收顺序：**几何/监督合约单测 → 校准相机小样本 overfit → 同协议 volume-only baseline 完成 → volume+GRU → 分别替换预测 K/T → 多场景标准指标**。GT camera 只在实现 oracle 表使用，不进入 RGB-only 主结果。

任何行为修复都新建注册模块/config，不能静默改变旧 head/loss 后续训原目录；新协议 baseline 先完成，再启动 method。完整预训练先验/RGB 旁路方案仍可作备选，但在本审计闭环之前，不再用旧 costvol 失败作为转向它的依据。

## 8. 2026-09-10 阿里云回传：几何输入失效已确认

### 8.1 来源与直接证据

来源为用户回传的完整终端 summary，运行标识 `mckzcNRl`，源快照 `b8e2dd3`，GPU 4，diagnostic v2。三个 checkpoint 均通过 runner 的文件检查/strict load，四项诊断完成。下表保留终端四位小数，**不是新 benchmark，也不是完整 JSON 的精度**；配置摘要与权重 hash 的原文尚待归档。每版只抽查 8 clips，每 clip 4 帧，不能当作 32 个独立统计样本。

| 抽查证据 | costvol (.5–80m) | costvol_d3 (3–80m) |
|---|---:|---:|
| 包含非有限 K 的帧 | 32/32 | 32/32 |
| pred_raw / pred_rescaled / predK_gtT 有效支持 | 全为 0 | 全为 0 |
| GT K + GT T 的有效支持 | .9491 | .9495 |
| GT K + 原始预测 T 的有效支持 | .9515 | .9525 |
| 真实 forward 的 raw-zero 像素比例 | 1.0000，所有 clips | 1.0000，所有 clips |
| GEV 归一化熵 | 1.0000，所有 clips | 1.0000，所有 clips |
| 第 8 次 lookup 前 index 越界比例 | .2082 | .2784 |
| 最终 raw output 低于 legacy .005 下界的比例 | .0460 | .0000 |

**定位链闭合：同一描述子，保留预测 K 而替换 T/恢复尺度仍无支持；替换为 GT K 后支持恢复。实际 head 的输入体也为空。** 这足以确认预测 K 是本轮空体故障的主要阻塞，不能再以这两个 checkpoint 的 `.4297/.4171` 否定 StereoGRU。

注意：support 是 GT true-bin 上的有效投影比例，不是所有假设/像素的可见性。GEV entropy 接近最大只说明 depth 概率接近均匀，不单独证明所有 logits 严格等于 0；第 8 次 lookup 是最后一次更新**之前**，不是最终 index 越界统计；output floor 不使用 GT mask，也不是远端真实深度像素占比。

### 8.2 有证据，不代表剩余契约已经修好

校准 GT camera 下，costvol 的 raw 检索 top1 `.3778` 对 chance `.0571`；d3 matcher 为 `.1364` 对 `.0331`；d3 backbone 为 `.1440` 对 `.0331`。GT-normalized 分别为 `.3797/.1365`，与 metric 模式总体接近，但不声称数值完全一致。

**不能横比 `.3778 > .1364` 来挑 .5m 配置**：两版 inverse-depth bin 宽度不同、有效候选数/chance 不同，粗 bin 的命中更容易，且都不是重建误差。d3 backbone 仅比 matcher 高 `.0076`，没有显示绕过 matcher 就恢复强匹配；也不能据此判定 matcher 无影响或差异在噪声内。

以下各格为 **mode top1 / 同一像素支持域上的 GT top1**；不跨行混用 GT 值：

| 相机替换 | costvol | costvol_d3 |
|---|---:|---:|
| GT K + 原始预测 T | .3191 / .3810 | .0836 / .1391 |
| GT K + 尺度还原预测 T | .2348 / .3780 | .0736 / .1374 |
| GT focal + 主点强设中心 + GT T | .3422 / .3778 | .1076 / .1365 |

含义：K 修好以后仍有 R/T 质量/共同尺度问题，单独把主点设为中心也损失检索证据。**尺度还原后的 top1 更低不能反证正确的单位公式**：当前预测 T 本身不准确，而旧训练没有满足共同几何/索引契约；不能为旧权重保留错误单位来挑更好看的 oracle。所有 GT-assisted 结果只用于定位，不进入 RGB-only 排名。

八路共同支持为 0 是预期结果：三路保留坏 K 已经全空；因此本次用 pairwise-GT 支持作有效比较，不能把共同支持 0 当作诊断程序失败。

### 8.3 相同的 GRU 统计说明什么

每个 checkpoint 的 8 clips 都打印出相同的逐轮统计：

| trace | costvol | costvol_d3 |
|---|---|---|
| lookup 越界 | [0, 0, .0466, .1260, .1540, .1786, .1961, .2082] | [0, 0, .1311, .2012, .2296, .2500, .2659, .2784] |
| delta mean | [-.1879, .8433, .2172, -.1579, -.1548, -.1341, -.1116, -.0855] | [.2678, 1.9726, .8832, .4774, .4909, .5247, .5550, .5898] |

这与“几何证据没有进入、更新退化为偏置/空间边界等先验”的故障模式一致，但 **相同的四位小数均值不是整张预测逐元素一致的证明**。

本地新增合成对照使用原 head、一个 inf focal 轴、默认 BN affine/running stats：raw 为 0、GEV std=0、替换全部 backbone 特征后输出最大差=0，matcher 梯度为 0，index-head 梯度范数≈`.43561`。这明确证明 **GRU loss 可以继续下降而没有图像匹配信号**。训练后的 BN 偏置/运行统计可能把非零值注入乘法图像引导支路，所以不能把该合成输出相等或梯度数值直接套到真实 checkpoint，更不能补写“10k 每步都为空体”。

### 8.4 运动分布与 ms_gem 参照的边界

同一组 clips 的 `translation_unit_metres` 如下（这是最大第一帧相对平移加小量，不是相邻帧 baseline）：

| sample index | unit (m) |
|---:|---:|
| 0 | .0021676884 |
| 37 | .0006615604 |
| 75 | .0017123126 |
| 112 | .0026382890 |
| 150 | .0014226980 |
| 187 | .2423851490 |
| 225 | 1.9823402166 |
| 263 | 2.8302040100 |

前五段处于近零运动量级；需区分真实近静止、GT 数值残差及 float32 相对位姿消减误差，不将它们直接解释为精确的毫米运动。这使汇总检索指标不能代替“有足够 parallax 的匹配能力”判断。下一步 overfit 从 **training scenes** 预先选可用运动/纹理样本，另保留静止负对照；不能为提高 benchmark 分数剔除静止帧，也不能据此称整个 VKITTI 无运动。

ms_gem 回传：非有限 K 帧比例 1.0000，中心误差 246.8767px，旋转误差 .5319°，平移方向余弦 .1137、幅值比 11.5224。`.6663` 是 focal 的**有限轴条件均值**，完整相机可用性仍为零；幅值比也可能被近零 GT 平移放大。`pred_consistent_warp_valid=0` 是 camera helper 对非有限输入直接拒绝的结果，和 costvol 实际 forward 的零体证据要分开说明。

这些是 **ms_gem 自己的相机权重**，不可搬作 costvol 的 R/T 误差。由于 ms_gem 的 `use_astt=false`、普通深度头不读取预测 K/T，它可以同时有较好深度和失效相机；此结果不自动否定其 `.1134` 深度观察，反而说明“相机辅助任务存在”不等于“相机能供几何投影使用”。

### 8.5 修复次序：不是再跑一轮原配置

1. **几何输入/表示先过合同测试。** 新版本显式检查 K/T 有限、焦距正值和可逆性；修正逐帧 crop/resize 主点。拒绝非法几何，不以 `nan_to_num(K)`、巨大焦距或静默空体冒充恢复。为训练和推理选定同一 depth/translation/bin 单位；GT 尺度只许出现在 oracle。
2. **校准 oracle 小样本 → volume-only baseline。** 用 training scenes 的固定可运动小样本、GT K/T、同一 bounds 的绝对 normalized inverse-depth/index 监督，先证明 raw/GEV 可学习、梯度正常。`q=0` 是有效远平面，不再把相对深度 loss 的 .005 下界套在物理 index 上。冻结 split、公共初始化、seed、预算、mask 和 evaluator；先完成无 GRU 基线。
3. **然后才开 raw+GEV 的 GRU 对照。** 恢复官方双体查表，独立约束内部 index，记录分轮预测/越界；与完成的 volume-only 对照比较。不要仅硬 clamp 来遮住失控，更不要给旧 checkpoint 换查表后把即时差异叫作训练修复收益。
4. **最后逐个接回预测 K/T。** 相机使用可学习的严格正焦距/有界 FoV、非零监督、可表达主点的表示与自洽共同尺度。分别验证 K、R/T 后才进入 RGB-only 正式基线/方法；所有行为版本用新增注册模块/config 表达，旧 run 冻结。GT-camera 结果永远不冒充 RGB-only SOTA。

本次已完成结果解释、诊断汇总修正与 36 项回归，**尚未声称实现上述新训练版本或取得修复后深度指标**。只需再从本次现成结果读取配置、权重 hash、fx/fy 和原始 trace 范围作归档，不需要重复整套 GPU 诊断来确认首要阻塞。