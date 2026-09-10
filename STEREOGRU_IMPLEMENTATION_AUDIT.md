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

FoV 激活是 ReLU，0 会在 `f=(W/2)/tan(FoV/2)` 中变成 inf；[warp](model/util/warp.py) 将非有限投影移到图外。这避免 NaN，却可能使某些帧没有几何证据。**finite loss / 无 NaN 不是代价体有效性证明。** 2026-09-10 实测已确认两版 checkpoint 抽查帧全部有非有限 K、实际 raw 体全零；后续首 clip 精确回传确认是 fy=+inf。原始 preactivation 未记录，不把合成负值构造冒充真实 logits。

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

来源为用户回传的完整终端 summary，运行标识 `mckzcNRl`，源快照 `b8e2dd3`，GPU 4，diagnostic v2。三个 checkpoint 均通过 runner 的文件检查/strict load，四项诊断完成。下表保留终端四位小数，**不是新 benchmark，也不是完整 JSON 的精度**；配置/权重 hash 和首 clip 精确统计已在后续回传归档，见第 8.6 节。每版只抽查 8 clips，每 clip 4 帧，不能当作 32 个独立统计样本。

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

第一次回传已完成结果解释、诊断汇总修正与 36 项回归。后续精确证据和新增隔离入口见下文；没有修复后真实数据深度指标，不再重复诊断取证。

### 8.6 第二次回传：fy=inf 与严格零 GEV，源码指纹吻合

机器可读记录见 [reported evidence](results/stereogru/20260910_reported_evidence.json)。六个源码 SHA256 已在本机逐一与 `b8e2dd3` 的 Git 对象核对，**全部一致**。provenance 中的 `eb2c9ac` 是 runner 查询的阿里云训练工作树 HEAD，不是执行快照版本；因此不是“运行到了旧诊断”。回传环境为 Torch `2.9.1+cu128`，CUDA/bf16 可用；不能把它自动追溯为历史训练环境。

权重 SHA256 原样归档；本机没有旧训练权重，不能声称独立复算了它们。回传配置确认 focal 权重 0、ASTT 关、32 bins/8 次 GRU/4 帧；配置来源仍是快照，非历史 model-only checkpoint 内嵌配置。

| 首 clip 精确证据 | costvol | costvol_d3 |
|---|---|---|
| fx（四帧） | 1204.89, 2803.06, 2923.40, 2904.92 | 1007.03, 1802.54, 1836.58, 1836.44 |
| fy（四帧） | **全部 +inf** | **全部 +inf** |
| GEV min/max/std/depth_std | **全部严格 0** | **全部严格 0** |
| final raw q 范围 | −.000737…1.917865 | .008518…2.595232 |
| final raw q>1 | **22.3369%** | **29.3950%** |

这里明确到竖向焦距，而非所有 fx/fy 都坏。之前统计已知各版 32/32 抽查帧的 K 有非有限项，但本次逐轴信息仅覆盖首 clip，不能把“首 clip 的 fy”扩写为所有帧逐轴统计。

严格零 GEV 排除了“只是 softmax 熵比较高”的歧义。head 从零 logits 得到均匀概率和初始 index=15.5，之后的更新读取的是没有图像匹配信息的 GEV/概率体与偏置初始化 hidden；后续非零输出不证明匹配已工作。`1.000000119` 的 entropy 是浮点舍入，不是概率逻辑异常。q>1 的最终上采样输出统计也与前文“第 8 次 lookup 前越界”不同，不能混成同一个比例。

这已经完成首要故障取证，**不再要求重复 GPU 诊断或猜测换哪个相机轴**。剩余问题转入正确实现的受控验证。

## 9. 新增隔离的 calibrated volume-only pilot（不是完整重训）

实现文件：[registered volume-only head](model/dpt_calibrated_volume_only_convnext.py)、[absolute index objective](loss/objective_calibrated_index.py)、[camera-safe training clips](dataset/stereogru_calibration.py)、[standalone runner](scripts/stereogru_calibrated_baseline.py)、[shell entry](scripts/run_stereogru_calibrated_baseline.sh)、[fixed config](config/stereogru/calibrated_volume_only.yaml)。旧 head、核心 train、旧 loss/config 均不改。

| 控制项 | 本次固定内容 |
|---|---|
| 唯一目的 | GT-camera 下，volume 本身能否获得梯度并拟合固定训练样本 |
| 数据 | 仅 Scene01/02/18 的 15-deg-left、Camera_0；2 个不重叠 clip，每 clip 连续 4 帧 |
| 选择 | 按 GT 最大第一帧相对平移 .5–5m 预选，跨 scene 优先；不按 loss 挑样本，不自动放宽门槛 |
| 相机/预处理 | 读取逐帧原始 K；原 E 先 float64 第一帧重基再 cast；共同固定 crop256、独立 sx/sy、OpenCV resize 与原生 ConvNeXt stride-conv 像素中心修正 |
| backbone | 干净官方 DINOv3 ConvNeXt-S；冻结且 eval，无 LoRA、无 GEM；预缓存特征 |
| 头 | 新注册 volume-only：保留 matcher、hourglass 和上采样，移除所有 GRU/lookup 参数 |
| 几何/监督 | 米制 K/T/depth 与 3–80m/32 bins；绝对 normalized inverse-depth L1，不用 SSI、相机 loss 或 .005 floor |
| 预算 | 200 步，逐 clip 循环，AdamW lr=.001、wd=0、clip-grad=1、seed0、FP32；只为 overfit 定义的新预算 |
| 指标 | 固定训练 clip 的 index-L1，以及不做 affine 对齐的 AbsRel/RMSE/δ1；**不代表泛化或 RGB-only** |
| 自动后续 | **无**；只完成这个 baseline pilot，绝不自动启动 GRU arm |

严格加载经本机真实干净权重核对：342 个 tensor，唯一显式忽略的导出键为 native `norm=False` 特征不使用的 `norms.3.weight/bias`；缺失/其他多余键报错。缺清洁权重、缺 K 标定文本、坏数据或繁忙 GPU 都退出，不换随机初始化、不拿旧 costvol 权重冒充 clean backbone。

数值防护：非有限 K/T、非正焦距、无效旋转直接报错；不把失败相机换成大数。`geometry_gauge='metric'` 必须显式确认，避免被旧 RGB-only `train.py` 误用。`q=0` 仍是有效远平面，softmax+convex baseline 的输出自然在 [0,1]，不是硬 clamp 掩盖 GRU 越界。预测 head bounds 与 objective bounds 必须相同。

坐标约定也单独验证：OpenCV resize 满足 `u'=s*(u+.5)-.5`；原生 ConvNeXt 的无 padding 4×4/2×2 stride-conv 使 stride8 特征中心落在 `8u+3.5`。新头只对传入旧 ratio-scaler 的主点做中心补偿，不改旧工具或 GT 存档。相机射线与实际 OpenCV 坐标斜坡采样均有回归，不能把 resize/crop 后的 K 再近似为固定中心。

输出为全新目录：保存固定 clip/变换/完整相机 manifest、权重和输入 SHA256、配置、初始 head、最终 head、逐步 loss/梯度/raw 诊断及初末指标。拒绝 resume/覆盖旧目录。head 新增参数会正常写入**新 pilot 目录**，不是此前只读诊断；历史 checkpoint 不改。

**验收不是“200 步跑完就说明方法可用”**：先检查 raw 非空与有限非零 matcher 梯度，再检查相同训练支持域上的初末 index-L1/深度误差及 eval 模式输出。没有学习信号就回到 raw/aggregation/监督链，不扩大预算。即使小样本能拟合，也仅通过实现 gate；下一步仍需完整同协议 volume-only baseline，再开 raw+GEV GRU 对照。

相对于历史 costvol，此 pilot 改变了几何输入、输出监督、clean/frozen backbone、crop、优化预算等，**不能拿它与 `.4297/.4171` 当单变量提点实验**。后续 baseline/method 必须共同遵循新协议与共享初始化，不把 overfit 和测试结果混表。

本地验证：**73 passed（11.03s）**，含新校准合约、train-only 文件夹/标定 fixture、无 GPU 的三步独立运行，以及原诊断、decoder/backbone/objective/freeze/mix 注册表回归。另用本地真实干净 backbone 权重验证 strict loading 和有限四级特征输出。随后用户已在阿里云完成真实 VKITTI 200 步，结果见第 10 节；本机未运行该真实数据训练。

### 9.1 部署阻塞：独立权重文件未找到，但可验证恢复冻结基座

阿里云两次尝试都没有启动训练：首次默认权重路径不存在；随后在已知 weights/checkpoint 目录按文件名查找，没有任何候选哈希输出。这只说明限定目录/命名下未找到，不证明整台机器没有文件；不再继续让用户猜路径。

新增 [verified recovery](scripts/recover_verified_convnext_backbone.py) 与[拒错测试](test/test_verified_backbone_recovery.py)。利用现成完整 PP-DPT checkpoint 中的 `pretrained.model` 参数：仅去除 DDP 前缀及已知 fc1/fc2 LoRA wrapper 的 `.base` 层级，丢弃 adapter、GEM、decoder；**不合并或反算 LoRA**。不能只因旧代码写了 freeze 就相信参数没变，必须核对全部基座张量。

官方原文件 SHA256 为 `296db49dcbd622625befd3fc23318cbbcd98049f4c4b0cc026463de6bcd24952`。本机按已有严格加载路径提取 342 个 native tensors（仅排除原导出多余、特征不使用的两个 norm 键），以排序后的名称/dtype/shape/连续原始字节构造可重复指纹：`eb323d607420145dc0baa072355d232ffd9e953554e9765f3fc625324b63246c`。

恢复程序要求数量和完整规范指纹同时一致；哪怕一个 tensor 的数值、形状、dtype 或名字变化，也拒绝导出。CLI 没有跳过校验的选项。只有验证通过才以独占写入方式产生新权重和来源报告，原 checkpoint 始终只读；CPU 恢复本身不占训练 GPU。**新序列化文件的文件 SHA256 不会等于官方原文件 SHA256；相等的是全部 native tensor 状态，二者明确分开记录。**

因此这不是拿 finetuned backbone 替换 clean 初始化：通过时恰好恢复官方相同参数，不改变 pilot 协议；不通过就停止，不能用“冻结过”“形状兼容”或只看几个 tensor 来放行。

本地全尺寸验证：真实官方 ConvNeXt-S 加入 72 处非零 LoRA，再包装成模拟 PP-DPT/DDP 完整 state dict；恢复时排除 144 个 adapter tensors，342 个基座张量指纹通过，重新走 pilot 严格加载后四级特征最大差均为 **0**。临时模拟导出已清理。随后用户已回传真实阿里云 checkpoint 同样通过官方指纹，记录见第 10 节；两种证据不可混写。

相关回归合计 **83 passed（11.73s）**；新增用例覆盖六种指纹变化、重复键/未知 wrapper、导出重读、源文件不变及失败时不写出。部署指令重新创建代码快照和独立权重目录，不依赖之前终端里的 `$D`；只在 `VERIFIED_OFFICIAL_BASE` 通过后继续相同 200 步 pilot，不改变任何训练协议。

## 10. 真实校准 pilot 已完成：可学习性通过，不等于 GRU/泛化通过

用户回传于 2026-09-10，运行标识 `stereogru_calibrated_WuYmJxow`，源快照 `25dd7f0`，GPU 4。数值归档见 [pilot evidence](results/stereogru/20260910_calibrated_pilot.json)。`VERIFIED_OFFICIAL_BASE` 确认真实 costvol checkpoint 中 342 个基座 tensor 与官方规范指纹完全一致，排除了 144 个 LoRA tensor 和 300 个其他条目；源权重未改。恢复文件 SHA256 为 `1a19497d8e8fa24e4051c871733b12fbdf343aca84d04107f6f485aa71171a35`，不能与官方序列化原文件哈希混淆。

### 10.1 同一批训练像素上的初末结果

| 指标 | initial（eval） | final（eval） |
|---|---:|---:|
| absolute normalized-index L1 | .3001017720 | **.0129091145** |
| 对应平均 bin index 绝对差（×31） | 9.30315 | **.40018** |
| AbsRel | .5722318420 | **.0910153738** |
| RMSE (m) | 21.9348572 | **4.7431782** |
| δ1 | .0982158818 | **.9553183372** |
| 有效监督像素 | 374807 | **374807** |

两个 clip 分别有 200894 / 173913 个有效像素，初末加权 index-L1 已按原输出独立核对。没有 affine alignment、没有丢弃困难像素来换这组下降。训练中首次 matcher 梯度范数（clip 后）为 `.0881171`，真实 raw-zero 从旧失效模型的全零状态变为本 pilot 的 `.0317383`，末次两个 clip 的 GEV depth-std 为 `1.11582 / 1.26184`，q 范围均在 [0,1]。

**通过的是：给定有效标定、正确索引目标与新协议，volume-only 路径能够得到梯度并学习拟合这两个 clip。** 不需要先靠 GRU 才能让 volume 读出学习，旧“cost volume 本身不 work”的解释应撤回。与此同时，不能把这套多项修正后的 overfit 分数与旧 `.4297/.4171` 作单因素效果比较，也不能与 ms_gem 的 held-out `.1134` 横比。

### 10.2 日志细节与不能夸大的内容

- 初末的 overall 指标均为 **eval 模式、两 clip 加权**；step200 的 `.0091611` 是一个 clip 在 train 模式下、该次 optimizer update 前的 loss，不等于 final `.0129091`。存在 BatchNorm 和 clip 差异，不能只由两者不等认定评测 bug。
- initial eval GEV std 约 `1e-6`，step1 train 约 `.2766`，这与 BN train/eval 不同相容，不应写成“一步训练已经突然恢复强匹配”；应看最终 eval 与初始 eval。
- `.0317383` 是 raw 数值全零的像素比例，不是“96.8% 真正可见/无遮挡”的证明；缺乏纹理、边界等具体来源尚未定位。
- 最终第二 clip 的 q_min `.00424935` 低于旧 `.005` floor 但合法；此 pilot 没有错误截断。初值 q 在边界低于 .5 与零 padding 的 convex upsample 相容，不自动是相机错误。
- 恢复干净 base 的成功只证明冻结基座未改变，不代表原 LoRA 没被坏几何训练影响。本 pilot 不使用这些 LoRA。
- `9.6779s` 是该小型缓存特征 pilot 的计时；包含 setup，但不含前置恢复，也没有每步 backbone 前向，不能推算正式全数据训练只需同样时间。
- Albumentations 版本查询超时和 `float(loss)` 的 scalar warning 没有终止训练；后者是日志读取已 backward 的 scalar，不改变优化目标。为保证旧 run 源码指纹可回放，本阶段不修改该旧入口。新 readout 关闭无关版本联网检查。

小样本 loss 下降仍可能主要依赖记忆、图像引导或宽泛先验；**非零 matcher 梯度不等于已证明模型利用了正确多视角深度顺序**。因此不立即跳到 GRU method、10k 全量训练或 RGB-only/SOTA 声明。

### 10.3 下一步仅做只读依赖/未拟合样本检查

新增 [readout](scripts/diagnose_stereogru_pilot.py) 和 [readout entry](scripts/run_stereogru_pilot_readout.sh)。**不训练、不更新 BN、不写 checkpoint**，只在新目录保存报告；旧 pilot 与训练源码保持不变。

1. 校验 pilot 的 config/manifest 一致性，原源码、输入 RGB/depth/标定文件与 clean backbone 哈希；严格加载已保存最终 head，先在原两个 clip 上复现 final 的有效像素数和四个汇总指标，超出 `rtol=1e-4, atol=1e-6` 则停止。
2. 原 pilot 未独立保存最终 head 哈希，所以不能声称“已有外部指纹证明历史 head 逐字节未变”；readout 记录当前 head/file 指纹、检查运行前后不变，并核对 checkpoint metadata 与原指标。此边界在报告中显式说明。
3. 原始体送入 aggregation 前，分别使用原体、全零体、逐像素沿 depth 均值展开的平体、depth 顺序反转的体。**同一 head、guides、GT K/T、bounds 和 GT 支持域，只改变 volume_stem 的输入**。原 raw 统计和实际干预后统计分开显示。干预损害说明敏感性，不单独证明正确三角化，尤其全零输入也会造成分布变化。
4. 从同一 training scene pool、同一预定义运动筛选规则追加 6 个与原拟合帧不重叠的 clip，不按 loss 选。原 fit prefix 必须与 manifest 完全相同。未参与拟合的训练 scene 与同 scene 新 clip 分开显示；后者可能只与拟合窗口相邻，报告帧间隙，**不冒称独立 held-out benchmark**。正式的 Scene06/20 不在此步骤作调参集。

解读顺序：若原指标不能复现，先排 artifact/环境；若 zero/flat/reverse 都基本不影响输出，不能把 overfit 当匹配证据；若干预有影响但新 clip 明显差，先处理覆盖/过拟合。只有实现 gate 与有意义的新样本检查成立，才固定完整 calibrated baseline 的协议和预算，**先跑完 baseline，再启动同协议 raw+GEV GRU**。当前 readout 没有 optimizer 或自动下一组开关。

本地最终回归 **94 passed（12.61s）**，含已保存合成 pilot 的完整回放、相同支持域、原始/有效 raw 区分、hook 异常清理、源码或指标不一致拒绝，以及原恢复/校准/各注册表测试。原 pilot 的 9 个 provenance-covered 源码文件仍与 `25dd7f0` 逐字节相同。随后用户已完成真实阿里云 readout，见第 11 节；没有新训练或 GRU 结果。

## 11. 只读 readout 回传：依赖 raw 输入，但 depth 轴增益尚未建立

运行标识 `stereogru_readout_JyCpLkpw`，源快照 `c194933`，GPU 4；[终端数值归档](results/stereogru/20260910_pilot_readout.json)。用户粘贴中同一运行 ID 和表格重复出现，**只计一轮、一个 checkpoint 的证据，不冒充重复运行或 paired seeds**。回传明确 `PILOT_REPLAYED=true`、`training_steps=0`、`head_state_unchanged=true`；原 fitted normal 行也与 pilot 最终指标在打印精度上相同。

### 11.1 核心数值与判断

下表为同一 group、相同 GT 支持域上的 AbsRel。各 group 各自的场景/支持不同，不能把跨 group 差值当作受控方法提升。

| group | clips | normal | zero raw | flat depth raw | reverse depth raw |
|---|---:|---:|---:|---:|---:|
| fitted | 2 | .091015 | .415644 | .095759 | .120664 |
| unfitted training scene | 2 | .253740 | .382535 | .244938 | .236112 |
| unfitted same scene | 4 | .151359 | .428079 | .149185 | .158266 |

**A. 对清空 raw 体高度敏感，但不能把这个效应全归因于正确几何。** fitted AbsRel 从 `.091015` 到 `.415644`，另外两组也显著退化。全零操作同时删除逐像素/group 的幅值、空间均值、depth 变化，并改变 aggregation/BN 所见的输入分布；它证明当前固定模型需要非零 raw 输入，不能证明它依赖正确的逐 bin 匹配峰。

**B. 拉平 depth 后大部分预测仍被保留，说明不能从 overfit 成功推出深度轴判别已充分学会。** fitted AbsRel 仅从 `.091015` 到 `.095759`（约 +5.21%）；两组未拟合 clip 的 AbsRel 反而小幅降低。mean absolute delta-q 只有 `.003841–.005232`，但这是所有输出像素统计，不能直接与 GT-masked index-L1 混算或换成“几何贡献百分比”。

这里的 flat 仍保留来自真实 GT-camera sweep 的逐像素/group 均值、view weighting/有效性影响，以及全部图像 guides。它**不是纯单帧或完全没有几何信息**；因此“模型肯定只走单目捷径”“几何贡献只占 5%”都不是本实验能给出的结论。

**C. depth 顺序不是完全无影响，但收益没有稳定迁移。** fitted 反转后 AbsRel `.120664`（约 +32.58%）、δ1 `.955318→.884759`，有明确敏感性；不能说 depth 轴完全未使用。未拟合 training scene 却是 AbsRel `.253740→.236112`（约 −6.95%），而 RMSE **`13.4298→14.2513`（约 +6.12%）**。这不是一致改善，更不能把“反转 depth”当作修复方向。

同样地，flat 在未拟合 training scene 的 RMSE 从 `13.4298→13.7521`，在未拟合同场景从 `8.2713→8.6437`，后者 δ1 也略降。不能只挑 AbsRel 变化就宣称平体胜出。

**D. 原 fitted 分数与新 clip 存在明显差距，但不能立即宣布跨域泛化失败。** normal 为 `.0910 / .1514 / .2537`，与小样本记忆/覆盖差异的解释相容；只有 2/4/2 clips，未提供的完整 manifest 才含具体 scene ID 和帧间隙。同场景样本可能相邻，`unfitted_training_scene` 仍是 training pool 中未拟合的场景，不是完整 held-out benchmark，更不是 zero-shot 四集。

### 11.2 当前检查到哪一层

| 检查层级 | 状态 |
|---|---|
| 历史相机 K 失效、真实 raw/GEV 空体 | 已定位；旧负结果不能否定 StereoGRU |
| 新校准 volume-only 有梯度、能拟合两 clip | 已通过 |
| 已保存 pilot 分数可重复回放，readout 不改状态 | 已通过本轮数值回放 |
| 固定模型依赖非零 raw 输入 | 本次干预支持，但有输入分布变化的限制 |
| 正确 depth-bin 信息带来稳定未拟合收益 | **尚未建立** |
| 完整同协议 baseline 与 GRU 增益 | **未运行** |
| RGB-only/SOTA | **尚不能讨论为实测结论** |

本轮至此结束只读诊断，不再要求重复 zero/flat/reverse。下一步改成**训练时即使用平体的 C0 baseline → 同初始化/同预算的完整体 C1**，以排除“仅在推理时改输入”的混淆。预登记问题、矩阵、数据 split、预算及验收见 [匹配训练对照方案](STEREOGRU_MATCHED_CONTROL_PLAN.md)。

协议固定 Scene01/02 共 32 个 training clips、Scene18 共 16 个开发 clips，每臂 1000 updates；具体清单/间隔满足性先核验，不满足就停止而非静默换数据。它是新开发协议，旧 full-volume 两 clip pilot 不能充作 C1；C0 完成后才能启动 C1，**G1/GRU 暂不启动**。

2026-09-10 用户要求开始落实后，已新增 [C0 注册头](model/dpt_calibrated_flat_volume_convnext.py)、[固定 manifest](dataset/stereogru_matched.py)、[顺序训练/证据校验](scripts/stereogru_matched_controls.py)与[单 arm 启动入口](scripts/run_stereogru_matched_controls.sh)，详见[执行契约](STEREOGRU_MATCHED_CONTROL_PLAN.md)。相关回归 **107 passed（15.62s）**，原有实验代码保持不变；真实 C0/C1 配额和训练结果尚待阿里云运行，不把代码完成当成 baseline 已跑完。

**后续配额回传：**v1 在准备阶段停止，Scene02 只有 15 个满足既定间隔/运动条件的 clips，不能满足每场景 16 个。Scene01/18 容量为 48/41；C0 未训练。已记录[失败计数](results/stereogru/20260910_matched_quota_preflight.json)，新增 [v2 配额配置](config/stereogru/matched_c0_c1_30train.yaml)：Scene01/02 各 15 个、共 30 train，dev 仍 16；其他阈值/预算/训练逻辑不动。原 v1 保留，改用新目录准备，不以失败输出冒充 baseline 或重跑旧诊断。

## 12. v2 C0 正式完成本轮开发预算，C1 已可按原协议运行

用户回传 `stereogru_matched30_cJPdMuhW/experiment` 的 C0，源快照 `d4f3388`。30 train/16 dev 清单及 46 clips 共享缓存均成功，1000 步结束后输出 `COMPLETED C0`。记录见 [C0 原始数值](results/stereogru/20260910_matched30_C0.json)和[对照计划第 8 节](STEREOGRU_MATCHED_CONTROL_PLAN.md)。

最终 C0 开发集（Scene18）为 **AbsRel .23792445 / RMSE 9.81047361 / δ1 .62542661**，有效像素 3591702；训练集为 .11015101 / 7.46385419 / .86170226。保留 final1000，不选 step750 较低的单项 AbsRel。此结果是已校准平体的开发 baseline，不是 RGB-only/SOTA，也不能单独决定完整体胜负。

C1 随后已完成并输出对照表；共享初值、配额、mask、缓存、顺序和预算都冻结，原完成门槛验证后解锁。该次训练源码/配置保持不变，结果及下一步见第 13 节。

## 13. 第一组匹配训练完成：不能把指标取舍写成全面提升

同一 v2 experiment、seed0、final1000 的 C1 已回传 `COMPLETED C1` 和六行逐场景比较。记录见 [C1 原值与曲线](results/stereogru/20260910_matched30_C1.json)。开发 Scene18 的 **AbsRel `.23792445→.25194988`（恶化 5.89%）**、index-L1 也更差；**RMSE `9.81047361→8.80519551`（改善 10.25%）**、**δ1 `.62542661→.63985013`（增加 1.44 个百分点）**。

这对比此前只读 zero/flat/reverse 更有解释力：两臂都在各自输入下训练，且匹配初始化/预算/数据；但单 seed 并未证明稳定统一收益。C1 在 250/500/750 步的 dev 指标较好，最后1000步出现取舍，不能据此改用750步作主分数、当场减预算或断言过拟合。聚合指标也不能证明“只改善远处、损害近处”，具体原因未定位。

当前仍没有运行 GRU，因此此结果既不支持“GRU 已提点”，也不支持“StereoGRU 架构被否定”。首要几何实现故障已定位；校准可学习性和第一对可比训练已完成；现在进行一次固定的重复性验证：**只追加原计划 seed1/2，保持全部数据/crop/模型/监督/预算/最后一步比较点，逐 seed C0 完成后才 C1。** 不继续改路线、不追加类似只读探针。

配对重复与汇总已作为[新独立入口](scripts/stereogru_matched_repeats.py)实现，规格见[匹配计划第 9–10 节](STEREOGRU_MATCHED_CONTROL_PLAN.md)。它不修改原训练器，不写旧 seed0 artifact；三个种子全部完成才汇总均值/样本标准差及配对差，仍只描述一个开发场景的 seed 变异，不声称统计显著性或零样本 SOTA。seed1/2 的真实结果尚待阿里云执行。