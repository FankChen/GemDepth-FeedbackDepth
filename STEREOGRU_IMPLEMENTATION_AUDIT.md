# StereoGRU：实现审计、历史结果重判与下一步取证

日期：2026-09-09。审计基线：`feat/registry-mixdata@eb2c9ac`。

**修正此前判断：先证明实现和几何/监督契约成立，再评价 StereoGRU；当前负结果不能否定这类方法。** 多篇论文验证了校准条件下的迭代匹配，但也不保证任意视频移植自动有效。本次不重训、不替换历史 head、不改变历史训练配置。

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

FoV 激活是 ReLU，0 会在 `f=(W/2)/tan(FoV/2)` 中变成 inf；[warp](model/util/warp.py) 将非有限投影移到图外。这避免 NaN，却可能使某些帧没有几何证据。**finite loss / 无 NaN 不是代价体有效性证明。** 旧真实 checkpoint 的非有限 K 比例、有效投影比例需阿里云取证；不能用随机小模型的 inf 比例冒充训练后结果。

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

### P0-E：GRU 第二路 lookup 接错了，但整个 volume 并未断梯度

[forward](model/dpt_cost_volume_convnext.py)当前调用 `VolumeIndexer(logits, softmax(logits))`。官方第二路保留 raw correlation，让迭代仍能直接查局部匹配证据。这里把两路都建立在正则化 logits 上。

不过 raw volume **仍通过 3D aggregation 和 hidden initialization 影响输出**；最终 loss 也能反传到 matcher。小型模型实测 matcher gradient norm≈`.00850`。所以“初值 loss 高 → volume 什么都没学到 → 全靠 GRU”的归因不成立；必须分别观察 raw/logits/index/update。

## 4. 哪些实现已通过，不能错怪

[专门的回归测试](test/test_stereogru_contracts.py)加上[原 smoke](test/test_cost_volume_head.py)和[诊断测试](test/test_diagnostics.py)验证了：

- `E_src @ inv(E_ref)` 的 world-to-camera 相对方向、旋转/非中心 K/非单位参考相机的投影。
- inverse-depth 轴 lookup、batch/H/W 展平与双层 pooling、亚 bin 插值。
- 三层 GRU 的 gate 形式、跨尺度更新顺序与官方一致；每轮 detach index 也来自官方，不是误切梯度。
- normalized inverse-depth index 的 convex upsample **不应该额外乘 stride**；不能照搬像素 disparity 的 stride 缩放。
- matcher 到最终输出有有限非零梯度，历史相机 detach 行为保留。
- pose dropout 改的是 ASTT 用的 embeddings，不改供 cost-volume 的原始 `extrinsic/intrinsic`。`use_astt=false` 的这条 arm 不能再归咎于那次 eval dropout bug。

本次 focused suite：**32 passed（13.76s）**，含 objective 回归和不依赖真实权重/GPU 的八路诊断 CLI 测试。未跑耗时全套，未在本机运行真实 costvol checkpoint；本地未找到这些 run 的目录，历史权重在阿里云。

## 5. 诊断自身已修的错误

原 [oracle](scripts/diagnose_cost_volume_oracle.py)有三处会误导结果的逻辑，现已修复并加测试：

1. 原先 rank=`1+count(score>true_score)`，全相等的 D-bin 体被算作 top1=100%。现在采用随机平局打破的期望命中率：32-bin 平体应为 **1/32**，并报告 unique winner、chance、margin。
2. true bin 改为**逆深度距离**最近，而非米制距离；GT 超出 sample bounds 单列，不伪装为边界 bin 命中。
3. `score_std` 改为每个像素沿 depth 轴的变化；不把不同像素的纹理差异当深度分辨力。

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