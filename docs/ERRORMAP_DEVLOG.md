# Error-Map Single-Head 实验记录 (DEVLOG)

## 对照协议（严格公平）
所有 arm 使用**完全相同**的训练协议，唯一变量是 DPT head 结构 / error-map warp signal：
- 预训练权重 `checkpoint/gemdepth.pth`
- `freeze_mode = head_only`（冻结 backbone，只训 head）
- 数据 VKITTI 2.0.3（50 序列）
- `total_step = 10000`

测试：KITTI Eigen（`infer.py --head_type errormap_single --warp_signal <sig>` → `eval.py` lstsq scale/shift 对齐）。指标 AbsRel↓ / RMSE↓ / δ1↑。

> **temporal baseline 与 em_single 各臂是同协议、同数据、同步数，仅 head 结构不同 → 严格可比。**
> em_single 方向要算成功，必须**追平或超过 temporal baseline 的 AbsRel 0.0677**。

## 结果表（KITTI, AbsRel↓ / RMSE↓ / δ1↑）

| 版本 | head | warp signal | AbsRel | RMSE | δ1 | 训练环境 | 备注 |
|---|---|---|---|---|---|---|---|
| **temporal baseline** | temporal | — | **0.0677** | 3.167 | 0.957 | Bosch j12511433 / 阿里云 lr1e-5 | **对照基准** |
| em_single rgb | errormap_single | rgb | 0.0883 | 4.098 | 0.9230 | 阿里云 DSW | 劣于 baseline ~0.021 |
| em_single feat | errormap_single | feat | **0.0847** | 4.010 | 0.9278 | 阿里云 DSW | 4 臂 AbsRel 最优，仍劣于 baseline |
| em_single rgbfeat | errormap_single | rgbfeat | 0.0863 | **3.757** | **0.9300** | 阿里云 DSW | RMSE/δ1 最优臂 |
| em_single hog | errormap_single | hog | 0.0873 | 4.075 | 0.9232 | 阿里云 DSW | ⚠ descriptor/可视化审计发现缺陷，结果保留但需 caveat |

## Zero-shot vs finetune（同环境阿里云，2026-07-09）
gemdepth.pth 原始权重直接测（无任何训练）:
- KITTI  zero-shot: **0.0678** / 3.110 / 0.9585 ；finetune(head_only VKITTI lr1e-5): 0.0676 / 3.167 / 0.9571
- VKITTI zero-shot: **0.0669** / 4.869 / 0.9512 ；finetune(baseline): 0.0690 / 4.606 / 0.9516
→ **head_only VKITTI finetune 对 baseline 基本无正收益**：KITTI AbsRel +0.4%(RMSE/δ1 反降)，VKITTI AbsRel **−3.2%**(同域反而变差)。
→ **真正天花板 = zero-shot 0.0678(KITTI) / 0.0669(VKITTI)**。error-map 4 臂(0.084+)距此更远。问题一半在 finetune 协议(setting)，非只结构 → 后续既改结构(BAT+Lin)也要改 finetune 数据/协议。

## Baseline LR sweep（temporal head，KITTI，阿里云复现）
证明环境/数据/评测链路可靠 + 定最优 lr（对照基准即取 lr1e-5）。

| lr | AbsRel | RMSE | δ1 | 备注 |
|---|---|---|---|---|
| 1e-4 | 0.0712 | 3.327 | 0.9499 | 偏大 |
| 5e-5 | 0.0691 | 3.249 | 0.9542 | |
| **1e-5** | **0.0676** | 3.167 | 0.9571 | **最优，≈ Bosch 0.0677** |
| 1e-6 | 0.0681 | 3.158 | 0.9573 | |

lr1e-5 中间步（早收敛，10k 略过训）：step6000 **0.06736** / 8000 0.06758 / 10000 0.06767。
→ baseline ~6000 步即达最优；反观 error-map 各臂训满才最好（em_feat step6000→final：0.0866→0.0847）。

## 时间线
- **2026-07-02**：temporal baseline 4 数据集复现（KITTI 0.0677/3.167/0.957 ≈ 论文）。em_single rgb 在阿里云启动训练。
- **2026-07-07**：KITTI 测试集搬到阿里云；em_single rgb eval = 0.0883/4.098/0.923，劣于 baseline。
- **2026-07-09**：4 臂 error-map + baseline LR sweep 全部 eval 完成（KITTI）。
  - baseline lr1e-5 复现 **0.0676** ≈ Bosch → 环境/数据/协议可靠，问题不在环境。
  - error-map 4 臂 AbsRel 0.0847~0.0883，**全劣于 baseline 0.0677**；AbsRel 最优 em_feat，RMSE/δ1 最优 em_rgbfeat。
  - 训练 loss 持续降但 KITTI eval 差 → **loss/eval 脱钩**，疑似 VKITTI→KITTI 域迁移过拟合。

## 待办 / 下一步
- [x] feat / rgbfeat / hog 训完 eval，回填结果表。
- [ ] 诊断 loss/eval 脱钩：① VKITTI held-out eval（若 error-map 在合成域好、真实域差 → 坐实域迁移过拟合）；② KITTI 上 error-map 可视化（看残差是否被动态/光照/位姿噪声主导）。
- [ ] 结构改进：误差注入位置 / 融合方式 / aux-loss 权重 / refine 深度 / 训练步数。
- [ ] 目标：errormap_single 追平并超过 temporal baseline 0.0677。

---

## GT-camera oracle 五臂结果与协议审计（2026-07-16）

独立实验分支：`exp/gt-error-channels`；DINOv2 ViT-L + LoRA、T=4、GEM/ASTT 关闭、训练排除 Scene20、测试仅 Scene20。GT K / world-to-camera extrinsics 只用于 oracle warp，GT depth 只监督独立 metric-depth branch，不进入 error-feedback 输入。实际统一评估 2088 clips / 8352 frames。

| arm | AbsRel↓ | RMSE↓ | δ1↑ | 备注 |
|---|---:|---:|---:|---|
| Multiscale-only baseline | 0.444656 | 15.126310 | 0.288045 | 无 metric/error-feedback 分支 |
| RGB | 0.248834 | 11.669141 | **0.574748** | δ1 与 Feature 近乎并列 |
| **Feature** | **0.233081** | **8.612631** | 0.574575 | 当前完整架构最优 |
| RGB+Feature | 0.303622 | 12.370620 | 0.516473 | 不及两个单流 |
| Geometry | 0.462033 | 15.347888 | 0.259086 | 劣于 baseline |

### Baseline 定义与跨表比较边界

- 此处 baseline 是 **multiscale-only baseline**：已有多尺度 DPT refine，只去掉 metric-depth/error-feedback 分支；不是最简单 temporal DPT。
- from-scratch 表 7 的 `0.0832` 来自 temporal head、全 VKITTI 训练（包含 Scene20）、KITTI Eigen 测试；此处 `0.444656` 来自排除 Scene20 训练、Scene20-only 测试。head / train split / test domain / evaluator 都不同，数字不能直接横比。
- 现阶段无法给出“multiscale 相对简单 temporal baseline”的 Scene20 净增益。必须补同协议的 Temporal DPT（B0）。

### 47.6% 增益的正确表述

Feature arm 相对 multiscale-only baseline：AbsRel 降 47.6%、RMSE 降 43.1%、δ1 提升 28.65 个百分点。但 Feature arm 同时增加 metric-depth heads、p2 error encoder、p2→p1 feedback 和 p1 correction；尤其 p1 correction 同时读取 p1 feature，即使 error 为零也有额外修正容量。因此当前只能写：

> 完整 Feature-feedback architecture 相对 multiscale-only baseline 提升 47.6%。

不能把全部增益写成 Feature error 的净贡献。需补：

1. **B0 Temporal DPT**：同 Scene20-held-out 协议的真正简单 baseline；
2. **B1 Multiscale-only**：当前 baseline；
3. **B2 Null-error capacity control**：保留 metric heads / encoder / correction，但 error 恒为零；
4. **M Feature feedback**：相对 B2 测 Feature error 的净贡献。

### RGB+Feature 实现与待验证假设

当前 p2/p1 实现：RGB residual 与 L2-normalized DPT feature residual 分别计算、分别按当前帧 valid 区域均值归一化，再直接组成 `[E_rgb, E_feat, validity]`，输入同一个 `Conv(3→64→256)` error encoder；没有 learned gating、独立 modality encoder、confidence 或 disagreement handling。

实现上存在三类合理假设：

- 独立归一化把两个流强制到相近量级，抹掉了“哪种信号更可信”的尺度信息；
- 两流共享 metric depth / GT camera / warp grid，可能高度相关而非互补，RGB 还引入纹理/光照残差；
- 共用小 encoder 可能在融合前就让两流互相干扰，联合 validity 的边界语义也需核查。

以上都只是待验证假设，不能仅凭最终指标当作因果结论。下一步不重训即可：从同一个 RGB+Feature checkpoint、固定同一个 Scene20 clip 导出 p2/p1 的两个 residual slot、validity、metric depth、p2 feedback norm、p1 signed correction 和相对 B1 的 improvement map；全 Scene20 统计 Pearson/Spearman、top-10% overlap、saturation、valid coverage、与真实 depth error 的相关性、correction 改善/恶化像素比例。

### HOG descriptor / 可视化审计结论

HOG 是旧 `em_single` 的手工梯度方向特征，**不是** GT-camera 五臂里的 Geometry（后者是预测 metric depth 的跨帧几何一致性误差）。当前 HOG 图“基本呈现错误”有代码依据：

1. `sqrt(gx²+gy²+eps)` 给零梯度平坦区制造非零 magnitude；逐像素 L2 normalization 随后放大底噪。`eps` 应只放 normalization denominator。
2. 可视化用归一化 HOG 各 bin 之和控制亮度、`argmax bin` 控制 hue；这既不是真实 Sobel magnitude，也不是经典 cell/bin line-glyph HOG。低梯度区仍被强制分配颜色方向。
3. 训练直接对 ImageNet-normalized 模型输入算 HOG；可视化先反归一化到 `[0,1]` 后再算，两者不是同一 signal。
4. `cell=8` 偶数 pooling kernel + padding=4 产生 H+1/W+1，再插值回原尺寸，存在轻微空间偏移风险。

因此旧 HOG 可视化不能作为正确性证据，旧 HOG 指标保留但必须标 caveat。若继续 HOG：统一训练/可视化输入；去掉 magnitude floor；低 magnitude 区屏蔽 orientation；使用原始 magnitude + 经典 HOG glyph；修复后重新训练。

### 下一步顺序（先诊断，再 Pose CNN）

- [ ] 补 B0 same-split Temporal baseline。
- [ ] 补 B2 null-error capacity control。
- [ ] 导出 p2/p1 RGB / Feature / RGB+Feature / Geometry error、validity 与 correction。
- [ ] 跑全 Scene20 residual/correction 统计。
- [ ] 根据实证决定 gating/独立 encoder/normalization 修改。
- [ ] 固定被验证的 Feature 方案后，再把 GT extrinsics 替换为简单 Pose CNN。

### Sample0 逐层可视化实证（2026-07-16）

输入：Scene20 / 15-deg-left / frame 0–3，展示 frame 1；同一 RGB+Feature checkpoint 做 normal / no-RGB / no-Feature / zero-residuals 推理干预。证据图已归档到 `docs/assets/gt_error_visualization_sample0/`。

确定性观察（仅该 sample）：

- p2 metric depth 近似常数，但 warp validity=97.8%；p1 metric depth 近乎归零，p1 RGB/Feature residual 与 validity 全零（validity=0）。
- p2 Feature raw residual mean 仅 `3.56e-5`，normalized map 主要由矩形边框支配；当前按自身均值归一化存在放大 padding/warp 数值伪影的风险。
- normal / no-RGB / no-Feature / zero-residuals 的 SSI 指标完全相同到打印精度（AbsRel 均 `0.2135786414`）。这只证明 residual 值未改变该 clip 的 SSI 可评估深度形状；validity 仍保留，且 SSI 会消除全局 affine 差异，不能扩大成“raw 输出逐位相同”。
- p1 correction 前后 AbsRel `0.213576→0.213579`，无改善；p1 error 三通道全零时 correction 仍非零，说明该支路可退化为 feature-only correction。
- p4→p1 累计深度逐渐变成低频平滑梯度，没有明显恢复汽车/树木等几何细节，但在上游 metric geometry 失效前，不先改 DPT 以免混杂。

模块定位顺序：

1. **先修 metric-depth head/loss**：当前 `log(clamp(pred,1e-3))` 对低于阈值的 Softplus 输出产生零梯度死区，是 p1 低值塌缩的首要代码候选。先移除死区并增加 p1/p2 metric range、gradient、validity 健康检查。
2. **再验证 warp 核心，不先重写**：同一投影代码在 p2 有高 validity，解析平面测试也通过；修复 metric depth 后，用 GT depth 替换做同 clip A/B，确认投影/相机约定。
3. **再修 error normalization**：避免以极小均值将边界伪影放大；考虑 absolute floor / robust percentile / confidence。
4. **再审计 fusion/correction**：补 validity-only、三通道全零和 raw-output max-diff，区分 residual、validity、bias 与纯 affine 作用。
5. **最后才动 multiscale DPT**。

因此当前首改模块已经收敛为 **metric-depth branch / supervision**，不是先改 warp 矩阵，也不是先改 DPT。

### 修复后最终结果（2026-07-17）

同一 Scene20-held-out 协议（2088 clips / 8352 frames）：

| 方案 | AbsRel↓ | RMSE↓ | δ1↑ |
|---|---:|---:|---:|
| Multiscale-only B1 | 0.444656 | 15.126310 | 0.288045 |
| 旧 RGB+Feature | 0.303622 | 12.370620 | 0.516473 |
| Metric log-depth fix only | 0.229364 | 10.451446 | 0.609936 |
| **Baseline-anchored DPT + Warp v2** | **0.084935** | **4.958671** | **0.897651** |

- Metric-only fix 相对 B1：AbsRel −48.4%、RMSE −30.9%、δ1 +32.19pp；证明 metric branch clamp dead zone/塌缩是旧实现的重要根因。
- 完整 v2 相对 B1：AbsRel −80.9%、RMSE −67.2%、δ1 +60.96pp；相对 metric-only fix，AbsRel 再降 63.0%。
- v2 最终 p4/p3/p2/p1 validity = 0.792/0.850/0.874/0.889，floor fraction 全 0；旧 p1 validity=0 故障已消失。
- 该增益是 baseline anchor、共享 readout/gauge、feedback placement、pixel-center warp、occlusion/border、cosine/fixed error scale 的组合结果，不能拆成单模块贡献。仍缺 same-split Temporal B0、null-error control 和多 seed。
- 最终 sample0 四层 relative error：p4 `0.23119` → p3 `0.22088` → p2 `0.22093` → p1 `0.22105`。几何/对应图已健康，但只有 p4→p3 明显改善；p2/p1 平台化并轻微反弹，所以“每层都提高”尚未成立。

### Scratch T=4 修正协议重评（2026-07-17）

- 根因：原 inference entry 没有把 config 的训练 `seq_len` 传给已经支持 `clip_len` 的模型接口；T=4 模型被按原 GemDepth 32 帧 overlap + 跨窗 affine 推理。`bc354cc` 修复为 non-overlap `clip_len=4`，并增加推理 finite fail-fast 与稳定中心化 SSI 对齐。
- DINOv2 Temporal T4：旧错误协议 `0.083218/3.407811/0.940155` → 修正协议 **`0.080037/3.352053/0.944425`**，AbsRel 改善 3.82%。
- DINOv2 Multiscale T4：训练已完整到 20k；修正协议最终 **`0.384146/11.539853/0.326224`**。该结果 finite 且 evaluator 正常完成，因此是有效的负结果，不再是“待评估”。
- 同配方下 Multiscale 相对 Temporal：AbsRel 4.80×、RMSE 3.44×、δ1 −61.82pp；旧 multiscale 结构明确失败，但尚不能从最终指标把失败归因于单一模块。
- 三个静态 backbone 的旧指标仍经过默认 32 帧 entry；虽无 temporal module，跨窗 affine 可能改变输出，需后续用 `clip_len=1` 重评后再锁定静态排序。
