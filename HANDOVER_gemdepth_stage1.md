# GemDepth Stage-1 复现 · 交接说明书

**更新：2026-08-19　分支：`feat/registry-mixdata`　状态：Stage-1 复现已成功，待扩展**

> 本文件用于对话上下文迁移，是一份可独立继承的完整情况说明。

---

## 0. 一句话现状

**Stage-1 复现已完成且成功**（对标论文 Table 4，KITTI +5.1% / Sintel +7.1%，均在 7% 内）。当前卡点不是"复现不出来"，而是目标错位——之前一直拿完整两阶段的 Table 1 当靶子。**下一步应转向 errmap 方法实验**，以本次 checkpoint 为对照基线。

---

## 1. 环境与访问约束（务必先读）

| 项 | 值 |
|---|---|
| **Bosch 工作树**（AI 能读写） | `/home/izi2sgh/MYDATA/mycode/PP-DPT` |
| Bosch 登录节点 | `rng-dl01-login{4,6,7}.de.bosch.com` |
| Python 环境 | `/home/izi2sgh/MYDATA/quanjie/liren/envs/gemdepth/bin/python` |
| **阿里云代码**（训练在此） | `/mnt/data/PROJECT_CHEN/code/PP-DPT` |
| 阿里云数据 | `/mnt/data/PROJECT_CHEN/data/{train,eval}` |
| 阿里云权重 | `/mnt/data/PROJECT_CHEN/weights/` |
| 阿里云硬件 | 8×H20（143771 MiB/卡） |

**硬约束（踩过坑）：**

- ⚠ **AI 无阿里云 SSH/终端权限**，所有阿里云命令必须用户手动执行。
- ⚠ 阿里云 ttyd 网页终端**粘贴多行会拼行吞字符** → 只能一行一条命令。
- ⚠ `/mnt/workspace` 和 `/mnt/data` 在 Bosch **不挂载**。
- ⚠ 长任务一律 `nohup` + 日志文件，用户已两次因前台中断丢失工作。
- ⚠ **禁用 `| tail -N`**（全缓冲看着像卡死）→ 用 `python -u` 或写日志 `tail -f`。
- ⚠ 远程编辑大文件（如 `train.py`）会严重卡顿 → 优先 `sed -i` / 一次性批量改完。

**Git 远端：**

```
origin   = Bosch GHES (github.boschdevcloud.com/IZI2SGH/PP-DPT)  # 阿里云拉不到
feedback = https://github.com/FankChen/GemDepth-FeedbackDepth     # 公开，阿里云的 origin
```

阿里云安全取单个文件（不动 checkpoint 软链）：

```bash
git fetch origin feat/registry-mixdata && git checkout FETCH_HEAD -- scripts/
```

---

## 2. 复现目标与已达成结果

### 2.1 正确对标 = 论文 Table 4（Stage-1 / 20k steps）

| 配置 | KITTI AbsRel | Sintel AbsRel | ScanNet TAE |
|---|---|---|---|
| Baseline (VDA) | 0.092 | 0.356 | 0.621 |
| + Spatial only | 0.082 | 0.337 | 0.609 |
| + Temporal only | 0.084 | 0.343 | 0.573 |
| + ASTT | 0.080 | 0.328 | 0.566 |
| **Full (ASTT+GEM)** ← 目标 | **0.074** | **0.295** | **0.538** |
| **本次复现** | **0.0778 (+5.1%)** | **0.3160 (+7.1%)** | 未测 |

**判定：复现成功。** 关键在于**相对位置正确**——同时优于 Baseline 和 +ASTT 两档，说明 ASTT 增益与 GEM 的额外增益都拿到了，不是只复现一半。

### 2.2 四集完整结果

`output_eval/gemdepth_stage1_repro_all`

| 数据集 | AbsRel | RMSE | δ1 |
|---|---|---|---|
| KITTI | 0.0778 | 3.3647 | 0.9497 |
| Sintel | 0.3160 | 6.5260 | 0.6286 |
| Bonn | 0.0513 | 0.1756 | 0.9734 |
| ScanNet | 0.0676 | 0.2000 | 0.9644 |

### 2.3 与我们自己历史实验对比（同评测口径，干净可比）

| 方案 | Sintel | KITTI | Bonn | ScanNet |
|---|---|---|---|---|
| ConvNeXt-S temporal mixdata | 0.3771 | 0.0961 | 0.0781 | 0.1165 |
| ConvNeXt-S multiscale mixdata | 0.3693 | 0.1119 | 0.0809 | 0.1211 |
| ConvNeXt-S temporal pr8（数据被砍） | 0.3594 | 0.1102 | 0.0801 | 0.1176 |
| **GemDepth-VDA 复现** | **0.3160** | **0.0778** | **0.0513** | **0.0676** |

**四集全面碾压**（ScanNet −42% / Bonn −34% / KITTI −19%）。→ 印证核心诊断：**差距大头是骨干先验（DINOv2 ViT-L）+ 真实数据蒸馏（VDA 初始化），不是解码头结构**。之前在 ConvNeXt-Small + 纯合成数据上反复调 head（multiscale / feedback / fullres 等十余组消融）始终碰不到这条天花板。

### 2.4 训练收敛

| 损失项 | 起始 | 末期 | 变化 |
|---|---|---|---|
| ssi | 0.20–0.26 | 0.022–0.071 | ↓75% |
| stable_loss | 0.071–0.178 | 0.013–0.022 | ↓85% |
| **pose_loss** | 0.72–0.55 | 0.10–0.27 | ↓75% |
| **quat（旋转）** | 0.30 | 0.004 | **↓98%** |
| trans（平移） | 0.42 | 0.10 | ↓65% |

→ **GEM 的位姿监督真实有效**，不是挂着不动。

---

## 3. ⚠️ 三条诚实偏差（写任何结论时必须标注）

1. **评测帧数不同**：我们 four-bench json = KITTI~110 / Sintel~100 / Bonn~110 / ScanNet~90 帧；论文 = Bonn·ScanNet·KITTI 各 **500 帧**、Sintel 50 帧。帧数少对视频方法**系统性偏乐观**（漂移积累少、仿射对齐在短序列拟合更好）。
   → **绝不能声称"超过论文 SOTA"**，尽管 Bonn 0.0513 < 论文 0.066、ScanNet 0.0676 < 论文 0.071。
2. **Stage-1 数据配比不同**（详见 §5）。
3. **无 Stage-2**（缺 IRS）。

---

## 4. 🔴 重大订正（2026-08-19）

之前记录的「Stage-2 需要 150K **野外立体视频**、论文没公开来源、**无法复现**」——**这是错的，已撤回。**

查证官方 GitHub（`Yuecheng919/GemDepth`）README 的 `✏️ Training Data` 章节，**只列 7 个数据集，全部公开可下载**：

> TartanAir · **VKITTI (1.3.1)** · VKITTI2 · PointOdyssey · MVS-Synth · Dynamic Replica · **IRS**

- **没有任何"野外立体视频"数据集。**
- IRS 有公开仓库 <https://github.com/HKBU-HPML/IRS>（不是只有被墙的 OneDrive）。
- 👉 **结论逆转：Stage-2 唯一缺 IRS，Table 1 完整两阶段复现是可行的。**
- 👉 另：官方训练数据**含 VKITTI 1.3.1**，而我们 `stage1_repro` 特意排除了它（手上已有 7,179 clips），属偏差，应加回。

**官方 README 其它有用信息：**

- `--json_file` 可选 `sintel_video.json` / `kitti_video_500.json` / **`scannet_video_tae.json`**；`--datasets` 只取基名 `sintel/kitti/bonn/scannet`，**协议由 json 决定**。
- 评测章节标题 `## ~500frame` → **500 帧才是论文默认协议**。
- 训练即 `accelerate launch train.py --config-name stage1` / `stage2`。
- 推理默认 INFER_LEN=32 / OVERLAP=10 需约 **44GB 显存**；可降到 16/6 (25GB) 或 8/4 (15GB)。
- 论文为 **ICML 2026**，构建于 VideoDepthAnything + VGGT + DepthAnythingV2。

**TAE 现状**：`evaluation/eval/eval.py:229` 只有一处 `mvs_synth/mvssynth_video_tae.json` 路径引用，**TAE 指标本身未实现**（eval.py 只算 AbsRel/RMSE/δ1），需自己写。

---

## 5. Stage-1 数据配比（已修正，有效权重 = clips × RATIO）

RATIO：`mvs_synth=26`、`vkitti1=15`、其余 `=1`；总计约 834,172 clips。

| 数据集 | clips | RATIO | 有效权重 | 我们 % | 论文 % | 偏差 |
|---|---|---|---|---|---|---|
| TartanAir | ~295K | 1 | 295,196 | 35.4% | 43.5% | 接近 ✓ |
| PointOdyssey | 301,594 | 1 | 301,594 | 36.2% | 10.1% | **3.6× 过量** |
| MVS-Synth | 8,280 | 26 | 215,280 | 25.8% | 11.6% | 2.2× 过量 |
| VKITTI2 | 11,342 | 1 | 11,342 | 1.4% | 5.8% | 4.3× 不足 |
| Dynamic Replica | 10,760 | 1 | 10,760 | 1.3% | 29.0% | **22× 不足** |

论文 Stage-1 帧数：TartanAir 300K + Dynamic Replica 200K + MVS-Synth 80K + PointOdyssey 70K + VKITTI2 40K = **690K**。

**根因**：Dynamic Replica 的 **train split（1.8TB）下载时被跳过**，只取了 `valid`。

⚠️ **看板 `assets/gemdepth_stage1_repro_dashboard.html` §六 的 "PointOdyssey ~90%" 是错的**（未乘 RATIO 算出的），应替换为上表。

**已设计但未实现**：在 `DepthVideoDataset.__init__` 加 `mix_weights={label: n}` 机制 + `_resample_clips(clips, target, rng)` 静态方法（短则重复、长则确定性下采样），使 config 可直接写论文帧数。

---

## 6. 关键配置与代码改动（均已 push）

### 6.1 `config/stages/stage1_repro.yaml`

```yaml
# @package _global_        ← 必须！否则 Hydra 把配置嵌套进目录包
backbone: DINOv2Backbone
decoder: DPTHeadTemporal
backbone_weights: "timm://vit_large_patch14_dinov2.lvd142m"   # ⚠ 见下
video_path: /mnt/data/PROJECT_CHEN/weights/video_depth_anything_vitl.pth
use_gem: true
use_astt: true
use_temporal: true
lora: false
pose_flag: true
crop_size: 518
seq_len: 32
batch_size: 8        # global，1/GPU
grad_accum: 2        # 有效 batch 16 = 论文
dec_lr: 1.0e-4       # 新模块
other_lr: 1.0e-6     # 预训练 head
vkitti_scene_split: false
total_step: 20000
```

- ⚠ `DINOv2Backbone` 在 `weights=null` 时是**随机初始化**（不像 DINOv3 ConvNeXt 自动 timm pretrained），必须显式给；随后被 VDA 的 `pretrained.*` 覆盖 = 论文真正起点。
- ⚠ `encoder_decoder_only` 与 GEM/ASTT **互斥**，不能设。

### 6.2 `train.py` 三处修复

1. **`NEW_MODULE_PREFIXES` 常量**（放在 `DATASET_MAX_DEPTH_DEFAULT` 之后）
   - ASTT：`spatial_blocks` `time_blocks` `dec_norm`
   - GEM：`global_blocks` `frame_blocks` `camera_token` `register_token` `camera_head` `cam_rot_encoder` `cam_trans_encoder` `cam_trans_scale_encoder`
   - → 全部归入 `dec_lr`。实测切分：新模块 **319.04M @1e-4** / 预训练 head **87.37M @1e-6**。
2. **checkpoint 守卫**：`ckpt_has_new_modules = any(k.startswith(NEW_MODULE_PREFIXES) for k in checkpoint)`；为 False 则打印 "depth-baseline init (VDA / DAv2)" 并设 `allow_extra=True`。`new_key_tags` 扩为 `*NEW_MODULE_PREFIXES, 'pos_encoder', 'temporal_transformer.layer_norm', 'fusion_gate1', 'head.proj.'`。
3. **DDP**：`grad_accum > 1` → `DistributedDataParallelKwargs(static_graph=False, find_unused_parameters=True)`；`== 1` 保持 `static_graph=True`。同时 `head.proj` **无条件冻结**（原先只在 `encoder_decoder_only` 下冻）→ decoder 可训练 87.372M → 84.486M。

### 6.3 `dataset/dataset_mix.py`

新增 `vkitti_scene_split=True` 参数（默认保持师兄行为）；`if self.vkitti_scene_split and not scene_is_selected(...)`；强制打印 `vkitti (2.0.3) scene_split={}` + clip 数。

### 6.4 `scripts/eval_gemdepth_4bench.sh`（commit `c338f98`）

```bash
for ds in $DATASETS; do
    base="${ds%_500}"
    if [ "$ds" != "$base" ]; then json="$BENCH/$base/${base}_video_500.json"
    else json="$BENCH/$base/${base}_video.json"; fi
    ... --datasets "$base" ...
done
# eval 阶段仍传 $DATASETS（带 _500），让 eval.py 取 max_eval_len=500
```

---

## 7. 复现过程中修掉的 6 个坑

| # | 问题 | 根因 | 修法 |
|---|---|---|---|
| 1 | `Key 'total_step' is not in struct` | config 移入子目录后缺 `# @package _global_` | 补上该行 |
| 2 | GEM 训不动 | 只有 ASTT 走 `dec_lr`，GEM 落在 1e-6 | `NEW_MODULE_PREFIXES` |
| 3 | 加载 VDA 触发"零 missing"断言 | 守卫不认识基线初始化 | 从 checkpoint 自身探测 |
| 4 | 8 卡首反向即崩 `expect_autograd_hooks_` (c10d/reducer.cpp:1634) | `static_graph=True` 与 `accelerator.accumulate()` 的 `no_sync()` 不兼容；**此前所有 8 卡实验都是 grad_accum=1 故未暴露，单卡 smoke 无 DDP 也不报** | grad_accum>1 改 `find_unused_parameters` |
| 5 | DDP 拒绝未使用参数 | legacy `head.proj` 可训练却永不收梯度 | 无条件冻结 |
| 6 | KITTI 莫名退化 14.7% | PR#8 的 `scene_is_selected` 把 VKITTI2 从 ~21k **静默砍到 1010 clips**（3 场景×1 variation），且该分支不打印 clip 数 | 加开关 + 强制打印 |

**其它教训**：`results.txt` 是**追加写**（同集跑两次出两行，实测 0.0778 vs 0.0780 = bf16 推理噪声）；8 个 DDP 进程各扫一遍数据集，启动时几分钟静默正常。

---

## 8. VDA 权重兼容性（已验证）

- VDA key 前缀 = `['head', 'pretrained']`，与本仓库**天然一致**（本仓库就是 VDA 代码基长出来的）→ **不需要映射脚本**。
- 实测 **519 张量全部匹配**（pretrained 343 + head 176），**零 shape 错、零 unused**。
- MISSING 248 = GEM + ASTT + head 24。其中 head 24 = 16 个 `head.motion_modules.*.temporal_transformer.{layer_norm, fusion_gate1}`（GemDepth 相对 VDA 的**真实结构增量**，在活跃路径上）+ 8 个 legacy `head.proj.*`（未使用）。
- DAv2 是 `['depth_head', 'pretrained']`，需改名。

---

## 9. 论文参考数据（权威转写自 `assets/gemdepth_paper_study.html`）

**Table 1**（列序 **Sintel / KITTI / Bonn / ScanNet**，AbsRel，完整两阶段）

| 方法 | Sintel | KITTI | Bonn | ScanNet | 延迟 |
|---|---|---|---|---|---|
| DAv2 | 0.390 | 0.127 | 0.150 | 0.137 | 79ms |
| **VDA**（我们的起点） | 0.295 | 0.071 | 0.089 | 0.083 | 85ms |
| GemDepth-DAv2 | 0.188 | 0.055 | 0.069 | 0.077 | 94ms |
| **GemDepth-VDA (SOTA)** | 0.157 | 0.051 | 0.066 | 0.071 | 99ms |

⚠️ **列序易错**：正确为 Sintel/KITTI/Bonn/ScanNet。`assets/mixdata_dashboard.html` 与 `MIXDATA_DASHBOARD.md` §二**当前写错了**（写成 Sintel/Bonn/ScanNet/KITTI），需回退修复。

**其它表：**

- Table 2（TAE）：DAv2 1.14 / RollingDepth 0.65 / VDA 0.57 / GemDepth-DAv2 0.50 / GemDepth-VDA 0.47
- Table 5：20% 位姿噪声 → 0.066→0.071
- Table 7（ASTT 位置）：Late 0.126/0.377/0.917，Mid 0.102/0.352/0.737，**Early 0.088/0.328/0.654**
- Table 9：50% 位姿注入 ≈ 100% 效果（0.071 vs 0.070）

**训练超参**：ViT-L，clip N=32，base 518，AdamW 判别式 LR，16×A800 batch 16，每 stage ≈3 天。
Loss = `L_ssi + 0.5·L_gm + 10·L_tgm + 0.2·L_cam`（前三项 = PP-DPT `VideoDepthLoss`，`L_cam` 由 `pose_flag: true` 启用）。

---

## 10. 复现命令

### 训练（8 卡，约 38 小时）

```bash
cd /mnt/data/PROJECT_CHEN/code/PP-DPT
OPENCV_IO_ENABLE_OPENEXR=1 NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_DEBUG=WARN \
nohup python -u -m accelerate.commands.launch --num_processes 8 --mixed_precision bf16 \
  train.py --config-name=stages/stage1_repro > runlogs/stage1_repro.log 2>&1 &
```

### 四集评测

```bash
cd /mnt/data/PROJECT_CHEN/code/PP-DPT
OPENCV_IO_ENABLE_OPENEXR=1 \
CONFIG=config/stages/stage1_repro.yaml \
CKPT=checkpoint/gemdepth_stage1_repro/final.pth \
BENCH=/mnt/data/PROJECT_CHEN/data/eval \
OUT=output_eval/gemdepth_stage1_repro_all \
DATASETS="kitti sintel bonn scannet" GPU=0 \
bash scripts/eval_gemdepth_4bench.sh
```

### 权重下载

```bash
mkdir -p /mnt/data/PROJECT_CHEN/weights && cd /mnt/data/PROJECT_CHEN/weights
wget -c -O video_depth_anything_vitl.pth \
  "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth?download=true"
```

**产物**：`checkpoint/gemdepth_stage1_repro/{final.pth 5.83G, final_model.pth 2.84G}`，20000 步耗时 **37h55m**（3.41–3.47 s/it）。单卡纯前向+反向 peak **62.5GB**。

---

## 11. 待办（优先级排序）

### P0 — 立即可做，不需重训

1. **跑 500 帧评测**（脚本已修，用**全新输出目录**——旧 `gemdepth_stage1_repro_500/kitti/` 有 1064 个中断残留 npy）

   ```bash
   OUT=output_eval/gd500 DATASETS="kitti_500 sintel bonn_500 scannet_500"
   ```

   验证日志出现 `(json=.../kitti_video_500.json, npy dir=.../kitti)`。
   ⚠ **2026-08-17 那次 `_500` 评测因该 bug 三个集全被静默跳过、结果无效。**

### P1 — 核心工作

2. **接入 errmap 方法**。`DPTHeadErrorMap` 已在 decoder registry 注册；`gemdepth.py` 有"探测 forward 签名决定是否传几何输入"的机制（`decoder_requires_geometry_inputs`），errmap 头正好需要 `images/extrinsics/intrinsics`，接入成本低。
   **对照组 = 我们自己的 `stage1_repro` checkpoint**（同数据/预算/代码/评测，唯一变量 errmap），**不是论文数字**。

### P2 — 加强 baseline

3. **下载 IRS** → 解锁 Stage-2 → 解锁 Table 1 完整复现。
4. **Table 4 消融行**：
   - Baseline（`use_gem=false, use_astt=false`）— 纯 config，38h
   - +ASTT（`use_astt=true, use_gem=false`）— 纯 config，38h
   - ⚠ **+Spatial only / +Temporal only 需要改代码**——`model/gemdepth.py` 中 spatial 与 temporal 是成对循环、无开关：

     ```python
     for blk1, blk2 in zip(self.spatial_blocks, self.time_blocks):
         feats[m] = blk1(feats[m], pos[m]); ...; feats[m] = blk2(feats[m])
     ```

5. **TAE 指标实现**：eval.py 未实现，需自写；官方用 `scannet_video_tae.json`。
6. **数据配比修正**：`mix_weights` 机制 + 补 DR train split（1.8TB）+ VKITTI2 第二相机（论文 40K ≈ 双相机，我们只用 Camera_0）+ 加回 VKITTI 1.3.1。

### P3 — 文档修正

7. `gemdepth_stage1_repro_dashboard.html` §六 "PointOdyssey ~90%" → 换成 §5 的 RATIO 修正表。
8. 同文件 §六 "Stage-2：+250K 帧（IRS + 野外立体视频）" → 按 §4 订正为"仅缺 IRS"。
9. `mixdata_dashboard.html` / `MIXDATA_DASHBOARD.md` §二 Table 1 列序错误 → 回退修复。

---

## 12. 给下一轮对话的战略建议

1. **承认 baseline 已经有了**。`stage1_repro` 就是可用基线，不要继续追 Table 1 的数字。
2. **errmap 的科学论证不依赖追平论文**。只要"有 errmap vs 无 errmap"在同配方下对照干净，结论就成立。反之，即使追平了 Table 1，errmap 的增益仍需这组对照来证明。
3. **实验对照硬规则**（用户长期偏好）：任何新协议/split/评测器/模型族，必须先跑同协议 baseline 再启 method arm；不匹配的旧结果只能参考，不能宣称提升。
4. **沟通风格**：全部中文；结果诚实记录不美化；师兄希望用户少依赖 AI 自己 debug，AI 做 coach（讲方法、指路），代码尽量用户动手。
