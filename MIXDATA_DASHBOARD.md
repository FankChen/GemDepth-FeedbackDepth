# Mixdata 多数据集混训 — 结果看板

- 更新日期：2026-07-31
- 配置：`config/scratch_ed_dinov3convnext_ms_mixdata_8gpu.yaml`
- checkpoint：`checkpoint/scratch_ed_dinov3convnext_ms_mixdata_8gpu/final.pth`（step 20000，训完于 2026-07-30 18:01）
- 训练环境：阿里云 DSW，8×H20，`accelerate --num_processes 8 --mixed_precision bf16`，global batch=32（4/GPU），seq_len=4，crop=448，total_step=20000
- 评测对齐：视差（inverse depth）空间 lstsq scale+shift（与 KITTI/temporal 完全一致），**不取倒数**

---

## 一、TL;DR

ConvNeXt-Small + LoRA 多尺度头（native-res + depth-feedback + video loss），在 **6 数据集混训**（VKITTI2 + TartanAir + VKITTI1.3.1 + MVS-Synth + PointOdyssey + Dynamic Replica）下训到 20k 步，训练全程稳定（自动跳过 9 个损坏样本、无崩溃）。四个视频深度 benchmark 全部评测完成：bonn 最好（AbsRel 0.0809 / δ1 0.9516），sintel 最难（0.3693）。

⚠ **对照说明**：本表是 6 数据集混训配方，与历史单训 VKITTI 系的 temporal / multiscale 基线**不是同一训练配方**（不同训练数据 / split / 预算），下方历史基线仅作参考，**不能据此宣称"混训更好/更差"**。公平结论需跑同 mix 配方的 temporal baseline（见待办）。

---

## 二、混训结果

### 2.1 四集 benchmark（final.pth，`output_eval/mixdata_final`）

| dataset | AbsRel↓ | RMSE↓ | δ1↑ | 备注 |
|---|---|---|---|---|
| kitti | 0.1119 | 3.7667 | 0.8940 | 室外真实 LiDAR 稀疏 GT |
| sintel | 0.3693 | 6.6069 | 0.5497 | 合成电影、非刚体运动，公认最难 |
| **bonn** | **0.0809** | 0.2320 | **0.9516** | 室内 RGBD，最好；po/dr 室内数据助力 |
| scannet | 0.1211 | 0.3227 | 0.8613 | 室内 RGBD |

### 2.2 VKITTI2 域内 dense（`eval_vkitti_dense.py`，final.pth，不 invert）

| checkpoint | AbsRel↓ | RMSE↓ | δ1↑ | 说明 |
|---|---|---|---|---|
| step1500（早期） | 0.1270 | 8.3378 | 0.8189 | 训练早期 |
| **final（step20000）** | **0.0915** | **6.2360** | **0.9045** | AbsRel 降 28%，δ1 0.819→0.905 |

---

## 三、历史基线（仅参考，非同配方）

> 均为单训 VKITTI 系、20k 步、KITTI `kitti_video.json` 同评测口径。**训练数据/split 与本次混训不同，不能直接对比。**

| 方案 | KITTI AbsRel↓ | 说明 |
|---|---|---|
| ConvNeXt-s temporal | 0.0891 | 历史最佳 temporal |
| ConvNeXt-s multiscale-C（video loss） | 0.0946 | 单训 multiscale |
| ConvNeXt-s E2a（native+feedback） | 0.0915 | fullres/feedback 系最佳 |
| **本次 mixdata multiscale（final）** | **0.1119** | 6 数据集混训（非同配方） |

---

## 四、待办 / 下一步

- [ ] **同口径 temporal baseline**：用同 6 数据集 mix 配方训一版 temporal，四集 benchmark 同脚本评测，才能公平回答"混训 multiscale vs temporal"。
- [ ] tar.gz（bonn 7.3G / scannet 22G / sintel 2.1G）解压后可删腾空间（共享存储，删前确认）。

---

## 五、复现命令

```bash
cd /mnt/workspace/liren/PP-DPT
CONFIG=config/scratch_ed_dinov3convnext_ms_mixdata_8gpu.yaml \
CKPT=checkpoint/scratch_ed_dinov3convnext_ms_mixdata_8gpu/final.pth \
BENCH=/mnt/workspace/gemdepth_eval \
OUT=output_eval/mixdata_final \
DATASETS="kitti sintel bonn scannet" \
GPU=0 \
bash scripts/eval_gemdepth_4bench.sh
```

VKITTI2 域内 dense：
```bash
python evaluation/inference/eval_vkitti_dense.py \
  --config config/scratch_ed_dinov3convnext_ms_mixdata_8gpu.yaml \
  --ckpt   checkpoint/scratch_ed_dinov3convnext_ms_mixdata_8gpu/final.pth \
  --vkitti_root /mnt/workspace/vkitti/vkitti/ \
  --out_viz runlogs/viz/vkitti_mixdata_final.png
```
