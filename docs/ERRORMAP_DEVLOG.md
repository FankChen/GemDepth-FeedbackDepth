# GemDepth Error-Map Co-Attention — Development Log

This document tracks the **error-map feedback** line of work on GemDepth: injecting cross-frame
reprojection-error signals into the DPT decoder so the head can refine depth where the current
geometry is temporally inconsistent.

- **SOTA table** (top): the running leaderboard of every trained model version.
- **Timeline** (below): one entry per update — what changed, the effect (improvement / decline),
  and the conclusion, with model version / training data / test data / metric.

> **Workflow rule:** every change to the repository lands via a **pull request** (no direct pushes
> to `main`). Each PR should add a timeline entry here and, once evaluated, a SOTA-table row.

---

## SOTA Table

Training data: **VKITTI 2.0.3** (frozen DINOv2 ViT-L backbone, only the DPT head is fine-tuned).
Test data: **KITTI** (Eigen video split). Lower is better for AbsRel / RMSE; higher is better for δ<1.25.

> **与论文对比须注意：** 本项目是「VKITTI 微调 head → KITTI 测试」（合成→真实迁移，且只训 head），而论文是大数据
> **zero-shot**。论文 Table 1 列序 = Sintel|Bonn|Scannet|**KITTI**，KITTI(500帧)选最后一对。我们预训 zero-shot 0.0677
> 已与 GemDepth-VDA(0.071) 持平略优、优于 VDA(0.083)；KITTI 上复现到位，论文优势主要在 Sintel/Bonn/Scannet。

| Model version | Head | Error modality | Train | Test | AbsRel ↓ | RMSE ↓ | δ<1.25 ↑ | Notes |
|---|---|---|---|---|---|---|---|---|
| pretrained (ours, zero-shot) | temporal (original) | — | — | KITTI | 0.0677 | 3.106 | 0.9585 | 预训练权重直接评测，未微调（本协议锚点） |
| **baseline (ours)** | temporal (original) | — | VKITTI | KITTI | **0.0679** | **3.167** | **0.957** | 初始基线；head-only 微调 10k 步（≈预训练，无退化） |
| _VDA (paper, ref)_ | — | — | zero-shot | KITTI | 0.083 | n/a | 0.944 | 论文表1 KITTI 列 |
| _GemDepth-DAV2 (paper)_ | — | — | zero-shot | KITTI | 0.077 | n/a | 0.950 | 论文表1 KITTI 列(含 GEM/ASTT) |
| _GemDepth-VDA (paper)_ | — | — | zero-shot | KITTI | 0.071 | n/a | 0.955 | 论文表1 KITTI 列(含 GEM/ASTT) |
| errormap-v1 | errormap (additive) | RGB | VKITTI | KITTI | _pending_ | _pending_ | _pending_ | zero-init additive injection |
| coattn-rgb | errormap_coattn (方案C) | RGB | VKITTI | KITTI | _pending_ | _pending_ | _pending_ | — |
| coattn-feat | errormap_coattn (方案C) | feature | VKITTI | KITTI | _pending_ | _pending_ | _pending_ | — |
| coattn-hog | errormap_coattn (方案C) | HOG | VKITTI | KITTI | _pending_ | _pending_ | _pending_ | — |
| coattn-rgbfeat | errormap_coattn (方案C) | RGB + feature | VKITTI | KITTI | _pending_ | _pending_ | _pending_ | — |

_Update each row's metrics after running `scripts/eval_kitti_arm.sh <arm>`._

---

## Timeline

### 2026-06-29 — baseline 微调复现，首个初始 SOTA 行

- **Model version:** `baseline` (head_type `temporal`)。从预训 `gemdepth.pth` head-only 微调 10k 步。
- **Train:** VKITTI 2.0.3；**Test:** KITTI Eigen video split。
- **Result:** AbsRel **0.0679** / RMSE **3.167** / δ<1.25 **0.957**（batch_a100 评测，job 12511433）。
- **与论文对比（注意 train/test 不同）:** 我们=VKITTI微调head→KITTI；论文=zero-shot。论文 KITTI: VDA 0.083、
  GemDepth-DAv2 0.077、GemDepth-VDA 0.071。我们 0.0679 已与 GemDepth-VDA 持平略优、优于 VDA — KITTI 复现到位，
  论文优势主要在 Sintel/Bonn/Scannet（待补测）— 作为所有 error-map 实验起点。
- **Conclusion:** baseline 复现到位，作为初始参照；后续 error-map 臂均在此基线上递增。

### 2026-06-25 — 方案C: bidirectional co-attention error-map head + 4 controlled arms

- **Model version:** `coattn-{rgb,feat,hog,rgbfeat}` (head_type `errormap_coattn`).
- **Training data:** VKITTI 2.0.3 (head-only fine-tune from pretrained `gemdepth.pth`).
- **Test data:** KITTI Eigen video split.
- **Metric:** AbsRel / RMSE / δ<1.25 (pending training).
- **Change:**
  - Generalised the reprojection warp into `signal_error_map` (any per-pixel signal), keeping
    `photometric_error_map` as a thin wrapper.
  - Added `model/util/hog.py` (dense soft-binned HOG descriptor) as a warpable signal.
  - New head `DPTHeadErrorMapCoAttn` (`model/dpt_errormap_coattn.py`): per error modality, build a
    reprojection-error map → encode to tokens → **symmetric co-attention** between the decoder
    "anchor" stream and every error stream → zero-init projection added back to the stage feature
    at s4/s3/s2. Identity at init (reproduces the baseline temporal head).
  - Four arms differ **only** in `model.error_modalities` ∈ {`rgb`, `feat`, `hog`, `rgbfeat`} — a
    clean controlled comparison of which error source the co-attention should consume.
  - Wired `head_type=errormap_coattn` + `error_modalities` through `gemdepth.py`, `train.py`,
    `evaluation/inference/infer.py`; added 4 configs and extended the train/eval launchers.
  - CPU smoke (`scripts/smoke_coattn.py`) passes for all four arms: identity-at-init
    (max_diff=0) and gradient flow through co-attention + modality encoders.
- **Effect:** _pending training/eval._
- **Conclusion:** _pending — to be filled once the four arms finish and are evaluated against the
  `baseline` and `errormap-v1` controls._

### (prior) — errormap-v1: additive RGB error-map head

- **Model version:** `errormap-v1` (head_type `errormap`).
- **Change:** per-stage coarse depth → RGB photometric error map → zero-init conv encoder → **added**
  into the decoder feature at s4/s3/s2. Simple additive injection (no attention).
- **Effect / Conclusion:** baseline for the co-attention work; metrics pending.

### (prior) — baseline: original temporal DPT head

- **Model version:** `baseline` (head_type `temporal`).
- **Change:** none (control). Frozen DINOv2 ViT-L backbone + original temporal DPT head.
- **Effect / Conclusion:** reference point for all error-map experiments.
