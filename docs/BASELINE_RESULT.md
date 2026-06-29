# GemDepth Baseline — Reproduction Result

纯原版 GemDepth（temporal DPT head）的受控基线：冻结主干，仅微调 DPT head，在 VKITTI 训练、KITTI 评测。
作为后续 error-map 改进的起点参照。

## 设置

- **Head:** `temporal`（原版 DPT 头，零改动；加载预训练 `missing=0 / unexpected=0`，精确复现）
- **起点权重:** `./checkpoint/gemdepth.pth`（作者预训练，主干含训练好的 GEM/ASTT）
- **训练范围:** `freeze_mode: head_only` — DINOv2 + GEM + ASTT 全冻结，仅 ~87M DPT head 可训
- **数据:** VKITTI 2.0.3（训）/ KITTI Eigen video split（测）
- **步数:** 10000，batch 1 × grad_accum 4，crop 518，seq_len 16，AdamW lr 1e-4

## 结果

| Model | Train | Test | AbsRel ↓ | RMSE ↓ | δ<1.25 ↑ | Notes |
|---|---|---|---|---|---|---|
| **baseline (ours)** | VKITTI | KITTI | **0.0679** | **3.167** | **0.957** | head-only finetune，非全量复现 |
| _VDA (paper, ref)_ | zero-shot | KITTI | 0.071 | n/a | 0.959 | 同设备参考 |
| _GemDepth-DAV2 (paper)_ | zero-shot | KITTI | 0.055 | n/a | 0.970 | 论文上限（含 GEM/ASTT 全训） |
| _GemDepth-VDA (paper)_ | zero-shot | KITTI | 0.051 | n/a | 0.978 | 论文上限（含 GEM/ASTT 全训） |

## 与论文对比须知

本基线是 **VKITTI 微调 head → KITTI 测试**（合成→真实迁移，仅 head-only），论文是大数据 **zero-shot**。
两者设置不同，论文 KITTI 值仅作上限参考，不直接对标。我们 0.0679/0.957 与 VDA 基线持平，距 GemDepth
本体 0.051~0.055 的差距即 GEM/ASTT 全量训练的增益。

> 评测在完整 `batch_a100` 上完成（job 12511433）。MIG 切片显存不足以跑 ViT-L 视频推理，会 OOM。
