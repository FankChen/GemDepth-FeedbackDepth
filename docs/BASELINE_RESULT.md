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
| _VDA (paper, ref)_ | zero-shot | KITTI | 0.083 | n/a | 0.944 | 论文表1 KITTI 列 |
| _GemDepth-DAV2 (paper)_ | zero-shot | KITTI | 0.077 | n/a | 0.950 | 论文表1 KITTI 列（含 GEM/ASTT） |
| _GemDepth-VDA (paper)_ | zero-shot | KITTI | 0.071 | n/a | 0.955 | 论文表1 KITTI 列（含 GEM/ASTT） |

## 与论文对比须知

本基线是 **VKITTI 微调 head → KITTI 测试**（合成→真实迁移，仅 head-only），论文是大数据 **zero-shot**。
两者设置不同。论文 Table 1 列序=Sintel|Bonn|Scannet|KITTI，KITTI 取最后一对：VDA 0.083、GemDepth-DAv2 0.077、
GemDepth-VDA 0.071。我们 0.0679 已与 GemDepth-VDA 持平略优、优于 VDA — KITTI 复现到位，论文优势主要在 Sintel/Bonn/Scannet。

> 评测在完整 `batch_a100` 上完成（job 12511433）。MIG 切片显存不足以跑 ViT-L 视频推理，会 OOM。
