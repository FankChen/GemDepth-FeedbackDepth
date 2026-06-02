# GemDepth Reproduction Progress

## 1. 复现目标

将 GemDepth 的训练和推理复现工作整理成可执行流程，重点包括：

- `GemDepth` 代码环境搭建与依赖验证
- 数据集准备与下载（TartanAir、VKITTI、KITTI、PointOdyssey 等）
- 训练配置和训练阶段复现（stage1 / stage2）
- 评估与推理验证（含 KITTI / Sintel / Bonn / ScanNet）
- 与 DA-V2 / DenseGRU 视频深度方案的并行对比设计

## 2. 当前已完成的工作

- 已确认本地 `liren/depth_baselines/GemDepth` 仓库存在，并且可进入。
- 已检查本地仓库当前分支为 `main`，远程 `origin` 指向 `https://github.com/Yuecheng919/GemDepth.git`。
- 已确认 `GemDepth` 原仓库代码可用，缺失的改动主要是临时文件和新增内容。
- 已完成以下复现准备性工作：
  - FlashAttention 环境验证、可用于 GemDepth 的训练性能优化。
  - KITTI Eigen 评估集数据路径确认，并确定当前重点为 KITTI / TartanAir 组合复现。
  - 发现 GemDepth 数据集 Loader 目前已支持 `vkitti` 与 `tartanair`，但 VKITTI 下载/访问仍有不稳定问题。
  - 确认可以使用 `TartanAir` 作为 GemDepth 复现的最小可行数据集路径。

## 3. 仍需完成的工作

### 3.1 环境与依赖

- 安装并固定 `python=3.10` 环境。
- 安装 `requirements.txt` 中依赖，并验证 `accelerate`、`torch`、`flash-attn` 等可用。
- 检查 `config/stage1.yaml` 和 `config/stage2.yaml` 的 GPU/内存配置是否适配当前硬件。

### 3.2 数据集准备

- 下载并解压 `TartanAir` 数据集。
- 如果条件允许，下载 `VKITTI` 并完成 `gemdepth` 训练所需的预处理。
- 准备 `KITTI Eigen` 数据集用于评估，不作为训练主路径。
- 如果需要进一步提升复现稳定性，可补充 `PointOdyssey` / `Dynamic Replica` / `MVS-Synth` 数据。

### 3.3 训练与验证

- 复现 stage1: 100k steps，batch_size 8，seq_len 16/32。
- 复现 stage2: 100k steps，继续训练并恢复前一阶段检查点。
- 基于 `TartanAir` 的最小可行训练路径进行 overfit / smoke test。
- 运行推理验证代码，确认 `evaluation/inference/run_video.py` 等脚本可正常执行。

### 3.4 对齐 DA-V2 + DenseGRU 工作

- 将 GemDepth 作为参考基线，记录与 DA-V2 + DenseGRU 的差异点。
- 将本地 `liren/DenseGRU-v1.0` 的设计与 GemDepth 的训练数据、评估指标对齐。
- 形成一个统一的对比实验目录，便于后续 PR 里说明“GemDepth 复现 vs DA-V2 主线”。

## 4. 当前进展状态

| 项目 | 进度 |
| --- | --- |
| 本地仓库准备 | 已完成 |
| 环境验证 | 已完成（含 flash-attn） |
| 数据集 Loader 检查 | 已完成 |
| TartanAir 下载启动 | 已启动 |
| VKITTI 下载可用性 | 进行中 / 可能需要备用方案 |
| KITTI Eigen 评估准备 | 已确认 |
| 训练脚本对齐 | 需要补充 |

## 5. 近期下一步

1. 在本地建立 PR 分支，提交本次 GemDepth 复现进度文档。
2. 如果你希望，我可以继续准备：
   - `GEMDepth-training` 仓库的 README 模板
   - 从 `GemDepth` 代码到你的 `glone` fork 的 PR 说明草案
   - 训练/评估的 `bsub` / `slurm` 提交脚本

## 6. 备注

- 当前操作已在本地完成文档整理，但未直接推送到 GitHub 远程仓库。
- 如果你希望，我可以继续帮你生成精确的 `git push` / `PR` 命令，供你在安全的环境中执行。
