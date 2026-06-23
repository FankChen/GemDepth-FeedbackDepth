# GemDepth Reproduction Progress

## 1. 复现目标

当前目标是在 `/home/izi2sgh/MYDATA/quanjie/liren/depth_baselines/GemDepth` 内复现 GemDepth 的训练与评估流程，并形成可重复执行的集群脚本。复现范围包括：

- 搭建并验证 GemDepth 运行环境。
- 下载和整理训练数据，优先级为 VKITTI 1.3.1 与 TartanAir。
- 跑通训练入口 `train.py`，包括 stage1 / stage2、checkpoint、resume、日志记录。
- 跑通推理和评估脚本，至少覆盖 KITTI evaluation。
- 分析单卡 A100 下的最小可行复现方案。
- 作为后续 DA-V2 / DenseGRU 视频深度方案的参考基线。

## 2. 对论文和代码的理解

GemDepth 面向视频深度估计，核心目标不是单帧深度，而是跨帧 3D 一致性。方法包括两个关键设计：

- Geometry-Embedding Module, GEM：显式预测帧间相机位姿，生成几何嵌入，将相机运动和全局 3D 结构先验注入视频深度网络。
- Alternating Spatio-Temporal Transformer, ASTT：交替建模空间和时间特征，捕获潜在点级对应关系，提升细节恢复和时间一致性。

当前仓库训练入口为：

```bash
accelerate launch train.py --config-name stage1
accelerate launch train.py --config-name stage2
```

默认训练设定接近：

- encoder: `vitl`
- crop size: `518`
- seq len: `32`
- global batch size: `8`
- precision: `bf16`
- stage1 / stage2: 各 `100000` steps
- pretrained checkpoint: `checkpoint/gemdepth.pth`

## 3. 当前环境状态

已完成环境检查，结果如下：

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| 仓库路径 | 已确认 | `/home/izi2sgh/MYDATA/quanjie/liren/depth_baselines/GemDepth` |
| Python | 已确认 | `Python 3.10.20` |
| PyTorch | 已确认 | `torch=2.3.1+cu121` |
| CUDA | 已确认 | GPU 节点上 `cuda=True` |
| flash-attn | 已确认 | `flash-attn=2.6.3` |
| 预训练权重 | 已确认 | `checkpoint/gemdepth.pth`, 约 2.7GB |
| accelerate | 已安装 | 训练入口依赖 |
| tensorboard | 已安装 | 训练日志依赖 |

已有环境检查日志：

```text
jobs/check_env.12440210.stdout
jobs/check_env.12440210.stderr
```

环境结论：GemDepth 的 Python / CUDA / PyTorch / flash-attn 基础环境可用，当前主要阻塞点是训练数据。

## 4. 数据集进展

### 4.1 VKITTI 1.3.1

GemDepth 当前 `dataset/dataset_mix.py` 支持 `vkitti`，并期望目录结构为：

```text
/home/izi2sgh/MYDATA/quanjie/liren/datasets/vkitti_1.3.1/
├── vkitti_1.3.1_rgb/
├── vkitti_1.3.1_depthgt/
└── vkitti_1.3.1_extrinsicsgt/
```

已经确认旧脚本中配置的 VKITTI 路径下缺少上述三个目录，因此训练数据尚未就绪。

尝试过的下载方式：

1. Google Drive mirror 下载：失败。原先 `scripts/download_vkitti.py` 使用 Google Drive / gdown 链接，但链接需要特殊权限或已失效。
2. Naver Labs 官方直链下载：脚本已改好，但集群计算节点访问失败。

官方链接包括：

```text
https://download.europe.naverlabs.com/virtual-kitti-1.3.1/vkitti%5F1.3.1%5Frgb.tar
https://download.europe.naverlabs.com/virtual-kitti-1.3.1/vkitti%5F1.3.1%5Fdepthgt.tar
https://download.europe.naverlabs.com/virtual-kitti-1.3.1/vkitti%5F1.3.1%5Fextrinsicsgt.tar.gz
```

问题：
集群计算节点无法通过代理访问 `download.europe.naverlabs.com`

- 在登录节点测试是否可以直接下载小文件 `extrinsicsgt`。
- 在本地机器下载三个官方 archive 后上传到集群。
- 寻找可出网节点或内部镜像。

相关文档已更新：

```text
VKITTI_SETUP.md
```

### 4.2 TartanAir

README 中列出 TartanAir 为训练数据之一，当前仓库 loader 也支持 `tartanair`。已有脚本：

```text
scripts/download_tartanair.bsub
```

但目前尚未确认 TartanAir 数据已经下载完成。此前讨论中判断：当前训练数据优先可走 VKITTI；若 VKITTI 下载受阻，可改为 TartanAir 最小训练路径。

### 4.3 KITTI Evaluation

已有 KITTI 相关评估日志与脚本痕迹：

```text
jobs/gemdepth_infer_kitti.12427107.stdout
jobs/gemdepth_infer_kitti.12427362.stdout
scripts/eval_gemdepth_kitti.bsub
```

KITTI 更适合作为 evaluation 数据，不作为当前训练主路径。

## 5. 已做的代码和脚本修改

### 5.1 VKITTI 下载脚本改造

修改文件：

```text
scripts/download_vkitti.py
scripts/download_vkitti.bsub
VKITTI_SETUP.md
```

改动内容：

- 删除或绕开原有 Google Drive / gdown 下载逻辑。
- 改用 Naver Labs 官方 URL。
- 支持 `wget -c` 或 `curl -C -` 断点续传。
- 将 archive 保存到：

```text
/home/izi2sgh/MYDATA/quanjie/liren/datasets/vkitti_1.3.1/archives/
```

- 自动解压到 GemDepth loader 期望目录。
- 增加 `--verify-only` 用于只验证目录结构。
- `scripts/download_vkitti.bsub` 队列从 GPU 队列改为 CPU 队列：

```text
batch_h100 -> batch_cpu
```

- `VKITTI_SETUP.md` 更新为官方直链下载、手动上传、解压和验证说明。

当前下载脚本本身已经是合理的，但受集群网络代理限制，计算节点无法完成官方下载。

### 5.2 训练 checkpoint / resume 支持

当前 `train.py` 已包含 checkpoint 相关逻辑：

- `find_latest_ckpt(checkpoint_dir)`：寻找最新 `checkpoint_<step>.pth`。
- `load_checkpoint(...)`：恢复 model / optimizer / scheduler / total_step。
- `save_checkpoint(...)`：保存完整训练状态。
- 中间 checkpoint 只保留最近 3 个。
- final 保存：

```text
final.pth
final_model.pth
```

注意：`final.pth` 是完整训练状态，`final_model.pth` 是 model-only 权重。后续 stage2 如果用 `model.load_state_dict()` 初始化，更适合指向 `final_model.pth`。

### 5.3 训练脚本准备

已有集群脚本：

```text
scripts/smoke_train.bsub
scripts/train_stage1_vkitti_only.bsub
scripts/train_stage1.bsub
scripts/train_stage2.bsub
scripts/check_env.bsub
```

其中：

- `smoke_train.bsub`：用于单 GPU smoke test。
- `train_stage1_vkitti_only.bsub`：VKITTI-only 最小训练脚本，计划 8 GPU / 50k steps。
- `train_stage1.bsub`：VKITTI + TartanAir 混合训练脚本。
- `train_stage2.bsub`：stage2 续训脚本。
- `check_env.bsub`：环境检查脚本。

当前尚未正式启动训练，原因是训练数据未就绪。

## 6. 当前发现的问题和风险

### 6.1 配置问题

`config/stage1.yaml` 当前存在结构问题：文件开头直接是：

```yaml
  seed: 0
  checkpoint_dir: ./checkpoint/gemdepth_stage1
```

但 `train.py` 使用：

```python
cfg.training.seed
cfg.training.checkpoint_dir
```

因此 stage1 配置应补成：

```yaml
training:
  seed: 0
  checkpoint_dir: ./checkpoint/gemdepth_stage1
```

`config/stage2.yaml` 已有 `training:` 顶层字段，stage1 需要对齐。

### 6.2 单卡 batch 逻辑问题

`train.py` 当前逻辑是：

```python
per_gpu_batch = cfg.dataloader.batch_size // world_size
```

默认 `batch_size=8`。在 8 GPU 时每卡 batch=1；但单卡时每卡 batch=8，容易 OOM。

因此单卡训练应显式设置：

```text
dataloader.batch_size=1
```

如果想保持 effective batch 8，需要实现 gradient accumulation，而当前训练代码尚未实现。

### 6.3 stage2 初始化权重路径风险

`train_stage2.bsub` 当前设计中 stage2 可能指向：

```text
checkpoint/gemdepth_stage1/final.pth
```

但 `final.pth` 包含 optimizer / scheduler / total_step 等完整状态，不是纯 model state dict。如果通过 `model.load_state_dict()` 加载，应使用：

```text
checkpoint/gemdepth_stage1/final_model.pth
```

### 6.4 数据复现风险

论文 README 列出多个训练数据集：

- TartanAir
- VKITTI / VKITTI2
- PointOdyssey
- MVS-Synth
- Dynamic Replica
- IRS

但当前代码 `dataset/dataset_mix.py` 直接支持的主要是：

- `vkitti`
- `TartanAir`

因此当前复现更接近“开源代码中可执行的 VKITTI / TartanAir 子集复现”，不是完整论文混合数据复现。

### 6.5 预训练权重语义风险

当前训练默认从：

```text
checkpoint/gemdepth.pth
```

加载权重。如果该权重是作者发布的最终 GemDepth 模型，那么从它继续训练属于 fine-tuning / training pipeline verification，而不是从论文初始预训练状态重新训练。实验记录中需要明确这一点。

## 7. 单卡 A100 复现方案设计

如果只用单卡 A100，建议将目标定义为“单卡可运行复现 / 方法验证”，而不是严格复现论文 8 卡训练。

### 7.1 目标分级

| 级别 | 目标 | 说明 |
| --- | --- | --- |
| Level 1 | Smoke test | 合成数据，验证 forward / loss / backward / optimizer |
| Level 2 | Real-data overfit | 少量真实序列，验证 loader、pose、depth、loss 是否正确 |
| Level 3 | Single-A100 training | 使用 VKITTI 或 TartanAir 子集进行 10k-50k steps |
| Level 4 | Paper-scale reproduction | 需要多卡、完整混合数据和更严格配置 |

### 7.2 推荐单卡配置

初始建议：

```text
GPU: 1 x A100
encoder: vitl
precision: bf16
crop_size: 518
seq_len: 8
micro_batch_size: 1
gradient_accumulation_steps: 8
effective_batch_size: 8
total_optimizer_steps: 10000 first, then 50000
checkpoint frequency: 1000
decoder LR: 1e-5
other trainable LR: 1e-6
weight decay: 0.01
grad clip: 1.0
```

如果显存不足，调整顺序建议：

1. 保持 `crop_size=518`，先降低 `seq_len` 到 4。
2. 使用 micro-batch 1 + gradient accumulation。
3. 尝试 activation checkpointing。
4. 最后再降低 crop size 到 `448` 或 `392`。

### 7.3 单卡课程式训练

建议顺序：

1. `seq_len=4`，跑 1k-5k steps，验证稳定性。
2. `seq_len=8`，跑 10k-50k steps。
3. `seq_len=16`，如果显存允许，从上一步 checkpoint 继续。
4. `seq_len=32`，仅在 A100 80GB 且显存允许时做短程微调。

### 7.4 需要补的代码能力

为了让单卡实验合理，需要补：

- 修复 `config/stage1.yaml` 的 `training:` 顶层字段。
- 在 `train.py` 中实现 gradient accumulation。
- 新增单卡配置，例如 `config/stage1_a100.yaml`。
- 新增单卡提交脚本，例如 `scripts/train_stage1_a100_single.bsub`。
- 修正 stage2 初始化权重为 `final_model.pth`。

## 8. 当前可执行命令

### 8.1 验证环境

```bash
cd /home/izi2sgh/MYDATA/quanjie/liren/depth_baselines/GemDepth
bsub < scripts/check_env.bsub
```

### 8.2 尝试 VKITTI 官方下载

如果集群网络可访问官方站点：

```bash
cd /home/izi2sgh/MYDATA/quanjie/liren/depth_baselines/GemDepth
bsub < scripts/download_vkitti.bsub
```

但已知当前计算节点会报 proxy 403。

### 8.3 手动上传 VKITTI archive 后验证

将三个 archive 放到：

```text
/home/izi2sgh/MYDATA/quanjie/liren/datasets/vkitti_1.3.1/archives/
```

然后执行：

```bash
cd /home/izi2sgh/MYDATA/quanjie/liren/depth_baselines/GemDepth
/home/izi2sgh/MYDATA/quanjie/liren/envs/gemdepth/bin/python scripts/download_vkitti.py \
  --target-dir /home/izi2sgh/MYDATA/quanjie/liren/datasets/vkitti_1.3.1 \
  --keep-archives
```

只验证结构：

```bash
/home/izi2sgh/MYDATA/quanjie/liren/envs/gemdepth/bin/python scripts/download_vkitti.py \
  --target-dir /home/izi2sgh/MYDATA/quanjie/liren/datasets/vkitti_1.3.1 \
  --verify-only
```

### 8.4 数据就绪后的训练

VKITTI 数据就绪后：

```bash
cd /home/izi2sgh/MYDATA/quanjie/liren/depth_baselines/GemDepth
bsub < scripts/smoke_train.bsub
bsub < scripts/train_stage1_vkitti_only.bsub
```

断点续训：

```bash
bsub < scripts/train_stage1_vkitti_only.bsub -resume
```

## 9. 下一步建议

优先级从高到低：

1. 解决 VKITTI 数据获取：本地下载三个官方 archive 并上传，或找到可访问 Naver 的节点。
2. 修复 `config/stage1.yaml` 顶层 `training:` 字段。
3. 新增单卡 A100 训练配置和 bsub 脚本。
4. 给 `train.py` 增加 gradient accumulation。
5. 用合成数据 smoke test 验证训练链路。
6. 用少量 VKITTI/TartanAir 做真实数据 overfit test。
7. 再启动较长的单卡或多卡 stage1 训练。

## 10. 当前结论

GemDepth 的环境和预训练权重已经具备，训练入口和集群脚本也基本准备好。当前主要阻塞是训练数据，尤其是 VKITTI 1.3.1 的下载受集群网络代理限制。方法层面已经完成了从 Google Drive 镜像到 Naver 官方下载脚本的改造，并明确了单卡 A100 的可行复现设计。下一步应优先完成数据落盘，然后再进入 smoke train、overfit test 和正式训练。
