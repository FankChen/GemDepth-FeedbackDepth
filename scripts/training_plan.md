# GemDepth 训练复现计划 (2026-06-03 更新)

## 当前状态

- ✅ VKITTI 1.3.1: **已就绪** (`/home/izi2sgh/MYDATA/quanjie/liren/datasets/vkitti_1.3.1/`)
- ❌ TartanAir: **未下载**
- ✅ 预训练权重: `checkpoint/gemdepth.pth` (2.7GB)
- ✅ eval 数据集: KITTI ✅ | Sintel ❌ | Bonn ❌ | ScanNet ❌

## 两线并行策略

### 线一：最小可行训练（VKITTI only，立即开始）

| Step | 操作 | 脚本 |
|------|------|------|
| 1 | Smoke test 确认 VKITTI loader 可用 | `smoke_train.py` |
| 2 | 提交 8×H200 训练 (50k steps, VKITTI) | `train_stage1_vkitti_only.bsub` |
| 3 | 监控 loss，判断收敛性 | `bjobs` + TensorBoard |

### 线二：基础设施（并行推进）

| Step | 操作 | 脚本 |
|------|------|------|
| 1 | 提交 TartanAir 下载 job | `download_tartanair.bsub` |
| 2 | 提交 Sintel/Bonn/ScanNet eval | `eval_gemdepth_{sintel,bonn,scannet}.bsub` |
| 3 | 读 GemDepth 论文确认 baseline 方案 | - |

## 新功能说明

### Resume 机制
- `train.py` 自动检测 checkpoint 目录中最新 checkpoint
- 保存 optimizer/scheduler/step 状态
- 保留最近 3 个中间 checkpoint + 最终 `final.pth` + `final_model.pth`
- 使用 `resume=true`（自动）或 `resume=/path/to/checkpoint_XXX.pth`（指定）

### 训练脚本
```bash
# VKITTI only (立即开始)
bsub < scripts/train_stage1_vkitti_only.bsub

# VKITTI + TartanAir (等 TartanAir 下完)
bsub < scripts/train_stage1.bsub

# 续训
bsub < scripts/train_stage1_vkitti_only.bsub -- -resume

# Stage2
bsub < scripts/train_stage2.bsub

# Eval
bsub < scripts/eval_gemdepth_sintel.bsub
bsub < scripts/eval_gemdepth_bonn.bsub
bsub < scripts/eval_gemdepth_scannet.bsub
```

### 数据加载器扩展
当前 `dataset_mix.py` 支持: `vkitti`, `TartanAir`
待扩展: `PointOdyssey`, `DynamicReplica`
