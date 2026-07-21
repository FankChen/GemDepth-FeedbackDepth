# Exp7 交接文档：DINOv3（ViT-S+ / ConvNeXt）× 时序 / Multiscale

> 分支：`exp/scratch-ed-backbones`
> 内容：4 个从零训练（encoder+decoder-only）实验 —— 时序 head 与 multiscale head，各在 DINOv3 ViT-S+ 和 DINOv3 ConvNeXt-s 上跑一遍。
> Backbone 冻结 + LoRA 微调，DPT 头从零训练，数据 VKITTI。

---

## 1. 四个实验一览

| 实验 | backbone | head_type | 配置文件 | aux_depth |
|---|---|---|---|---|
| ViT-S+ · 时序 | `dinov3_vitsplus` | temporal | `config/scratch_ed_dinov3vits_temporal.yaml` | 0.0 |
| ViT-S+ · multiscale | `dinov3_vitsplus` | multiscale | `config/scratch_ed_dinov3vits_multiscale.yaml` | 1.0（disparity） |
| ConvNeXt-s · 时序 | `dinov3_convnext_s` | temporal | `config/scratch_ed_dinov3convnext_temporal.yaml` | 0.0 |
| ConvNeXt-s · multiscale | `dinov3_convnext_s` | multiscale | `config/scratch_ed_dinov3convnext_multiscale.yaml` | 1.0（disparity） |

**共同超参**：`crop_size 448`、`seq_len 4`、`use_temporal true`、`num_gpus 2`、`total_step 20000`、`batch_size 8`（总，跨 2 卡每卡 4）、LoRA `r=8 / alpha=16 / dropout=0`、`video_path: null`（不加载 GemDepth，只用 DINOv3 预训练 backbone）。

---

## 2. DINOv3 backbone 在哪、怎么体现 ViT-S+ / ConvNeXt

**DINOv3 没有单独文件，是 timm 封装，全部在 `model/backbones.py`。**

- 注册表 `_REGISTRY`：名字 → timm 模型
  - `dinov3_vitsplus` → `vit_small_plus_patch16_dinov3.lvd1689m`（kind=`vit`，LoRA 目标 `qkv,proj`）
  - `dinov3_convnext_s` → `convnext_small.dinov3_lvd1689m`（kind=`convnext`，LoRA 目标 `fc1,fc2`）
- 两个封装类，把两种结构统一成 4 层 NCHW 特征喂给 DPT 头：

| | `_ViTBackbone` | `_ConvNeXtBackbone` |
|---|---|---|
| 结构 | 各向同性 ViT | 层级金字塔 |
| `is_hierarchical` | False | True |
| `embed_dims` | `[384,384,384,384]` | `[96,192,384,768]` |
| `feat_strides` | `[16,16,16,16]` | `[4,8,16,32]` |

- `build_backbone(name, ...)`：按名字选 timm 模型 + 对应封装类 + 注入 LoRA。
- 权重：配置 `backbone_weights: null` → timm/HF 官方 `lvd1689m` 预训练权重（Meta 官方 DINOv3 checkpoint 的 timm 转换版）。

**下游分派**（`model/gemdepth.py:124–131`）：
```python
self.pretrained = build_backbone(backbone, ...)
self.backbone_kind = 'convnext' if self.pretrained.is_hierarchical else 'vit'
```
→ ConvNeXt 走层级头（`DPTHeadTemporalConvNeXt` / `DPTHeadMultiScaleRefineConvNeXt`）；ViT 走 token 头（`DPTHeadTemporal` / `DPTHeadMultiScaleRefine`）。

---

## 3. LoRA 怎么加（`model/util/lora.py`）

- `LoRALinear`：包一个冻结的 `nn.Linear`，旁挂低秩 `A(r×in)`、`B(out×r)`，`B` 零初始化 → 起步 ΔW=0（等价原层）；`forward = base(x) + (x @ Aᵀ @ Bᵀ) · (alpha/r)`。只训 A/B，原权重冻结。
- `inject_lora(root, targets)`：遍历 backbone，把**属性名**在 `targets` 里的子 `nn.Linear` 就地换成 `LoRALinear`。
- 目标层：**ViT-S+ → `qkv,proj`（注意力）；ConvNeXt-s → `fc1,fc2`（每个 block 的 MLP 倒瓶颈两层线性）**。
- ConvNeXt-Small 有 36 个 block（depths 3/3/27/3），每 block 2 个 → 约 **72** 个线性层被换成 LoRALinear。

**冻结策略**（`train.py:319–355`）：
1. `model.pretrained.requires_grad_(False)` —— 整个 backbone 全冻；
2. `freeze_mode: default` —— 不额外冻结（DPT 头保持可训练）；
3. 把名字含 `lora_A/lora_B` 的参数重新打开 —— 只有 LoRA 适配器可训；
4. 断言兜底：backbone 里除 LoRA 外不允许有任何可训练参数。

→ **可训练 = LoRA(A/B) + DPT 头**；DINOv3 主干其余全部冻结。

---

## 4. 模块代码文件

| 文件 | 作用 |
|---|---|
| `model/backbones.py` | **DINOv3 核心**：ViT-S+/ConvNeXt 的 timm 封装 + `_REGISTRY` + `build_backbone` + LoRA 注入 |
| `model/util/lora.py` | `LoRALinear` + `inject_lora` |
| `model/gemdepth.py` | 建 backbone + 按 `backbone_kind` 分派 DPT 头 |
| `model/dpt_temporal.py` | `DPTHeadTemporal`（ViT 时序头，含抗死ReLU初始化） |
| `model/dpt_convnext.py` | `DPTHeadTemporalConvNeXt`（ConvNeXt 时序头） |
| `model/dpt_multiscale.py` | `DPTHeadMultiScaleRefine`（ViT multiscale 头，raw-delta + 抗死ReLU修复） |
| `model/dpt_multiscale_convnext.py` | **新增**：`DPTHeadMultiScaleRefineConvNeXt`（ConvNeXt multiscale 头） |
| `train.py` | 冻结策略（仅 LoRA + DPT 头可训） |
| `test/test_dpt_multiscale_convnext.py` | 新 ConvNeXt multiscale 头 CPU 冒烟 |

---

## 5. 实验启动命令

### 5.1 博世（LSF / bsub）

启动脚本 `scripts/train_single_a100.bsub` 已配好：队列 `batch_h200`、`module load conda/4.8.5 cuda/12.4.0`、`PY=.../envs/gemdepth/bin/python`、`TORCH_CUDA_ARCH_LIST=9.0`。它内部调用 `scripts/train_single_a100.sh <ARM>`。

**注意：bsub 默认申请 1 卡，但这 4 个实验是 2 卡（num_gpus=2）。** 先把 GPU 行改成 2 卡：
```bash
#BSUB -gpu "num=2:aff=no"      # 原为 num=1
```

逐个提交（`ARM` 即 `train_single_a100.sh` 里的 arm 名）：
```bash
ARM=ed_dinov3vits_temporal        bsub < scripts/train_single_a100.bsub
ARM=ed_dinov3vits_multiscale      bsub < scripts/train_single_a100.bsub
ARM=ed_dinov3convnext_temporal    bsub < scripts/train_single_a100.bsub
ARM=ed_dinov3convnext_multiscale  bsub < scripts/train_single_a100.bsub
```
- 断点续跑：`ARM=... RESUME=-resume bsub < scripts/train_single_a100.bsub`
- 日志：`jobs/gemdepth_single_a100.<JobID>.stdout`
- 底层等价：`accelerate launch --num_processes 2 --mixed_precision bf16 train.py --config-name <config>`
- 仅 1 卡时：配置改 `num_gpus:1`、提交加 `NUM_PROC=1`，并把 `batch_size` 8→4 防 OOM。

### 5.2 阿里云（bash + nohup，无 LSF）

阿里云没有 LSF，直接用 `scripts/train_single_a100.sh` 后台跑。VKITTI 默认路径 `/mnt/workspace/vkitti/vkitti/` 在阿里云就是对的，**无需设 `VKITTI_ROOT`**。

```bash
cd /mnt/workspace/liren/PP-DPT      # 阿里云 worktree（按实际路径）
git fetch origin && git checkout exp/scratch-ed-backbones && git pull

unset HF_ENDPOINT HF_HOME           # 关键：否则 timm/HF 下 DINOv3 权重会走坏镜像

# 逐个后台启动（2 卡；PY 用系统 python3 或 conda gemdepth 的 python）
CUDA_VISIBLE_DEVICES=0,1 PY=/usr/local/bin/python3 NUM_PROC=2 \
  nohup bash scripts/train_single_a100.sh ed_dinov3convnext_multiscale > jobs/cnx_ms.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1 PY=/usr/local/bin/python3 NUM_PROC=2 \
  nohup bash scripts/train_single_a100.sh ed_dinov3vits_multiscale     > jobs/vits_ms.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1 PY=/usr/local/bin/python3 NUM_PROC=2 \
  nohup bash scripts/train_single_a100.sh ed_dinov3convnext_temporal   > jobs/cnx_tp.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1 PY=/usr/local/bin/python3 NUM_PROC=2 \
  nohup bash scripts/train_single_a100.sh ed_dinov3vits_temporal       > jobs/vits_tp.log 2>&1 &
```
- **别让多个任务抢同一对 GPU**：要么串行跑，要么给不同的 `CUDA_VISIBLE_DEVICES`（如 `0,1` 与 `2,3`）。
- 监控：`tail -f jobs/cnx_ms.log`；续跑：`bash scripts/train_single_a100.sh <arm> -resume`。
- 仅 1 卡：`CUDA_VISIBLE_DEVICES=0 NUM_PROC=1`，且配置 `num_gpus:1`、`batch_size` 8→4。

---

## 6. 测试命令

```bash
PY=/home/izi2sgh/MYDATA/quanjie/liren/envs/gemdepth/bin/python
```

**(a) CPU 冒烟 —— 无需 GPU / 权重，任意节点可跑**（验证新 ConvNeXt multiscale 头 + 抗死ReLU初始化）：
```bash
$PY test/test_dpt_multiscale_convnext.py       # 直接 __main__ 跑（pytest 可能未装）
```

**(b) GPU 冒烟 —— 需 1 卡 + DINOv3 权重**：
```bash
$PY test/smoke_backbones.py     # build_backbone + LoRA 前反向，检查 4 层特征 + LoRA 梯度
$PY test/probe_dinov3.py        # timm 能否建 DINOv3 并加载本地 .pth，打印特征形状
```

---

## 7. 博世注意事项（易踩坑）

1. **GPU 数**：4 个实验都是 2 卡，bsub 的 `#BSUB -gpu` 记得改 `num=2`，否则 `--num_processes 2` 与 1 卡分配冲突。
2. **VKITTI 路径**：配置默认 `VKITTI_ROOT=/mnt/workspace/vkitti/vkitti/`（阿里云路径，博世不存在）。提交前 `export VKITTI_ROOT=<博世 vkitti 路径>`。
3. **DINOv3 权重**：配置 `backbone_weights: null` → timm/HF 在线下载。博世 compute 节点若无外网会失败，二选一：
   - 在有网的 login 节点先 `$PY test/probe_dinov3.py` 预热缓存；
   - 或把权重放本地并把配置 `backbone_weights` 指到 `.pth`：
     - `checkpoint/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth`
     - `checkpoint/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth`
4. **开跑先看健康检查**（尤其 multiscale —— DINOv2 时代 raw-delta 从零训会塌）：
   - `[backbone] dinov3_convnext_s: injected LoRA into 72 layers` ← N 不能为 0（否则 targets 没匹配上，LoRA 实际没加进去）
   - 训练 loss 正常下降、深度输出不全 0（抗死ReLU初始化已修，理论上不会再塌）。

---

## 8. 备注：multiscale 抗死 ReLU 修复

`DPTHeadMultiScaleRefine` / `DPTHeadMultiScaleRefineConvNeXt` 的 delta 头采用「原始 raw-delta」结构（师兄原版），只改了初始化：最后一层 conv 权重置零、最粗尺度 bias=0.5，使未训练时输出为正常数（≈0.5）而非塌成 0。避免从零训练时经外层 `F.relu` 陷入 dead-ReLU 全零陷阱。对应提交 `724e88e`。
