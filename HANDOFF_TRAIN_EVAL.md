# GemDepth · em_single 臂 训练 & 测试 交接文档

> 目的：在**任意一台带 NVIDIA GPU 的机器**上，从零把本方案（`em_single` 臂）跑起来——
> 训练 + 评估。文档不依赖任何特定集群（LSF / Bosch 均非必需），全程复制粘贴即可。
>
> 本方案 = 在冻结的 GemDepth 主干上，只微调一个**单阶段 error-map DPT 头**（`errormap_single`），
> 在 VKITTI 2.0.3 上 head-only 微调，在 KITTI / Sintel / Bonn / ScanNet 上评估。
> 与 `baseline` 臂唯一的区别就是这个 DPT 头，属于受控对比实验。

---

## 0. 一览：需要准备什么

| 东西 | 说明 | 大小 | 放哪 / 怎么指 |
| --- | --- | --- | --- |
| 代码 | `feat/errormap-single-clean` 分支 | — | `git clone` |
| Python 环境 | python 3.10 + torch 2.3.1+cu121 | — | conda / venv |
| **预训练权重** `gemdepth.pth` | 含 DINOv2 编码器，训练&baseline测试都要 | 2.84 GB | `checkpoint/gemdepth.pth` |
| **训练数据** VKITTI 2.0.3 | 5 Scene × 10 天气 = 50 序列 | ~30 GB | `VKITTI_ROOT` 环境变量 |
| **测试数据** gemdepth_eval | kitti / sintel / bonn / scannet | ~十几 GB | `--benchmark_path` |

> 权重、VKITTI、gemdepth_eval 三份数据我另外提供给你（或从下方来源自取）。
> 代码是公开仓库，直接 clone。

---

## 1. 拉代码

```bash
git clone -b feat/errormap-single-clean \
  https://github.com/FankChen/GemDepth-FeedbackDepth.git GemDepth
cd GemDepth
```

> 一定用 `feat/errormap-single-clean` 分支——它带了本方案的头和训练脚本，且已修好权重加载的 allowlist。

---

## 2. 装环境

```bash
conda create -n gemdepth python=3.10 -y
conda activate gemdepth

# 先按官方索引装 torch（cu121），再装其余依赖
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> - 国内加速可加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`（torch 那条要保留官方 index-url）。
> - 若模型 `import flash_attn` 报缺失：`pip install flash-attn==2.6.3 --no-build-isolation`
>   （需与 CUDA/torch 匹配；实测 flash-attn 2.6.3 可用）。
> - 快速自检：`python -c "import torch;print(torch.__version__, torch.cuda.is_available())"`
>   应输出 `2.3.1+cu121 True`。

---

## 3. 放权重 + 指数据路径（含软链接）

代码里有几个默认路径是我这边的绝对路径。你在自己机器上**不用改代码**，用下面三种方式把
数据「指」过去即可（推荐环境变量 / CLI 覆盖，软链接作为备选）。

### 3.1 预训练权重 `checkpoint/gemdepth.pth`
```bash
mkdir -p checkpoint
# 方式①：直接放到位
cp /你的路径/gemdepth.pth checkpoint/gemdepth.pth
# 方式②：软链接（文件在别处、不想复制 2.84G 时）
ln -s /你的路径/gemdepth.pth checkpoint/gemdepth.pth
# 方式③：训练/测试命令里显式指定（见后），不放这里也行

# 校验完整性（可选）
md5sum checkpoint/gemdepth.pth   # 期望 03c25b64b420faa12c01e7a933eaf5e9
```

### 3.2 训练数据 VKITTI 2.0.3 → 用 `VKITTI_ROOT`
解压后的目录里应直接是 `Scene01/ Scene02/ Scene06/ Scene18/ Scene20/`：
```
$VKITTI_ROOT/
└── Scene01/{clone,morning,rain,fog,overcast,sunset,15-deg-left,15-deg-right,30-deg-left,30-deg-right}/
    ├── frames/rgb/Camera_0/rgb_XXXXX.jpg
    ├── frames/depth/Camera_0/depth_XXXXX.png
    └── extrinsic.txt
```
```bash
# 推荐：直接用环境变量（config 已经读它）
export VKITTI_ROOT=/你的路径/vkitti          # 这个目录下就是 Scene01..Scene20

# 或者：软链接到仓库内一个固定名字，再指过去
ln -s /你的路径/vkitti  $PWD/vkitti_data
export VKITTI_ROOT=$PWD/vkitti_data
```
> VKITTI 2.0.3 官方来源（3 个 tar，解压到同一目录即上面的结构）：
> - `https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar` (7.0G)
> - `https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_depth.tar` (7.6G)
> - `https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_textgt.tar.gz` (含 `extrinsic.txt`)

### 3.3 测试数据 gemdepth_eval → 用 `--benchmark_path`
```
$EVAL_ROOT/                     # 即 gemdepth_eval
├── kitti/   kitti_video.json          + 图像
├── sintel/  sintel_video.json         + 图像
├── bonn/    bonn_video_500.json       + 图像
└── scannet/ scannet_video_500.json    + 图像
```
测试命令直接把 `--benchmark_path` / `--json_file` 指到这里即可，**无需软链接**。
```bash
export EVAL_ROOT=/你的路径/gemdepth_eval
```

---

## 4. 训练

### 4.1 一条命令（推荐）
```bash
conda activate gemdepth
export VKITTI_ROOT=/你的路径/vkitti          # 含 Scene01..Scene20
PY=$(which python) ./scripts/train_single_a100.sh em_single
```
脚本内部等价于：
```bash
accelerate launch --num_processes 1 --mixed_precision bf16 \
  train.py --config-name single_a100_em_single
```

### 4.2 不用脚本、纯手工（想覆盖路径时更直观）
```bash
accelerate launch --num_processes 1 --mixed_precision bf16 \
  train.py --config-name single_a100_em_single \
    dataset.train.data_dirs="['/你的路径/vkitti/']" \
    model.video_path=./checkpoint/gemdepth.pth
```

### 4.3 训练产物 & 关键配置
- **配置文件**：[config/single_a100_em_single.yaml](config/single_a100_em_single.yaml)
  - `head_type: errormap_single`，`warp_signal: rgb`，`freeze_mode: head_only`（只训头）
  - `total_step: 10000`，`save_freq: 2000`，`aux_depth_weight: 0.1`
  - `dataloader.batch_size: 1`，`grad_accum: 4`（等效 batch=4），`seq_len: 16`，`crop: 518`
- **checkpoint 目录**：`checkpoint/single_a100_em_single/`
  - 每 2000 步存 `checkpoint_{step}.pth`（含 model/optim/sched，仅保留最近 3 个）
  - 结束存 `final.pth`（全套） + **`final_model.pth`（仅模型权重，测试用这个）**
- **日志**：`logs/single_a100_em_single/`，`tensorboard --logdir logs/single_a100_em_single`
- **断点续训**：脚本加 `-resume`，或手工加 `resume=true`（自动找最新 checkpoint）

> **正常启动的标志**：日志出现
> `[init] loaded pretrained: missing=N unexpected=M`，其中 missing 只应是
> `error_encoder.* / refine_head.*`（新头的模块，预训练里没有，正常），loss 非 NaN。
> 单卡显存约需 ~40G 量级（seq_len16、crop518）；显存不够可调小 `dataset.train.seq_len`。

---

## 5. 测试 / 评估

评估分两步：**推理**（`infer.py` 生成每帧深度 `.npy`）→ **打分**（`eval.py` 出指标）。
结果写入 `output_eval/results.txt`。

### 5.1 四个数据集的参数速查
| 数据集 | `--json_file` | 推理 `--datasets` | 打分 `--datasets` |
| --- | --- | --- | --- |
| KITTI   | `kitti/kitti_video.json`         | `kitti`   | `kitti`       |
| Sintel  | `sintel/sintel_video.json`       | `sintel`  | `sintel`      |
| Bonn    | `bonn/bonn_video_500.json`       | `bonn`    | `bonn_500`    |
| ScanNet | `scannet/scannet_video_500.json` | `scannet` | `scannet_500` |

### 5.2 评估「预训练 baseline」（gemdepth.pth，原始 temporal 头）
以 KITTI 为例，其余数据集把三处名字按上表替换：
```bash
export EVAL_ROOT=/你的路径/gemdepth_eval
OUT=./output_eval

python evaluation/inference/infer.py \
  --infer_path $OUT --json_file $EVAL_ROOT/kitti/kitti_video.json \
  --datasets kitti --input_size 518 --encoder vitl \
  --ckpt ./checkpoint/gemdepth.pth --head_type temporal

python evaluation/eval/eval.py \
  --infer_path $OUT --benchmark_path $EVAL_ROOT --datasets kitti

cat $OUT/results.txt
```

### 5.3 评估「你自己训出来的 em_single 模型」
关键差异：`--ckpt` 指向训练产物 `final_model.pth`，`--head_type errormap_single`，加 `--warp_signal rgb`：
```bash
export EVAL_ROOT=/你的路径/gemdepth_eval
OUT=./output_eval_single

python evaluation/inference/infer.py \
  --infer_path $OUT --json_file $EVAL_ROOT/kitti/kitti_video.json \
  --datasets kitti --input_size 518 --encoder vitl \
  --ckpt ./checkpoint/single_a100_em_single/final_model.pth \
  --head_type errormap_single --warp_signal rgb

python evaluation/eval/eval.py \
  --infer_path $OUT --benchmark_path $EVAL_ROOT --datasets kitti
```

> **测中间步 checkpoint**（如 `checkpoint_8000.pth`）：它是「全套」格式，`infer.py` 需要
> 「仅模型」state_dict，先抽一下：
> ```bash
> python -c "import torch; d=torch.load('checkpoint/single_a100_em_single/checkpoint_8000.pth',map_location='cpu'); torch.save(d['model_state_dict'],'checkpoint/single_a100_em_single/model_8000.pth')"
> ```
> 然后 `--ckpt checkpoint/single_a100_em_single/model_8000.pth`。

### 5.4 预训练 baseline 参考指标（我们已复现，与论文吻合）
| 数据集 | AbsRel ↓ / δ1 ↑ |
| --- | --- |
| KITTI   | 0.068 / 0.959 |
| Sintel  | 0.159 / 0.827 |
| Bonn    | 0.053 / 0.974 |
| ScanNet | 0.068 / 0.961 |

> 你用 `gemdepth.pth` + `temporal` 头跑 5.2 应能复现出接近上表的数；
> 能对上就说明环境/数据/权重都没问题，再去评估自己训的 em_single 模型。

---

## 6. 常见问题
- **`missing keys ... refine_head`**：确认在 `feat/errormap-single-clean` 分支（allowlist 已含 `refine_head`）。
- **`FileNotFoundError` / 数据集为空**：`VKITTI_ROOT` 指的目录下要**直接**是 `Scene01..Scene20`；
  自检 `find $VKITTI_ROOT -path '*frames/rgb/Camera_0' -type d | wc -l` 应为 50。
- **`load_state_dict ... strict=True` 报错**：测试时 `--head_type` 必须与权重匹配
  （`gemdepth.pth`→`temporal`；自训模型→`errormap_single`），且用 `final_model.pth` 而非 `final.pth`。
- **flash-attn 相关报错**：见第 2 步补装 `flash-attn==2.6.3`。

---

## 7. 一页速查
```bash
# 环境
conda activate gemdepth
# 训练
export VKITTI_ROOT=/path/vkitti
PY=$(which python) ./scripts/train_single_a100.sh em_single
# 测试(自训模型, KITTI)
export EVAL_ROOT=/path/gemdepth_eval
python evaluation/inference/infer.py --infer_path ./out_single \
  --json_file $EVAL_ROOT/kitti/kitti_video.json --datasets kitti \
  --input_size 518 --encoder vitl \
  --ckpt ./checkpoint/single_a100_em_single/final_model.pth \
  --head_type errormap_single --warp_signal rgb
python evaluation/eval/eval.py --infer_path ./out_single \
  --benchmark_path $EVAL_ROOT --datasets kitti
cat ./out_single/results.txt
```
