# Multilevel DPT on Virtual KITTI 2

This repository trains a video depth model with a frozen pretrained DINOv3
ConvNeXt-S backbone, LoRA adapters, and a temporal coarse-to-fine multiscale DPT
decoder. The canonical workflow targets **Virtual KITTI 2.0.3 (VKITTI2)**.

## Environment

Use Python 3.10 and install the existing dependencies:

```bash
conda create -n multilevel-dpt python=3.10
conda activate multilevel-dpt
pip install -r requirements.txt
```

Training uses PyTorch, Hydra, Accelerate, and bf16 mixed precision. The default
backbone is loaded through `timm`. If compute nodes cannot access the model
host, download the DINOv3 ConvNeXt-S weights beforehand and pass
`model.backbone_weights=/absolute/path/to/weights.pth` to the training command.

## VKITTI2 data

The default dataset root is `/mnt/workspace/vkitti/vkitti`. Set `VKITTI_ROOT`
only when using another location. The loader searches recursively, so extra
wrapper directories are allowed, but each scene/variation must have this
effective layout:

```text
$VKITTI_ROOT/
└── .../
    └── Scene01/
        └── 15-deg-left/
            ├── extrinsic.txt
            └── frames/
                ├── rgb/Camera_0/rgb_00000.jpg
                └── depth/Camera_0/depth_00000.png
```

Required VKITTI2 components are RGB, depth, and text ground truth (extrinsics).
Depth PNG values are interpreted as centimetres and converted to metres.
Camera 0 is used. Clips contain four consecutive frames.

The split is scene-based with no frame overlap:

| Split | Scenes | Variation | Camera |
| --- | --- | --- | --- |
| Train | `Scene01`, `Scene02`, `Scene18` | `15-deg-left` | `Camera_0` |
| Test | `Scene06`, `Scene20` | `15-deg-left` | `Camera_0` |

All frames from the selected training scenes are used as sliding four-frame
clips. All frames from the selected test scenes are evaluated in non-overlapping
four-frame windows; an incomplete final window is discarded. Evaluation reports
dense `AbsRel`, `RMSE`, and `delta1` after scale-and-shift alignment in
inverse-depth space.

## Train

All local training and testing goes through one script:

```bash
bash scripts/vkitti.sh train
```

The canonical config is `config/vkitti/vkitti.yaml`. Defaults are one GPU,
global batch size 8, crop size 448, clip length 4, 20,000 optimizer steps,
AdamW, and bf16. Checkpoints are written to `checkpoint/vkitti/`; TensorBoard
logs are written to `logs/vkitti/`.

Common overrides follow normal Hydra syntax:

```bash
# Two GPUs; dataloader.batch_size remains the global batch size.
NUM_PROCESSES=2 bash scripts/vkitti.sh train dataloader.batch_size=8

# Resume from the latest checkpoint in checkpoint/vkitti/.
RESUME=true bash scripts/vkitti.sh train

# Use a local pretrained backbone and change the training length.
bash scripts/vkitti.sh train \
  model.backbone_weights=/path/to/dinov3_convnext_small.pth \
  total_step=50000
```

`final.pth` contains the complete training state. `final_model.pth` contains the
model weights used for evaluation.

## Test

```bash
bash scripts/vkitti.sh test
```

By default this evaluates `checkpoint/vkitti/final_model.pth`, writes a dense
visualization to `outputs/vkitti/vkitti_dense.png`, and prints aggregate
metrics. Override paths with environment variables:

```bash
CKPT=/path/to/final_model.pth \
OUTPUT=outputs/vkitti/custom.png \
bash scripts/vkitti.sh test --limit_seqs 8
```

The canonical model predicts inverse depth, so `--invert` must not be used.

## LSF cluster

Cluster submission is isolated from portable experiment logic:

```bash
export VKITTI_ROOT=/path/to/vkitti2
export PYTHON=/path/to/env/bin/python

# Training
MODE=train bsub < bsub/vkitti.bsub

# Resume training
MODE=train RESUME=true bsub < bsub/vkitti.bsub

# Evaluation
MODE=test CKPT=/path/to/final_model.pth bsub < bsub/vkitti.bsub
```

Adjust the `#BSUB` queue, project, reservation, GPU, memory, and wall-time lines
in `bsub/vkitti.bsub` for the target cluster. The `.bsub` file only requests
resources and loads modules; all experiment behavior remains in
`scripts/vkitti.sh`.

## Canonical files

```text
config/vkitti/vkitti.yaml          VKITTI2 experiment configuration
config/stages/                     original two-stage GemDepth configs
config/single_a100/                controlled single-GPU configs
config/scratch/dinov2/             scratch DINOv2 experiments
config/scratch/dinov3_vits/        scratch DINOv3 ViT-S+ experiments
config/scratch/dinov3_convnext/    scratch DINOv3 ConvNeXt experiments
config/multilevel_decoder/          iterative decoder experiments
scripts/vkitti.sh                   only training/testing shell entrypoint
bsub/vkitti.bsub                    LSF submission wrapper
train.py                            Hydra/Accelerate training implementation
evaluation/inference/eval_vkitti_dense.py
                                    dense VKITTI2 evaluation
```

## Backbone and decoder selection

Experiments select both implementations by their exact Python class names:

```yaml
model:
  backbone: DINOv3ConvNeXtSmallBackbone
  decoder: DPTHeadTemporalConvNeXt
```

Backbone and decoder classes register themselves with the plain `@register`
decorator. `model/backbones.py`, `model/backbone_*.py`, and `model/dpt_*.py`
modules are discovered automatically. Component-specific constructor options
can be supplied under `model.backbone_kwargs` and `model.decoder_kwargs`.

New backbones use the same pattern in `model/backbone_*.py`:

```python
from model.backbone_registry import register


@register
class MyBackbone(nn.Module):
    feature_format = "pyramid"
    ...
```

New decoder implementations only need to live in a `model/dpt_*.py` file and
decorate the class:

```python
import torch.nn as nn

from model.decoder_registry import register


@register
class MyIterativeDecoder(nn.Module):
    def __init__(self, in_channels_list, features=256, iterations=4):
        ...
```

The constructor receives matching standard arguments automatically. Additional
arguments are configured without changing the registry:

```yaml
model:
  decoder: MyIterativeDecoder
  decoder_kwargs:
    iterations: 6
```

The iterative decoder baseline is:

```bash
CONFIG_NAME=multilevel_decoder/baseline \
bash scripts/vkitti.sh train
```
