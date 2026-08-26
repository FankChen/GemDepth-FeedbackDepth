This doc tells you what you need to know about the project. It is intended for new contributors, and for people who want to understand the project at a high level. It is not intended to be a comprehensive reference for the codebase.


# Python virtual environment (local dev)

The project uses a conda environment named **`gemdepth`** (Python 3.10) with all
dependencies from [requirements.txt](requirements.txt) (torch 2.3.1+cu121, torchvision 0.18.1+cu121, numpy 2.2.6, ...).

- Env location: `/home/izi2sgh/.conda/envs/gemdepth`
- Conda is **not** on `PATH` by default — it lives at `/fs/applications/anaconda/4.11.0/`.
- The internal Bosch Artifactory must **bypass the external proxy**, otherwise conda/pip fail with a `ProxyError`. Export `no_proxy=".bosch.com"` before running conda/pip.

## How to activate

```bash
 source /fs/applications/anaconda/4.11.0/etc/profile.d/conda.sh
 
conda activate gemdepth
```

> Note: On the HPC cluster, jobs instead load conda via `module load conda/4.11.0` and
> activate their own environment (see the bsub section below).


# How to submit a job to the HPC cluster


```
Every job submitted by LSF is assigned with a project flag, a custom label which can be set for each job. If not provided LSF will set the project label to "default".

The project flag can be assigned to a job, by either using the -P parameter or, by setting the LSB_DEFAULTPROJECT environment variable:

My project number is BH-000425-08-05, so I will use this as an example below.

Setting the project parameter via command line



bsub -P BH-000425-08-05 [...]


Setting in a BSUB file



#BSUB -P BH-000425-08-05
[...]


Setting via environment variable (can also be added to the .bashrc file, to automatically load during session). Every job submitted, will get automatically the project flag assigned. Can be overwritten with -P.



export LSB_DEFAULTPROJECT=BH-000425-08-05


Project parameter can be checked via the bjobs -l command, e.g.: 



$ bjobs -l 8930521
Job <8930521>, User <wsh7fe>, Project <BH-000425-08-05>, Status <RUN>, Queue <admin>
                     , Job Priority <50>, Command <sleep 600>, Esub <bosch_dl>
[...]

```

# How to debug a job on the HPC cluster

```
Firstly export project number export LSB_DEFAULTPROJECT=BH-000425-08-05

Apply a an interactive node with the following command:

 bsub  -Is  -q inter_a100 -n 8  -M 15000 -gpu 'num=1' -W 8:00   /bin/bash

```



This doc tells you how to use HPC to run experiments. 


# how to edit a job shell script 

The following shows the setting of applying a computing node resource. 

Logs are save in .stdout and .stderr files

environment is loaded using module command and using `navsim` conda python environment for this project. 

Usually, we use GPU queue of `batch_b200_mig`, `batch_b200`. or  `batch_h200`. 

but h200 queue is crowd. I have a reserved node is rng-dl01-w24n01, you should use `BSUB -U  resv_cr_tfx_h200` if you want to use `rng-dl01-w24n01` node. 

```
#!/bin/bash -l

## Scheduler parameters ##

# BSUB -J BH-000425-anchor-free-dd                # job name
# BSUB -o logs/anchor_free_dd.%J.stdout      # optional: Have output written to specific file
# BSUB -e logs/anchor_free_dd.%J.stderr      # optional: Have errors written to specific file
# BSUB -q batch_b200_mig          # optional: use highend nodes w/ Volta GPUs (default: Geforce GPUs)
# #BSUB -U  resv_cr_tfx_h200  # optional: use this user group
# BSUB -W 36:00                       # fill in desired wallclock time [hours,]minutes (hours are optional)
# BSUB -n 8                       # min CPU cores,max CPU cores (max cores is optional)
# BSUB -M 50240                       # fill in required amount of RAM (in Mbyte)
# #BSUB -R "span[hosts=2]"          # optional: run on single host (if using more than 1 CPU cores)
# #BSUB -R "span[ptile=28]"         # optional: fill in to specify cores per node (max 28)
# #BSUB -P myProject                # optional: fill in cluster project
# #BSUB -m rng-dl01-w24n01  # optional: specify node id 
# BSUB -gpu "num=4:mode=exclusive_process:mps=no:aff=no" # the number of gpu 

## Job parameters ##

module purge

module load conda/4.8.5

module load cuda/12.4.0

conda activate navsim


```


```
# submit a job 

bsub <  xx.bsub

# monitor jobs 

bjobs 

# kill a job 

bkill  JOB_ID

```

---

# Paper ↔ Code: GemDepth architecture

Reference: *GemDepth: Geometry-Embedded Features for 3D-Consistent Video Depth* (arXiv:2605.10525).
This repo is the **GemDepth-VDA** variant: a frozen DINOv2 encoder + a **temporal DPT decoder**
(DepthAnythingV2 / VideoDepthAnything style), augmented with two novel modules — **GEM** and **ASTT**.

Input `X ∈ R^{B×T×C×H×W}` (a video clip of `T=32` frames at base res `518×518`); the model returns
`depth (B,T,H,W)`, `pose_enc_list`, predicted `extrinsic (B,T,4,4)`, `intrinsic (B,T,3,3)`.
The whole forward lives in `GemDepth.forward` in [model/gemdepth.py](model/gemdepth.py).

## Pipeline (in forward order)

1. **Feature Extraction (§3.1)** — *frozen* DINOv2 backbone.
   - Code: `self.pretrained = DINOv2(...)` in [model/dinov2.py](model/dinov2.py) (+ [model/dinov2_layers/](model/dinov2_layers/), [model/depth_anything_v2/](model/depth_anything_v2/)).
   - `get_intermediate_layers(x.flatten(0,1), intermediate_layer_idx['vitl']=[4,11,17,23], return_class_token=True)` → 4 multi-scale feature maps `feats[0..3]` (paper's `F_j`, layers 5/12/18/24 1-indexed). Frame dim `T` folded into batch.

2. **GEM — Geometry-Embedding Module (§3.2)** — predicts camera pose, turns it into geometric embeddings fused into `F4 = feats[3]`.
   - Learnable `camera_token` + `register_token` injected into `feats[3]`.
   - 4-layer alternating attention (EfficientPoseNet-like): `frame_blocks` (intra-frame) + `global_blocks` (inter-frame), both `vggt_Block` from [model/vggt/layers/block.py](model/vggt/layers/block.py); driven by `_process_frame_attention` / `_process_global_attention`.
   - Pose head: `CameraHead` ([model/tools/camera.py](model/tools/camera.py)) → 6-DoF `pose_enc_list`; `pose_encoding_to_extri_intri` ([model/tools/pose_enc.py](model/tools/pose_enc.py)) → `extrinsic`, `intrinsic`.
   - Canonical-frame + scale normalisation: `transform_pose_using_quats_and_trans_2_to_1`, `normalize_pose_translations` ([model/tools/geometry.py](model/tools/geometry.py)). Global scale `Z = mean ||T_i||`.
   - Geometric MLP encoders → `F_cam`: `cam_rot_encoder` (quat, 4ch), `cam_trans_encoder` (trans, 3ch), `cam_trans_scale_encoder` (log-scale, 1ch), all `GlobalRepresentationEncoder`. Their outputs are added into `feats[3]`.
   - Random pose-integration masks (`per_sample_*_mask`, ~0.5–0.9) implement the "pose integration probability" ablation (paper Tab. 9); GEM injection is stochastic during training.

3. **ASTT — Alternating Spatio-Temporal Transformer (§3.3)** — applied **early (Position 1)**, right after feature extraction, on `feats[3]` only (loop `for m in range(3,4)`).
   - Positional encoding: RoPE2D spatial ([model/tools/pos_embed.py](model/tools/pos_embed.py)) + `image_idx_emb` (1D sincos chronological index).
   - Alternates `spatial_blocks` (intra/inter-frame spatial attention) and `time_blocks` (temporal attention for point-level alignment), both `Block` from [model/tools/blocks.py](model/tools/blocks.py). Reshapes between `(b t) l c` ↔ `(b l) t c` isolate spatial vs temporal axes.

4. **DPT decoder head** — the depth head. Selected by `head_type`:
   - `temporal` (default, = paper): `DPTHeadTemporal` [model/dpt_temporal.py](model/dpt_temporal.py), subclass of `DPTHead` [model/dpt.py](model/dpt.py) (DAv2 DPT). Adds 4× `TemporalModule` motion modules ([model/motion_module/motion_module.py](model/motion_module/motion_module.py)) interleaved into the refinenet top-down path (`path_4..path_1`) for temporal smoothing.
   - `errormap` (research): `DPTHeadErrorMap` [model/dpt_errormap.py](model/dpt_errormap.py) — decodes coarse depth per stage, inverse-warps neighbour frames with GEM poses, injects photometric error map.
   - `perlayer` (research): `DPTHeadPerLayer` [model/dpt_perlayer.py](model/dpt_perlayer.py) — per-layer refine `z_l` + warp residual `dz_l` with deep supervision (`layer_depths`, aliased to `aux_depths`).
   - Warp utilities for the research heads: [model/util/warp.py](model/util/warp.py) (`photometric_error_map`, `signal_error_map`, `scale_intrinsics`), [model/util/hog.py](model/util/hog.py).
   - Final depth = original `output_conv` on `path_1`, interpolated to input res, `relu`.

## Loss (§3.4) — [loss/videoloss.py](loss/videoloss.py) `VideoDepthLoss`

`L_total = L_ssi + α·L_gm + β·L_tgm + γ·L_cam` (paper α=0.5, β=10, γ=0.2).
Code mapping (note the renamed symbols):
- `spatial_loss` = `TrimmedProcrustesLoss(alpha=0.5)` → returns `ssi + 0.5·gm` (scale-shift-invariant + multi-scale gradient matching).
- `stable_loss` = `TemporalGradientMatchingLoss` × `stable_scale=10` → the `L_tgm` temporal consistency term (paper β=10).
- `camera_loss` = `Cameraloss` (Huber on rotation + translation) × `beta=0.2` → `L_cam` (paper γ=0.2), gated by `pose_flag`.
- Research heads add masked multi-scale L1 deep supervision via `compute_aux_depth_loss` on `head.aux_depths`, weight `training.aux_depth_weight`.

## Two-stage training (§3.5) & configs

- **Stage 1** (joint pose + depth): [config/stage1.yaml](config/stage1.yaml). ~690K frames w/ GT poses.
- **Stage 2** (freeze GEM, fine-tune rest): [config/stage2.yaml](config/stage2.yaml). ~250K pose-free frames.
- Controlled single-A100 head-only experiments: [config/single_a100_baseline.yaml](config/single_a100_baseline.yaml) (temporal), [config/single_a100_errormap.yaml](config/single_a100_errormap.yaml), [config/single_a100_perlayer.yaml](config/single_a100_perlayer.yaml). `freeze_mode: head_only` trains only the DPT head from pretrained `checkpoint/gemdepth.pth`.
- Entry point: [train.py](train.py) (Hydra + Accelerate). Optimizer: AdamW, `1e-4` for new ASTT/GEM modules, `1e-6/1e-5` for pretrained parts. Discriminative LR groups: `spatial_blocks`/`time_blocks` = `dec_lr`, everything else = `other_lr`.

## Repo layout (by responsibility)

- [model/gemdepth.py](model/gemdepth.py) — top-level `GemDepth`: DINOv2 + GEM + ASTT + head wiring, plus inference helpers (`INFER_LEN=32`, `OVERLAP=10`, keyframe interpolation).
- [model/dinov2.py](model/dinov2.py), [model/dinov2_layers/](model/dinov2_layers/), [model/depth_anything_v2/](model/depth_anything_v2/) — frozen encoder.
- [model/dpt.py](model/dpt.py), [model/dpt_temporal.py](model/dpt_temporal.py), [model/dpt_errormap.py](model/dpt_errormap.py), [model/dpt_perlayer.py](model/dpt_perlayer.py) — DPT decoder heads.
- [model/motion_module/](model/motion_module/) — temporal attention `TemporalModule` for the DPT head.
- [model/tools/](model/tools/) — GEM/ASTT building blocks: `camera.py` (pose head), `pose_enc.py`, `geometry.py`, `rotation.py`, `pos_embed.py`, `blocks.py`, `self_attention.py`, `transformer.py`, `position_encoding.py`.
- [model/vggt/](model/vggt/) — VGGT-derived attention blocks/layers used by GEM (`frame_blocks`, `global_blocks`) and camera head.
- [model/util/](model/util/) — `transform.py` (image preprocessing), `warp.py`, `hog.py` (research-head warp signals).
- [loss/videoloss.py](loss/videoloss.py) — all training losses.
- [dataset/dataset_mix.py](dataset/dataset_mix.py) — `DepthVideoDataset` mixed-dataset video loader; [dataset/dataset_extract/](dataset/dataset_extract/) — per-dataset extractors (kitti/bonn/scannet/sintel) for eval.
- [evaluation/](evaluation/) — `eval/eval.py`, `eval/metric.py` (AbsRel, δ1, TAE, TCD, F1, ATE), `inference/` (video + point-cloud inference).
- [scripts/](scripts/) — `.bsub` launchers, `train_single_a100.sh`, and GPU smoke tests (`smoke_perlayer.py`, `smoke_perlayer_e2e.py`, `smoke_errormap.py`).

## Key implementation notes

- ASTT is placed at **Position 1** (early, on raw features) — the paper shows this beats mid/late placement (Tab. 7). In code this is the `for m in range(3,4)` block operating on `feats[3]` before the DPT head.
- The head runs under `torch.autocast("cuda", enabled=False)` — DPT decode is done in fp32 for stability.
- Camera pose/intrinsics are **detached** before being fed to the research heads' warp (poses are a fixed geometric prior for depth, not supervised through the warp).
- The warp branch is numerically sensitive: with a **randomly-initialised** GEM it produces degenerate poses → NaN grads. Always load pretrained `checkpoint/gemdepth.pth` before any forward/backward test.


```