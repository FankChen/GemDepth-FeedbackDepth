# Multiscale 多尺度 refine — 周末进度报告

- 日期：2026-07-13 ｜ 分支：`exp/multiscale-aux-disp`（基于 `bugfix/clean`，师兄原分支未改动）
- 环境：阿里云 DSW（8×H20），`head_only + VKITTI + 10k step + lr1e-4`，与 baseline 同协议
- 评测：KITTI Eigen（`kitti_video.json`），逆深度（inverse depth = 1/depth）空间 lstsq scale/shift 对齐（与 baseline 完全一致）

---

## 一、TL;DR

多尺度 refine head 训练本身正常收敛（loss 56.4→4.85）。但首次评测 AbsRel 0.243，异常偏高。定位后确认**不是方法差，是训练时辅助损失把模型输出从「逆深度空间」拉到了「米制深度空间」，与评测假设错配**。已按"复制一份、不动原件"做修复副本并重训，**AbsRel 0.243→0.0725、δ1 0.523→0.950，追平 baseline（0.0712），根因与修复均验证成立**。

---

## 二、实验数据（KITTI Eigen）

| 方案 | AbsRel↓ | RMSE↓ | δ1↑ | 说明 |
|---|---|---|---|---|
| baseline lr1e-4（同配方对照） | **0.0712** | – | – | temporal head |
| multiscale（原始输出，直接评测） | 0.2430 | 8.138 | 0.523 | 评测错配 → 虚高 |
| multiscale（输出取倒数后评测） | 0.1435 | 5.936 | 0.806 | 纠正错配后仍偏高 |
| **multiscale_fix（修复后重训）** | **0.0725** | **3.267** | **0.950** | 直接评测，追平 baseline |

诊断：multiscale 输出 `min/median/max = 4.73 / 26.1 / 195.3` → 是**米制深度**（逆深度应为 0.01~0.3）。

---

## 三、问题分析（根因）

模型整体在**逆深度空间**（inverse depth = 1/depth，即 MiDaS 家族的输出量；社区也俗称 "disparity"，但严格不是双目视差）：主损失 SSI、评测对齐都按逆深度。但 multiscale 的**辅助损失把输出拉向了米制深度**，两者互为倒数，仿射对齐救不回来 → AbsRel 虚高。三处叠加导致：

**① 主损失监督的是「逆深度」** — `loss/videoloss.py` `VideoDepthLoss.forward`
```python
target_inverse[valid_mask] = 1 / target[valid_mask]          # 逆深度 = 1/depth
loss_dict['spatial_loss'], ... = self.spatial_loss(
        prediction=prediction..., target=target_inverse...)   # 主损失对齐到逆深度
```

**② 辅助损失却用「米制深度」直接 L1** — `train.py` `compute_aux_depth_loss`
```python
gt = depth_gt.flatten(0, 1).float()        # depth_gt = data['depth']，米制真实深度(未取倒数)
...
diff = (d.float() - gt_s).abs() * m_s       # |预测 − 米制深度|，绝对空间 L1
```

**③ 且该辅助损失权重是 errormap/perlayer 的 10 倍** — `config/single_a100_multiscale.yaml`
```yaml
aux_depth_weight: 1.0     # errormap / perlayer 都只有 0.1
```

> 量化验证（smoke）：同一批预测，旧辅助损失（米制深度）= **29.55**，逆深度空间 = **0.026**，量级差约 1000 倍。weight=1.0 时它直接主导训练，把输出拽向米制深度。这与之前 perlayer 的 `deep_sup` 量纲问题是同一类。

---

## 四、修复（已做，复制一份、原件不动）

原则：**师兄的 `single_a100_multiscale.yaml` / `dpt_multiscale.py` / `videoloss.py` 一律不动**，全部为加法，所有已有实验行为不变（`git diff` 三文件为空）。

**新增逆深度空间的辅助损失** — `train.py` `compute_aux_depth_loss_disp`
```python
gt_disp[valid] = 1.0 / gt[valid].clamp(min=1e-3)                 # 改为监督 GT 逆深度
...
with torch.no_grad():
    scale, shift = compute_scale_and_shift(d..., gt_s..., m_s...) # 每样本 SSI 对齐(detach)
d_aligned = scale.view(-1,1,1,1) * d + shift.view(-1,1,1,1)
diff = (d_aligned - gt_s).abs() * m_s                            # 逆深度空间 masked L1
```
- 新开关 `training.aux_depth_space`（默认 `'depth'` = 原行为；仅新 config 设 `'disparity'`）。
- 新配置 `config/single_a100_multiscale_fix.yaml`：copy 自原 multiscale，**只改 `aux_depth_space: disparity`**（其余、包括 `aux_depth_weight: 1.0` 全部保持，隔离单一变量）。
- 训练脚本新增 `multiscale_fix` arm。

正确性验证（smoke）：GT 逆深度的仿射变换喂入新损失 → **loss = 0.0**，证明 SSI 对齐只监督形状、正确工作。

---

## 五、需要师兄确认

1. **修复方向是否认可**：把辅助损失从「米制深度绝对 L1」改为「逆深度空间 + SSI 对齐」，与主损失同空间。
2. **是否本就打算配一套「深度空间」的评测**？如果 multiscale 有意输出米制深度，那就不是 bug，而是缺一套对应的 eval——这点只有你知道。
3. **已重训验证（2026-07-13）**：`multiscale_fix` AbsRel **0.0725** / δ1 **0.950**，直接评测即回正，**追平 baseline 0.0712**。说明多尺度 refine 修复后单帧精度**无明显增益也无损害**，之前的"差"纯属评测错配。要体现其价值，建议下一步看时序一致性（TAE）或进一步优化 head。
