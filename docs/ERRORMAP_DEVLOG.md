# Error-Map Single-Head 实验记录 (DEVLOG)

## 对照协议（严格公平）
所有 arm 使用**完全相同**的训练协议，唯一变量是 DPT head 结构 / error-map warp signal：
- 预训练权重 `checkpoint/gemdepth.pth`
- `freeze_mode = head_only`（冻结 backbone，只训 head）
- 数据 VKITTI 2.0.3（50 序列）
- `total_step = 10000`

测试：KITTI Eigen（`infer.py --head_type errormap_single --warp_signal <sig>` → `eval.py` lstsq scale/shift 对齐）。指标 AbsRel↓ / RMSE↓ / δ1↑。

> **temporal baseline 与 em_single 各臂是同协议、同数据、同步数，仅 head 结构不同 → 严格可比。**
> em_single 方向要算成功，必须**追平或超过 temporal baseline 的 AbsRel 0.0677**。

## 结果表（KITTI, AbsRel↓ / RMSE↓ / δ1↑）

| 版本 | head | warp signal | AbsRel | RMSE | δ1 | 训练环境 | 备注 |
|---|---|---|---|---|---|---|---|
| **temporal baseline** | temporal | — | **0.0677** | 3.167 | 0.957 | Bosch j12511433 / 阿里云 lr1e-5 | **对照基准** |
| em_single rgb | errormap_single | rgb | 0.0883 | 4.098 | 0.9230 | 阿里云 DSW | 劣于 baseline ~0.021 |
| em_single feat | errormap_single | feat | **0.0847** | 4.010 | 0.9278 | 阿里云 DSW | 4 臂 AbsRel 最优，仍劣于 baseline |
| em_single rgbfeat | errormap_single | rgbfeat | 0.0863 | **3.757** | **0.9300** | 阿里云 DSW | RMSE/δ1 最优臂 |
| em_single hog | errormap_single | hog | 0.0873 | 4.075 | 0.9232 | 阿里云 DSW | — |

## Baseline LR sweep（temporal head，KITTI，阿里云复现）
证明环境/数据/评测链路可靠 + 定最优 lr（对照基准即取 lr1e-5）。

| lr | AbsRel | RMSE | δ1 | 备注 |
|---|---|---|---|---|
| 1e-4 | 0.0712 | 3.327 | 0.9499 | 偏大 |
| 5e-5 | 0.0691 | 3.249 | 0.9542 | |
| **1e-5** | **0.0676** | 3.167 | 0.9571 | **最优，≈ Bosch 0.0677** |
| 1e-6 | 0.0681 | 3.158 | 0.9573 | |

lr1e-5 中间步（早收敛，10k 略过训）：step6000 **0.06736** / 8000 0.06758 / 10000 0.06767。
→ baseline ~6000 步即达最优；反观 error-map 各臂训满才最好（em_feat step6000→final：0.0866→0.0847）。

## 时间线
- **2026-07-02**：temporal baseline 4 数据集复现（KITTI 0.0677/3.167/0.957 ≈ 论文）。em_single rgb 在阿里云启动训练。
- **2026-07-07**：KITTI 测试集搬到阿里云；em_single rgb eval = 0.0883/4.098/0.923，劣于 baseline。
- **2026-07-09**：4 臂 error-map + baseline LR sweep 全部 eval 完成（KITTI）。
  - baseline lr1e-5 复现 **0.0676** ≈ Bosch → 环境/数据/协议可靠，问题不在环境。
  - error-map 4 臂 AbsRel 0.0847~0.0883，**全劣于 baseline 0.0677**；AbsRel 最优 em_feat，RMSE/δ1 最优 em_rgbfeat。
  - 训练 loss 持续降但 KITTI eval 差 → **loss/eval 脱钩**，疑似 VKITTI→KITTI 域迁移过拟合。

## 待办 / 下一步
- [x] feat / rgbfeat / hog 训完 eval，回填结果表。
- [ ] 诊断 loss/eval 脱钩：① VKITTI held-out eval（若 error-map 在合成域好、真实域差 → 坐实域迁移过拟合）；② KITTI 上 error-map 可视化（看残差是否被动态/光照/位姿噪声主导）。
- [ ] 结构改进：误差注入位置 / 融合方式 / aux-loss 权重 / refine 深度 / 训练步数。
- [ ] 目标：errormap_single 追平并超过 temporal baseline 0.0677。
