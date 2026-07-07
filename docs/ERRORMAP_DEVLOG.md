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
| **temporal baseline** | temporal | — | **0.0677** | 3.167 | 0.957 | Bosch job12511433 | **对照基准** |
| em_single rgb | errormap_single | rgb | 0.0881 | 4.095 | 0.923 | 阿里云 DSW | 劣于 baseline ~0.02，结构待改进 |
| em_single feat | errormap_single | feat | _TBD_ | _TBD_ | _TBD_ | 阿里云 DSW | 训练中/待 eval |
| em_single rgbfeat | errormap_single | rgbfeat | _TBD_ | _TBD_ | _TBD_ | 阿里云 DSW | 训练中/待 eval |
| em_single hog | errormap_single | hog | _TBD_ | _TBD_ | _TBD_ | 阿里云 DSW | 训练中/待 eval |

## 时间线
- **2026-07-02**：temporal baseline 4 数据集复现（KITTI 0.0677/3.167/0.957 ≈ 论文）。em_single rgb 在阿里云启动训练。
- **2026-07-07**：KITTI 测试集经 git 中继搬到阿里云；em_single rgb eval = **0.0881/4.095/0.923**。
  → 现结构下 errormap_single(rgb) **劣于** temporal baseline，确认这是待解决的结构问题，非"协议不可比"。
  feat/rgbfeat/hog 训练进行中。

## 待办 / 下一步
- [ ] feat / rgbfeat / hog 训完 eval，回填结果表。
- [ ] 若 4 臂均劣于 temporal 0.0677 → 分析 error-map head 结构瓶颈（误差注入位置 / 融合方式 / aux-loss 权重 / refine 深度），改进结构。
- [ ] 目标：errormap_single 追平并超过 temporal baseline。
