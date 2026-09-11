# Corrected G1：实现验证与匹配运行契约

状态（2026-09-11）：阿里云首轮 G1 已准备成功，但 seed0 在 **C1 指标回放**阶段停止，尚无门控优化或正式训练。已修复回放前遗漏的 FP32/TF32 初始化，**148 项相关回归通过；远端修复回放仍待执行**。C0/C1 六组已完成，原源码/配置/权重/输出保持只读。

## 1. 固定问题与矩阵

在已校准的 raw/GEV 上增加循环更新，相比已完成 C1 是否改善固定 final1000 开发指标？不预设胜者，不追加种子或挑早期 checkpoint。

| 项目 | C1（已完成） | G1（本次） |
|---|---|---|
| 头 | calibrated full-volume，0 次循环 | 新注册 calibrated GRU，8 次循环 |
| 数据 | Scene01/02 各15训练 clips；Scene18 16开发 clips | 逐 seed 复用完全相同 manifest/crop/camera/mask/features |
| 输入 | 4帧、256 crop、clean frozen ConvNeXt-S、GT metric K/T | 相同；不是 RGB-only |
| 深度 | 3–80m，32 inverse-depth bins，stride8，G8 | 相同 |
| 初始化 | 各 seed 原始共同 initial，含 BN 状态 | 精确映射 C1 全部公共 initial；新增循环模块按同 seed 初始化 |
| 优化 | AdamW lr .001 / wd0 / clip1 / FP32 / batch1clip | 相同，空优化器；原 training_order |
| 正式预算 | seeds0/1/2，各1000 updates | 相同；不使用任何门控训练权重 |
| loss | 最终 normalized absolute index-L1 | 相同，无中间/sequence loss |
| 评测 | 固定支持域，无 affine；final1000 | 相同；逐 clip/scene 重算聚合，三种子原值/均值/样本sd/配对差 |
| head 参数 | 1,234,505 | 5,593,097（新增4,358,592） |

因此可比较的是**整个 G1 变体**，不是容量/计算匹配的纯迭代因果消融。冻结骨干不计入 head 参数；循环额外计算与每轮诊断同步开销均需披露。

## 2. 修复内容与官方差异

[新注册头](model/dpt_calibrated_gru_convnext.py)不修改旧 head 或核心训练器：

- lookup 输入为 regularized logits 和 **raw correlation 的 group mean**，每层顺序 GEV→raw；softmax 仅产生初始 index，不再冒充 raw。G8 原体仍完整送入原 C1 aggregation。
- 三层 hidden stems 逐层计算后分别 tanh；循环 coarse→mid→fine。每轮只 detach 查询坐标，hidden 和两个 volume 的梯度保留。
- GT 相机有限性、正焦距、旋转、metric gauge 和 native ConvNeXt 像素中心显式检查；不使用预测 GEM 相机、SSI 尺度或输出 floor。
- 官方 IGEV-MVS 使用无界加法与初始/多轮监督。本轮为了保留 C1 监督及物理 index 契约，**使用 final-only loss 和有界更新变体，不宣称精确复现官方 IGEV**。

令 $M=D-1$，$u=\tanh(2\Delta/M)$：

$$
i'=\begin{cases}i+(M-i)u,&\Delta\ge0\\i+iu,&\Delta<0.\end{cases}
$$

零更新恒等、镜像对称、输出保留在 $[0,M]$；无 hard clamp/epsilon/NaN 替换。边界附近梯度衰减，有限精度下可能饱和，不能用有界输出掩盖不稳定。日志保留未约束加法 proposal 越界比例、实际更新、每轮查询范围/raw lookup 幅度与最终边界比例。

官方比对依据：gangweiX/IGEV 的 IGEV-MVS，commit `e02312b8f9615c52346bd04c08f8d219c64c404d`（core/corr、core/update、core/igev_mvs、train_mvs）。本实验仍使用 C1 的 backbone、hourglass、view weighting 与上采样，不把它们称为官方等价组件。

## 3. 门控 → 丢弃 → 全部重置 → 正式训练

[固定配置](config/stereogru/corrected_gru.yaml)、[独立运行器](scripts/stereogru_corrected_gru.py)、[阿里云入口](scripts/run_stereogru_corrected_gru.sh)。

1. 读取已完成 repetitions 与 seed summary；校验旧全部 contract/source/runtime/input/cache/权重/完成证书，并从 clip 行重算 C0/C1 每场景和三种子汇总。重复验证脚本也进入新 source hash，旧文件不改。
2. 新建独立 G1 输出；公共参数严格按 key/dtype/shape 从各 seed **initial** 映射并核指纹。缺/多 key 拒绝，不用 permissive load。
3. 每个 seed 在独立进程内先执行与 C1 相同的 seed/后端设置（matmul highest、CUDA matmul/cuDNN TF32 关闭、cuDNN benchmark 关闭），再重放 C1 final 的 train/dev 指标；容差仍为 rtol1e-4/atol1e-6。回放保存实际/期望指标和后端设置，失败明确打印场景、指标与差值。trained C1 final 仅用于回放和独立 **零迭代等价探针**，随后销毁；绝不作为门控或正式初始化。
4. 门控使用每个训练场景固定首个 clip，合计2 clips，200 updates。只在这些训练 clips 上评价初末 loss，要求 final/initial ≤ .5；首步 matcher/classifier/encoder/三个 GRU/index_head 七模块梯度均有限且正。开发集只用于 C1 重放，不进入门控优化或通过阈值。
5. 三个 seed 的门控全部有效，才允许创建任何正式输出。门控参数不保存、不复用；额外 **600 次 disposable updates** 单独记账，不算作正式 warmup。
6. 正式训练重新加载保存的 G1 initial 和空优化器，各1000步，原训练顺序。任何失败/超时/证据变化立即停止，不自动放宽阈值、追加步数、换种子或续训。
7. 三组完整完成后输出 `G1_FINAL_STEP=1000` 与 C1/G1 配对表；正式 checkpoint/metrics/init/progress/trace 全部验证后才写完成证书。

入口默认 GPU5，每段先拒绝已有 GPU 计算进程或显存占用超过2GiB；不终止其他作业。每个 gate/formal 最长3600s，超时不获得完成证书。若准备后 GPU 变忙，保留新目录，后续显式继续尚未启动的阶段，**不要盲目重跑 prepare**。没有自动 resume/覆盖。

## 4. 已有验证与执行边界

- [头部测试](test/test_stereogru_corrected_head.py)：raw/GEV 真实入参、两支梯度、坐标 detach、8轮顺序、独立 ConvGRU 公式、严格初值映射、有界更新、图像依赖与坏相机拒绝。
- [流程测试](test/test_stereogru_corrected_workflow.py)：合成三种子 C0/C1→门控→重置→G1→汇总，旧 artifact 字节不变；拒绝缺门控、他 seed 坏 cache、空/缺梯度、非有限/负 parity、错误支持域/scene/summary/hash。
- 本轮新旧完整相关回归 **139 passed in72.24s**；shell 语法和 diff 格式检查通过；独立只读复核无剩余所列阻塞项。
- 正式尺寸 CPU 合成 smoke（随机特征，不加载真实图像/backbone）：4帧256²、D32/G8/hidden128、8轮，输出4×1×256×256；零迭代与 C1 完全相同。loss .1381412，输出范围 [.2076815,.5252613]，七个关键模块梯度有限且正；计算段约2.93s，不是 GPU 吞吐估计。
- **未验证项：真实 VKITTI 两 clip 门控是否通过、GPU 显存/总耗时、真实三种子 G1 效果。** 本地合成通过不保证实测门控或精度；失败如实保留，不能推断 GRU 已提点。
- 阿里云只能由用户执行；当前环境没有该机器的终端/SSH 访问。发布快照仅归档 model/loss/dataset/config/scripts，不切远端工作树，不触碰 checkpoint。

## 5. 2026-09-11：独立进程回放初始化修复

首轮失败记录：用户运行 `8918b769a82a06e278774f65ee2c1e1e50f6eba4`，输出为 `/mnt/data/PROJECT_CHEN/code/PP-DPT/stereogru_corrected_gru_FGClc5Gx/run`。三种子初值/参数校验通过，seed0 在 `C1 final metrics did not replay: train/index_l1` 停止。该位置在创建门控优化器之前，**没有 G1 优化更新，更没有正式结果**；旧错误未输出实际/期望值，不能编造误差幅度或断言远端问题已解决。

- 确定的代码遗漏：原 `train_arm` 在模型前向前执行 `seed_everything`（同时设置 FP32/TF32），新 `run_gate` 仅在随后构造 GRU 时才执行。shell 的 prepare/gate 是不同 Python 进程，准备阶段的后端开关不会继承。cuDNN TF32 等设置因而可能与已完成 C1 不同。
- 上轮 CPU 同进程流程测试继承了 prepare 的全局开关，未覆盖该入口差异。本轮增加三个 seed 的回归，主动重置为 TF32-enabled 状态，在**第一次 C1 evaluate**处断言已恢复原设置；另增加独立子进程检查及故意污染全局设置的完整合成流程。容差边界用例保留 rtol1e-4/atol1e-6，拒绝越界差值。
- 修复仅调整新 runner 的回放初始化和日志；模型、loss、数据、seed、初始权重映射、正式预算与旧基线源码均不变。不靠放宽容差让门控通过。
- 本轮完整相关回归 **148 passed in59.21s**，shell 语法与 diff 检查通过。这证明入口设置已修，不代替真实 H20 回放验证。
- **重启使用新源码快照、新 G1 目录**：源码 hash 已变，不能修改旧已准备 contract 或删除失败 gate 来续跑。首轮失败目录保留；只复用原已完成 C0/C1 的 manifest/cache/initial/证书，不重训基线、不追加试验 seed。新入口通过 C1 回放与所有既定门控后，才会运行原定三组正式训练。