# GemDepth 阿里云一键复现 · 分步操作手册（中文）

> 适用对象：拿到一台**阿里云 GPU 服务器（ECS，带 NVIDIA 卡）**，想从零把 GemDepth
> 的训练四/六臂跑起来。全程**复制粘贴命令**即可，每步都给了「预期输出」和「常见报错」。
>
> 你不需要懂 conda / CUDA —— 第 2 步的体检脚本会自动帮你探测并告诉你缺什么。

---

## 0. 名词与总览（30 秒）

| 名词 | 含义 |
| --- | --- |
| **臂 (arm)** | 一个受控实验配置。共 6 个：`baseline` `errormap` `em_rgb` `em_feat` `em_hog` `em_rgbfeat` |
| **VKITTI_ROOT** | 训练数据根目录（解压后含 `Scene01/ Scene02/ ...`） |
| **gemdepth.pth** | 2.84GB 预训练权重，**不在 git 仓库里**，需要你单独传上去 |
| **EVAL_ROOT** | KITTI 评测数据（**现在先不管**，只跑训练用不到） |

整体流程：
```
git clone 代码  →  体检 check  →  装环境 install  →  放权重 gemdepth.pth
   →  准备 VKITTI 数据 + 设 VKITTI_ROOT  →  verify 验证  →  CPU 冒烟  →  起训  →  监控
```

---

## 1. 登录阿里云并拉取代码

在你的电脑上 SSH 登录阿里云 ECS（把 `<ECS公网IP>` 换成你的）：
```bash
ssh root@<ECS公网IP>
```

选一个数据盘目录（**别放系统盘**，系统盘通常很小），克隆代码：
```bash
cd /mnt/data            # 或你的数据盘挂载点；没有就用 ~/
git clone -b feat/aliyun-repro https://github.com/FankChen/GemDepth-FeedbackDepth.git GemDepth
cd GemDepth
```

> ⚠️ 这里用 **GitHub 公网仓库**（feedback 远端）。Bosch 企业镜像
> `github.boschdevcloud.com/IZI2SGH/PP-DPT` 在阿里云公网**访问不到**，别用它 clone。
>
> **预期输出**：`Cloning into 'GemDepth'...`，最后 `Resolving deltas: 100%`。
>
> **常见报错**：
> - `Connection timed out` / 卡住：阿里云访问 GitHub 偶尔慢。重试，或给 git 配代理；
>   也可以先在你本机 clone 再用第 4 步的 scp 方式传整个目录上去。

---

## 2. 一键体检（不改动任何东西，只是探测）

```bash
bash scripts/setup_aliyun.sh check
```

这一步会自动告诉我们：操作系统、有没有 conda、python 版本、**GPU 型号、驱动版本、
驱动支持的最高 CUDA、GPU 数量**、磁盘空间、权重在不在、VKITTI 在不在。

> **预期输出**（示例，你的会不同）：
> ```
> == 3) NVIDIA 驱动 / GPU / CUDA ==
>   [ OK ]  驱动版本: 535.xxx | 驱动支持的最高 CUDA: 12.2
>   [ OK ]  CUDA 12.2 >= 12.1，满足 torch 2.3.1+cu121
>   [INFO]  GPU 列表:
>         0, NVIDIA A100-SXM4-40GB, 40960 MiB
>   [ OK ]  可见 GPU 数量: 1
> ...
> 体检结论: 发现 N 项待处理 ⚠️
> ```
>
> **请把这一整屏输出发给我**，我据此确认你的 GPU/CUDA 是否够用、要不要装 conda。
>
> **关键判断**：
> - 若出现 `[MISS] CUDA x.x < 12.1` → 驱动太旧，需在阿里云控制台换 GPU 驱动镜像或升级驱动（告诉我，我给你具体步骤）。
> - `nvidia-smi 不存在` → 这台机器没装 NVIDIA 驱动 / 不是 GPU 实例。

---

## 3. 安装环境（创建虚拟环境 + 装依赖）

```bash
bash scripts/setup_aliyun.sh install
```

脚本逻辑（你不用记）：
- **有 conda** → 自动建 `conda create -n gemdepth python=3.10` 并装依赖。装完用 `conda activate gemdepth`。
- **没 conda** → 用 `python3.10/3.11` 建 venv（目录 `.venv-gemdepth/`）。装完用 `source .venv-gemdepth/bin/activate`。

> **中国大陆加速**（强烈建议，否则 pip 很慢）：
> ```bash
> PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple bash scripts/setup_aliyun.sh install
> ```
>
> **预期输出**：最后一行 `[ OK ] 安装完成。激活方式: conda activate gemdepth`（或 `source .venv-gemdepth/bin/activate`）。
>
> **常见报错**：
> - `默认 python3=3.9 < 3.10`：系统没有 3.10/3.11。两个办法：
>   ① 装 conda（推荐，最省事）：
>   ```bash
>   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
>   bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
>   source $HOME/miniconda3/etc/profile.d/conda.sh
>   ```
>   然后重跑 `bash scripts/setup_aliyun.sh install`。
>   ② 用系统包管理器装 python3.10（Ubuntu: `apt install python3.10 python3.10-venv`）。
> - `torch ... not found`：脚本会自动用 PyTorch 官方索引兜底重试；若仍失败，手动：
>   ```bash
>   pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
>   ```

**装完先激活环境**（后续所有命令都在激活状态下执行）：
```bash
conda activate gemdepth          # 或  source .venv-gemdepth/bin/activate
```

---

## 4. 放置预训练权重 `checkpoint/gemdepth.pth`（2.84GB）

这是**训练必须**的文件（里面打包了 DINOv2 编码器权重，无需联网下载）。
它不在 git 里，需要你从已有的地方传上来。目标位置固定为：
```
<仓库>/checkpoint/gemdepth.pth
```

先建目录：
```bash
mkdir -p checkpoint
```

**传输方式三选一**（我会根据你「文件现在在哪」帮你定，先看推荐）：

### 方式 A（推荐）：阿里云 OSS（断点续传、内网下载快，最适合 2.84GB）
1. 在能访问到该文件的机器（你的本地 Windows / Bosch 集群）上装 `ossutil` 并配置 AK/SK。
2. 上传到你的 OSS bucket：
   ```bash
   ossutil cp gemdepth.pth oss://<你的bucket>/gemdepth.pth
   ```
3. 在阿里云 ECS 上下载（同地域内网极快）：
   ```bash
   ossutil cp oss://<你的bucket>/gemdepth.pth checkpoint/gemdepth.pth
   ```

### 方式 B：从你本地电脑直接 scp 上传（文件就在你本机时最简单）
在**你本地电脑**（不是 ECS）执行：
```bash
# Windows PowerShell 或 Linux/Mac 终端
scp /本地路径/gemdepth.pth root@<ECS公网IP>:/mnt/data/GemDepth/checkpoint/gemdepth.pth
```

### 方式 C：文件只在 Bosch 集群上
Bosch 集群与阿里云之间通常不能直连。先把文件 scp 到你**本地电脑**，再用方式 A 或 B 上传：
```bash
# 在本地电脑执行：先从 Bosch 拉到本地
scp izi2sgh@rng-dl01-login.de.bosch.com:/home/izi2sgh/MYDATA/quanjie/liren/depth_baselines/GemDepth/checkpoint/gemdepth.pth ./gemdepth.pth
# 再用方式 A(OSS) 或 方式 B(scp 到 ECS) 上传
```

> **传完务必校验大小**（应为 2844171337 字节 ≈ 2.7G）：
> ```bash
> ls -l checkpoint/gemdepth.pth
> stat -c%s checkpoint/gemdepth.pth      # 期望: 2844171337
> ```
> 大小不对 = 没传完，重传。

---

## 5. 准备 VKITTI 训练数据并设置 `VKITTI_ROOT`

你说数据（tar 包）在阿里云 `/mnt/data/vkitti2/`。需要解压成下面这种结构：
```
<VKITTI_ROOT>/
  Scene01/<variation>/frames/rgb/Camera_0/rgb_00000.jpg
  Scene01/<variation>/frames/depth/Camera_0/depth_00000.png
  Scene01/<variation>/extrinsic.txt
  Scene02/ ... Scene20/
```

### 5.1 解压（如果还没解压）
```bash
cd /mnt/data/vkitti2
tar -xf vkitti_2.0.3_rgb.tar
tar -xf vkitti_2.0.3_depth.tar
tar -xzf vkitti_2.0.3_textgt.tar.gz
```
> 这三个包解压后**没有外层目录**，直接出 `Scene01..Scene20`。三个包会合并到同一层。
>
> 如果你只有 `vkitti_2.0.3_textgt.tar.gz`（位姿），还缺 rgb/depth 两个大包，需要补传或重新下载
> （rgb≈7.5GB、depth≈8.1GB）。告诉我，我给下载/传输步骤。

### 5.2 设置环境变量
```bash
export VKITTI_ROOT=/mnt/data/vkitti2/        # 指向含 Scene01.. 的那一层
```
> 想长期生效就写进 `~/.bashrc`：
> ```bash
> echo 'export VKITTI_ROOT=/mnt/data/vkitti2/' >> ~/.bashrc && source ~/.bashrc
> ```

### 5.3 回到仓库再体检一次（确认数据被识别）
```bash
cd /mnt/data/GemDepth          # 回到代码仓库
bash scripts/setup_aliyun.sh check
```
> **预期**：第 6 节变成
> `[ OK ] 找到 frames/rgb/Camera_0 结构 (Scene 目录数: 5)`。
> 若仍 `[MISS] 未找到 .../frames/rgb/Camera_0` → `VKITTI_ROOT` 指错层了，
> 用 `find /mnt/data/vkitti2 -type d -name Camera_0 | head` 找到真实位置，把 `VKITTI_ROOT`
> 指到 `Camera_0` 上面第 4 层（即 `Scene*` 的父目录）。

---

## 6. 验证 PyTorch 能用 GPU

```bash
bash scripts/setup_aliyun.sh verify
```
> **预期输出**：
> ```
>   [ OK ]  torch          2.3.1
>   torch.version.cuda      = 12.1
>   torch.cuda.is_available = True
>       GPU0: NVIDIA A100-...
>   [ OK ]  GPU 矩阵乘法成功
>   [ OK ]  验证通过 ✅
> ```
> 若 `torch.cuda.is_available = False`：99% 是驱动 CUDA < 12.1 或驱动没装好。把第 2 步
> 体检输出发我。

---

## 7. CPU 冒烟测试（不占 GPU，确认代码与权重能加载）

这一步在 CPU 上快速验证模型能搭起来、权重能加载、各臂前向通：
```bash
export OMP_NUM_THREADS=4
python scripts/smoke_coattn.py          # 方案C 四臂 (em_*) 的协同注意力头
python scripts/smoke_errormap.py        # errormap-v1 头
python scripts/smoke_vkitti2_loader.py "$VKITTI_ROOT" 4   # 数据加载器（需 VKITTI 已解压）
```
> **预期**：每个脚本打印 `OK` / 初始恒等 `max_diff=0` / 样本 shape 正常，无 traceback。
>
> **常见报错**：
> - `FileNotFoundError: ./checkpoint/gemdepth.pth` → 回第 4 步放权重。
> - 卡很久不动 → 登录/CPU 机器线程超载，已在脚本里设 `set_num_threads`；确保
>   `export OMP_NUM_THREADS=4` 生效即可。

---

## 8. 起训

> 训练设定（所有臂一致）：冻结除 DPT 头外的全部参数，从 `gemdepth.pth` 微调，
> `batch=1, grad_accum=4, seq_len=16, crop=518, total_step=10000`，每 2000 步存一次 checkpoint 到
> `checkpoint/single_a100_<arm>/`，日志在 `logs/<arm>.log`。

### 8.1 先单臂验证流程跑通（建议第一次只跑 baseline）
```bash
bash scripts/train_single_a100.sh baseline
```
> **预期**：先打印 `nvidia-smi`，然后 `missing=0 unexpected=0`（baseline 精确复现权重），
> 接着出现 tqdm 进度条 `it/s`。看到稳定推进就说明跑通了。可以 `Ctrl-C` 停掉再用第 8.3 后台正式跑。

### 8.2 中途断了想续训
```bash
bash scripts/train_single_a100.sh baseline -resume
```

### 8.3 后台正式跑（断开 SSH 也不停）
单臂：
```bash
nohup bash scripts/train_single_a100.sh baseline > logs/baseline.log 2>&1 &
tail -f logs/baseline.log          # 看进度，Ctrl-C 只退出查看不杀训练
```

### 8.4 跑多臂（你要「越多越好」）
- **单 GPU**：只能**串行**一个个跑（一个跑完再跑下一个）：
  ```bash
  nohup bash scripts/run_arms.sh all > logs/run_all.log 2>&1 &
  tail -f logs/run_all.log
  ```
  > `all` = 6 个臂依次跑。也可只列要跑的：`bash scripts/run_arms.sh baseline errormap em_rgb`。
- **多 GPU**（比如 2 张卡）：让脚本把臂轮流分到各卡**并行**：
  ```bash
  GPUS="0 1" nohup bash scripts/run_arms.sh all > logs/run_all.log 2>&1 &
  ```
  > 臂会按 GPU0、GPU1、GPU0... 轮流后台并行。日志在 `logs/<arm>.gpu<n>.log`。

> 单 GPU 上 6 个臂全跑完耗时较长（每臂 1 万步）。如果只想先看方法对比，建议先跑
> `baseline` + `errormap` + `em_rgbfeat` 三个最有代表性的。

---

## 9. 监控训练

```bash
nvidia-smi                 # 看 GPU 占用/显存
tail -f logs/baseline.log  # 看某个臂的进度
ls -lh checkpoint/single_a100_baseline/   # 看 checkpoint 是否每 2000 步落盘
```
> 正常：显存被占用、tqdm 持续推进、`checkpoint_2000.pth / 4000.pth ...` 陆续出现，
> 训练结束生成 `final_model.pth`。

---

## 10.（以后）KITTI 评估 + 回填 SOTA 表

> 现在先不做。等你准备好 KITTI 评测数据（`datasets/gemdepth_eval/kitti/ + kitti_video.json`）
> 并 `export EVAL_ROOT=/path/to/gemdepth_eval` 后：
```bash
bash scripts/eval_kitti_arm.sh baseline      # 对每个训练好的臂各跑一次
bash scripts/eval_kitti_arm.sh errormap
# ... 其余臂同理
```
> 结果在 `output_eval/<arm>/results.txt`（absrel / rmse / delta1）。
> 把各臂数字回填到 `docs/ERRORMAP_DEVLOG.md` 的 SOTA 表。

---

## 11. 常见报错速查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `torch.cuda.is_available()=False` | 驱动 CUDA < 12.1 或没装驱动 | 升级阿里云 GPU 驱动；重跑第 2 步体检 |
| `默认 python3=3.9 < 3.10` | 系统 python 太旧 | 装 conda 或 python3.10（第 3 步） |
| `FileNotFoundError: gemdepth.pth` | 权重没放 | 第 4 步，校验大小 2844171337 |
| `未找到 frames/rgb/Camera_0` | VKITTI_ROOT 指错层 | `find ... -name Camera_0` 修正路径（第 5.3） |
| `CUDA out of memory` | 显存不足 | 配置已是 batch=1；换更大显存卡，或减小 `seq_len` |
| pip 安装极慢/超时 | 没用国内镜像 | 加 `PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple` |
| `xFormers not available` 告警 | 正常 | 代码用 PyTorch SDPA，**忽略即可**，不影响结果 |
| clone GitHub 超时 | 阿里云访问 GitHub 不稳 | 重试 / 配代理 / 本机 clone 后 scp 上传 |

---

## 一页速查（全程命令汇总）

```bash
# 1. 拉代码
cd /mnt/data && git clone -b feat/aliyun-repro https://github.com/FankChen/GemDepth-FeedbackDepth.git GemDepth && cd GemDepth
# 2. 体检
bash scripts/setup_aliyun.sh check
# 3. 装环境（国内加速）
PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple bash scripts/setup_aliyun.sh install
conda activate gemdepth   # 或 source .venv-gemdepth/bin/activate
# 4. 放权重（示例：OSS）
mkdir -p checkpoint && ossutil cp oss://<bucket>/gemdepth.pth checkpoint/gemdepth.pth
# 5. 数据
cd /mnt/data/vkitti2 && tar -xf vkitti_2.0.3_rgb.tar && tar -xf vkitti_2.0.3_depth.tar && tar -xzf vkitti_2.0.3_textgt.tar.gz
export VKITTI_ROOT=/mnt/data/vkitti2/ ; cd /mnt/data/GemDepth
# 6+7. 验证 + 冒烟
bash scripts/setup_aliyun.sh verify
export OMP_NUM_THREADS=4 && python scripts/smoke_coattn.py
# 8. 起训（单卡串行全臂）
nohup bash scripts/run_arms.sh all > logs/run_all.log 2>&1 & tail -f logs/run_all.log
```
