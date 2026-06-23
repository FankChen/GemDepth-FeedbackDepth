# VKITTI 1.3.1 Dataset Setup Guide

## 状态

Google Drive 镜像链接需要特殊权限，已经确认不可靠。当前使用 **Naver Labs 官方直链**下载 VKITTI 1.3.1。

官方页面：

https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-1

GemDepth 训练只需要三个组件：

| 组件 | 官方文件 | 大小 |
| --- | --- | --- |
| RGB 图像 | `vkitti_1.3.1_rgb.tar` | 约 14GB |
| Depth GT | `vkitti_1.3.1_depthgt.tar` | 约 5.1GB |
| Camera extrinsics | `vkitti_1.3.1_extrinsicsgt.tar.gz` | 约 1.1MB |

## 方法一：提交集群下载任务（推荐）

在登录节点执行：

```bash
cd /home/izi2sgh/MYDATA/quanjie/liren/depth_baselines/GemDepth
bsub < scripts/download_vkitti.bsub
```

下载目标目录：

```bash
/home/izi2sgh/MYDATA/quanjie/liren/datasets/vkitti_1.3.1
```

日志位置：

```bash
/home/izi2sgh/MYDATA/quanjie/liren/depth_baselines/GemDepth/logs/download_vkitti.<JOB_ID>.stdout
/home/izi2sgh/MYDATA/quanjie/liren/depth_baselines/GemDepth/logs/download_vkitti.<JOB_ID>.stderr
```

监控命令：

```bash
bjobs
bjobs -l <JOB_ID>
tail -f logs/download_vkitti.<JOB_ID>.stdout
```

脚本会使用 `scripts/download_vkitti.py` 执行：

1. 从 Naver 官方 URL 下载三个 archive；
2. 解压到目标目录；
3. 保留 archive 到 `archives/` 子目录，便于断点续传和复用；
4. 下载完成后验证目录结构。

## 方法二：手动下载官方文件

如果集群节点无法访问外网，可以在本地浏览器下载以下三个官方文件，再上传到集群：

```text
https://download.europe.naverlabs.com/virtual-kitti-1.3.1/vkitti%5F1.3.1%5Frgb.tar
https://download.europe.naverlabs.com/virtual-kitti-1.3.1/vkitti%5F1.3.1%5Fdepthgt.tar
https://download.europe.naverlabs.com/virtual-kitti-1.3.1/vkitti%5F1.3.1%5Fextrinsicsgt.tar.gz
```

上传到：

```bash
/home/izi2sgh/MYDATA/quanjie/liren/datasets/vkitti_1.3.1/archives/
```

然后在集群上解压：

```bash
TARGET_DIR=/home/izi2sgh/MYDATA/quanjie/liren/datasets/vkitti_1.3.1
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"
tar -xf archives/vkitti_1.3.1_rgb.tar -C "$TARGET_DIR"
tar -xf archives/vkitti_1.3.1_depthgt.tar -C "$TARGET_DIR"
tar -xzf archives/vkitti_1.3.1_extrinsicsgt.tar.gz -C "$TARGET_DIR"
```

## 验证结构

下载完成后执行：

```bash
cd /home/izi2sgh/MYDATA/quanjie/liren/depth_baselines/GemDepth
/home/izi2sgh/MYDATA/quanjie/liren/envs/gemdepth/bin/python scripts/download_vkitti.py \
  --target-dir /home/izi2sgh/MYDATA/quanjie/liren/datasets/vkitti_1.3.1 \
  --verify-only
```

正确结构应为：

```text
/home/izi2sgh/MYDATA/quanjie/liren/datasets/vkitti_1.3.1/
├── archives/
│   ├── vkitti_1.3.1_rgb.tar
│   ├── vkitti_1.3.1_depthgt.tar
│   └── vkitti_1.3.1_extrinsicsgt.tar.gz
├── vkitti_1.3.1_rgb/
├── vkitti_1.3.1_depthgt/
└── vkitti_1.3.1_extrinsicsgt/
```

## GemDepth 训练配置注意事项

`dataset/dataset_mix.py` 里判断 VKITTI 的逻辑是：只要 `data_dir` 字符串包含 `vkitti`，它会在该路径下拼接：

```text
<data_dir>/vkitti_1.3.1_rgb
<data_dir>/vkitti_1.3.1_depthgt
<data_dir>/vkitti_1.3.1_extrinsicsgt
```

因此 VKITTI-only 训练脚本应传：

```bash
dataset.train.data_dirs="['/home/izi2sgh/MYDATA/quanjie/liren/datasets/vkitti_1.3.1/']"
```

现有脚本 `scripts/train_stage1_vkitti_only.bsub` 已按这个路径设置。

## 下一步

完成 VKITTI 下载和验证后：

```bash
cd /home/izi2sgh/MYDATA/quanjie/liren/depth_baselines/GemDepth
bsub < scripts/smoke_train.bsub
bsub < scripts/train_stage1_vkitti_only.bsub
```

如果需要断点续训：

```bash
bsub < scripts/train_stage1_vkitti_only.bsub -resume
```

---

**最后更新**: 2026-06-08
