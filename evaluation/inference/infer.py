import argparse
import os
import cv2
import json
import torch
from tqdm import tqdm
import numpy as np
import sys  
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)  
if root_dir not in sys.path:
    sys.path.append(root_dir)
from model.gemdepth import GemDepth
from model.backbone_registry import available_backbone_names
from model.decoder_registry import available_decoder_names
from model.factory import build_gemdepth_from_config
from omegaconf import OmegaConf
from protocol import infer_video_with_protocol, resolve_inference_clip_len

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_path', type=str, default='')
    parser.add_argument('--json_file', type=str, default="")
    parser.add_argument('--datasets', type=str, nargs='+', default=['kitti'])
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitl'])
    parser.add_argument('--ckpt', type=str, default='./checkpoint/gemdepth.pth',
                        help='Path to the model checkpoint (model-only state dict).')
    parser.add_argument('--backbone', type=str, default='DINOv2Backbone',
                        choices=available_backbone_names(),
                        help='Exact registered backbone class name.')
    parser.add_argument('--decoder', type=str, default='DPTHeadTemporal',
                        choices=available_decoder_names(),
                        help='Exact decoder implementation name; must match the checkpoint.')
    parser.add_argument('--config', type=str, default='',
                        help='Experiment config yaml. When set, the model is built via the shared '
                             'factory (model/factory.py) so inference matches training EXACTLY '
                             '(backbone / decoder / use_temporal / use_gem / use_astt / lora). '
                             'Required for the scratch encoder+decoder experiments.')

    args = parser.parse_args()
    clip_len = None
    for dataset in args.datasets:
        with open(args.json_file, 'r') as fs:
            path_json = json.load(fs)
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        if args.config:
            # Build identically to training. Backbone weights come from the checkpoint
            # (strict load below), so skip re-downloading pretrained backbones.
            cfg = OmegaConf.load(args.config)
            gemdepth = build_gemdepth_from_config(cfg, load_backbone_pretrained=False)
            clip_len = resolve_inference_clip_len(cfg)
            if clip_len is not None:
                print(
                    f"[inference] scratch protocol: non-overlapping "
                    f"clip_len={clip_len} (matches training T)")
        else:
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            }
            gemdepth = GemDepth(
                **model_configs[args.encoder],
                backbone=args.backbone,
                decoder=args.decoder,
            )
        checkpoint = torch.load(args.ckpt, map_location='cpu',weights_only=False)
        gemdepth.load_state_dict(checkpoint, strict=True)
        gemdepth = gemdepth.to(DEVICE).eval()
        json_data = path_json[dataset]
        root_path = os.path.dirname(args.json_file)
        for data in tqdm(json_data):
             for key in data.keys():
                value = data[key]
                infer_paths = []
                videos = []
                for images in value:
                    image_path = os.path.join(root_path, images['image'])
                    infer_path = (args.infer_path + '/'+ dataset +'/' + images['image']).replace('.jpg', '.npy').replace('.png', '.npy')
                    infer_paths.append(infer_path)
                    img = cv2.imread(image_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    videos.append(img)
                videos = np.stack(videos, axis=0)
                target_fps=1
                depths, fps = infer_video_with_protocol(
                    gemdepth, videos, target_fps,
                    input_size=args.input_size, device=DEVICE, fp32=True,
                    clip_len=clip_len, dataset=dataset, sequence=key)
                if depths.shape[0] != len(infer_paths):
                    raise ValueError(
                        f"Inference frame-count mismatch: output={depths.shape[0]} "
                        f"expected={len(infer_paths)} dataset={dataset} sequence={key}")
                for i in range(len(infer_paths)):
                    infer_path = infer_paths[i]
                    os.makedirs(os.path.dirname(infer_path), exist_ok=True)
                    depth = depths[i]
                    np.save(infer_path, depth)
                    