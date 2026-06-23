"""Full-model GPU smoke test: builds GemDepth with both head types, loads the pretrained
weights, and runs a tiny forward pass to validate the head_type plumbing end to end.

Run on a GPU node:
    cd /home/izi2sgh/MYDATA/quanjie/liren/depth_baselines/GemDepth
    /home/izi2sgh/MYDATA/quanjie/liren/envs/gemdepth/bin/python scripts/smoke_full_model.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gemdepth import GemDepth

CKPT = "./checkpoint/gemdepth.pth"
VITL = dict(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])


def build(head_type):
    model = GemDepth(**VITL, head_type=head_type).cuda()
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    em_only = [m for m in missing if 'depth_heads' not in m and 'error_encoders' not in m]
    print(f"[{head_type}] missing={len(missing)} unexpected={len(unexpected)} non_errormap_missing={len(em_only)}")
    assert len(em_only) == 0 and len(unexpected) == 0
    return model


def run(head_type):
    model = build(head_type)
    model.eval()
    B, T, H, W = 1, 4, 98, 98
    x = torch.rand(B, T, 3, H, W, device='cuda')
    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            depth, pose_enc_list, extrinsic, intrinsic = model(x)
    print(f"[{head_type}] depth={tuple(depth.shape)} extrinsic={tuple(extrinsic.shape)} intrinsic={tuple(intrinsic.shape)}")
    if head_type == 'errormap':
        aux = model.head.aux_depths
        print(f"[{head_type}] aux_depths={[tuple(a.shape) for a in aux]}")
        assert len(aux) == len(model.head.error_stages)
    assert depth.shape[:2] == (B, T)
    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    assert torch.cuda.is_available(), "this smoke test needs a GPU"
    run('temporal')
    run('errormap')
    print("FULL-MODEL SMOKE PASSED")
