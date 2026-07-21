"""Probe: can timm build DINOv3 ViT-S+/16 and ConvNeXt-S and load our local .pth?

Prints the exact timm model names, load missing/unexpected keys, and multi-scale
feature-map shapes (what the DPT head will consume). Run on a GPU node:
    $PY test/probe_dinov3.py
"""
import os, sys
import torch
import timm

CKPT_VIT = 'checkpoint/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth'
CKPT_CNX = 'checkpoint/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth'


def list_names():
    print('timm', timm.__version__)
    for pat in ('*vit_small*dinov3*', '*convnext_small*dinov3*'):
        print(f'[names] {pat}:', timm.list_models(pat))


def try_build(candidates, ckpt, is_vit):
    for name in candidates:
        try:
            if is_vit:
                m = timm.create_model(
                    name, pretrained=True, num_classes=0,
                    pretrained_cfg_overlay=dict(file=ckpt))
                print(f"[build] OK {name}")
                m.eval()
                x = torch.randn(1, 3, 224, 224)
                feats = m.forward_intermediates(
                    x, indices=4, norm=False, output_fmt='NCHW',
                    intermediates_only=True)
                print('  [vit intermediates]', [tuple(f.shape) for f in feats])
            else:
                # ConvNeXt: raw DINOv3 ckpt has extra keys (norms.3) -> load non-strict.
                from timm.models.convnext import checkpoint_filter_fn
                m = timm.create_model(name, pretrained=False, num_classes=0)
                raw = torch.load(ckpt, map_location='cpu', weights_only=False)
                raw = raw.get('model', raw) if isinstance(raw, dict) else raw
                filt = checkpoint_filter_fn(raw, m)
                miss, unexp = m.load_state_dict(filt, strict=False)
                print(f"[build] OK {name}  missing={len(miss)} unexpected={len(unexp)}")
                print('   unexpected:', list(unexp)[:6])
                m.eval()
                x = torch.randn(1, 3, 256, 256)
                feats = m.forward_intermediates(
                    x, indices=4, norm=False, output_fmt='NCHW',
                    intermediates_only=True)
                print('  [convnext intermediates]', [tuple(f.shape) for f in feats])
            return name
        except Exception as e:
            import traceback
            print(f"[build] FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
    return None


if __name__ == '__main__':
    list_names()
    print('\n=== ViT-S+/16 ===')
    vit_cands = [
        'vit_small_plus_patch16_dinov3.lvd1689m',
        'vit_smallplus_patch16_dinov3.lvd1689m',
        'vit_small_patch16_dinov3.lvd1689m',
    ]
    try_build(vit_cands, CKPT_VIT, is_vit=True)
    print('\n=== ConvNeXt-S ===')
    cnx_cands = [
        'convnext_small.dinov3_lvd1689m',
        'convnextv2_small.dinov3_lvd1689m',
    ]
    try_build(cnx_cands, CKPT_CNX, is_vit=False)
    print('\nPROBE DONE')
