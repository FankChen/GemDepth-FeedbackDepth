"""Bounded training-only calibrated volume baseline, no GEM or GRU method arm.

Cache clean, frozen backbone features; optimise only the registered volume-only
head on fixed training clips. Report unaligned metric depth on those SAME clips
solely as an overfit diagnostic. It is neither generalisation nor RGB-only SOTA.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.stereogru_calibration import load_training_clips  # noqa: E402
from loss.objective_registry import build_objective  # noqa: E402
from model.backbone_registry import build_backbone  # noqa: E402
from model.decoder_registry import build_decoder, get_decoder_class  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with open(path, "x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def load_clean_backbone(name, path):
    """Validate every feature-producing tensor, never swallow arbitrary mismatches."""
    from timm.models.convnext import checkpoint_filter_fn

    backbone = build_backbone(name, pretrained=False, lora=False)
    if getattr(backbone, "feature_format", None) != "pyramid":
        raise ValueError("This calibrated control requires the native ConvNeXt pyramid")
    blob = torch.load(path, map_location="cpu", weights_only=True)
    blob = blob.get("model", blob)
    state = checkpoint_filter_fn(blob, backbone.model)
    # The official export has an additional final norm not used by timm's native
    # norm=False intermediates. This exact extra pair was checked locally. No
    # missing stem/stage tensor, LoRA checkpoint, or other unexpected key is allowed.
    unused = {"norms.3.weight", "norms.3.bias"} - set(backbone.model.state_dict())
    ignored = sorted(set(state) & unused)
    state = {key: value for key, value in state.items() if key not in unused}
    backbone.model.load_state_dict(state, strict=True)
    return backbone.requires_grad_(False).eval(), {"ignored_unused_export_norm": ignored,
                                                  "loaded_tensors": len(state)}


def predict(head, clip):
    images = clip["images"]
    patch_size = head.volume_stride // 2 ** head.volume_level
    return head(clip["features"], images.shape[-2] // patch_size,
                images.shape[-1] // patch_size, images.shape[1], images=images,
                extrinsics=clip["extrinsics"], intrinsics=clip["intrinsics"],
                geometry_gauge="metric").unflatten(0, images.shape[:2])


@torch.no_grad()
def evaluate_overfit(head, clips, objective):
    head.eval()
    rows = []
    absolute_sum = squared_sum = delta_sum = index_sum = 0.
    count = 0
    for clip in clips:
        q = predict(head, clip)
        loss = objective(q, clip["depth"], clip["mask"])
        inverse = 1. / objective.depth_max + q * (1. / objective.depth_min - 1. / objective.depth_max)
        if not torch.isfinite(inverse).all() or (inverse <= 0).any():
            raise FloatingPointError("Invalid physical inverse-depth prediction; no clamping fallback")
        depth = inverse.reciprocal()
        target = clip["depth"]
        valid = (clip["mask"].bool() & torch.isfinite(target)
                 & (target >= objective.depth_min) & (target <= objective.depth_max))
        pred, gt = depth[valid], target[valid]
        pixels = int(valid.sum())
        absolute_sum += float(((pred - gt).abs() / gt).sum())
        squared_sum += float((pred - gt).square().sum())
        delta_sum += float((torch.maximum(pred / gt, gt / pred) < 1.25).sum())
        index_sum += float(loss["index_l1"]) * pixels
        count += pixels
        rows.append({**head.last_diagnostics, "index_l1": float(loss["index_l1"]),
                     "valid_pixels": pixels, "q_min": float(q.min()), "q_max": float(q.max())})
    return {"index_l1": index_sum / count, "absrel": absolute_sum / count,
            "rmse": (squared_sum / count) ** .5, "delta1": delta_sum / count,
            "valid_pixels": count, "clips": rows}


def run(cfg, weight_path, data_root, output, device):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Output must be a new empty directory; no resume/overwrite")
    if int(cfg.steps) < 1 or int(cfg.steps) > 1000:
        raise ValueError("Implementation pilot is bounded to 1..1000 steps, not a full-training launcher")
    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    start = time.monotonic()
    write_json(output / "config.json", OmegaConf.to_container(cfg, resolve=True))
    backbone, load_info = load_clean_backbone(str(cfg.backbone.name), weight_path)
    backbone = backbone.to(device).eval()
    clips = load_training_clips(data_root, **dict(cfg.dataset), seed=seed)
    manifest = [clip["manifest"] for clip in clips]
    write_json(output / "manifest.json", manifest)
    for clip in clips:
        for key in ("images", "depth", "mask", "intrinsics", "extrinsics"):
            clip[key] = clip[key].to(device)
        with torch.no_grad():
            clip["features"] = [value.detach().float() for value in backbone(clip["images"].flatten(0, 1))]
    torch.manual_seed(seed)  # Head initialisation does not depend on subset enumeration.
    head = build_decoder(get_decoder_class(str(cfg.decoder.name)), dict(cfg.decoder.kwargs),
                         in_channels_list=backbone.embed_dims, patch_size=backbone.patch_size).to(device)
    objective = build_objective(str(cfg.objective.name), dict(cfg.objective.kwargs)).to(device)
    if (head.depth_min, head.depth_max) != (objective.depth_min, objective.depth_max):
        raise ValueError("Head hypotheses and supervision must use identical depth bounds")
    if head.iters != 0 or getattr(head, "output_space", None) != "normalized_inverse_depth_index":
        raise ValueError("Baseline-only entry; GRU arms require a separately completed matched baseline")
    import timm
    source_files = ("model/dpt_calibrated_volume_only_convnext.py", "model/dpt_cost_volume_convnext.py",
                    "model/util/cost_volume.py", "model/util/warp.py", "model/backbones.py",
                    "dataset/stereogru_calibration.py", "dataset/vkitti_split.py",
                    "loss/objective_calibrated_index.py", "scripts/stereogru_calibrated_baseline.py")
    data_files = sorted({p for row in manifest for key in ("rgb", "depth", "calibration") for p in row[key]})
    provenance = {"torch": torch.__version__, "timm": timm.__version__, "device": str(device),
                  "weights": str(Path(weight_path).resolve()), "weights_sha256": sha256(weight_path),
                  "loading": load_info, "source_sha256": {p: sha256(ROOT / p) for p in source_files},
                  "input_sha256": {p: sha256(p) for p in data_files},
                  "backbone_frozen": not any(p.requires_grad for p in backbone.parameters()),
                  "backbone_eval": not backbone.training, "lora": False, "gem": False,
                  "head_parameters": sum(p.numel() for p in head.parameters()),
                  "input_protocol": "GT metric K/T; training-only fixed clips; native clean frozen backbone"}
    write_json(output / "provenance.json", provenance)
    with open(output / "initial_head.pth", "xb") as handle:
        torch.save(head.state_dict(), handle)
    del backbone
    initial = evaluate_overfit(head, clips, objective)
    print("INITIAL", json.dumps(initial), flush=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(cfg.learning_rate),
                                 weight_decay=float(cfg.weight_decay))
    first_matcher_grad = None
    with open(output / "progress.jsonl", "x") as log:
        for step in range(1, int(cfg.steps) + 1):
            head.train()
            clip = clips[(step - 1) % len(clips)]
            optimizer.zero_grad(set_to_none=True)
            prediction = predict(head, clip)
            loss = objective(prediction, clip["depth"], clip["mask"])["total_loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite calibrated loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), float(cfg.grad_clip), error_if_nonfinite=True)
            if first_matcher_grad is None:
                first_matcher_grad = sum(float(p.grad.square().sum()) for p in head.matcher.parameters() if p.grad is not None) ** .5
                if first_matcher_grad == 0:
                    raise RuntimeError("No matcher gradient even with calibrated cameras; inspect before training")
            optimizer.step()
            if step == 1 or step % int(cfg.log_every) == 0 or step == int(cfg.steps):
                row = {"step": step, "loss": float(loss), "grad_norm_before_clip": float(grad_norm),
                       **head.last_diagnostics}
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
    final = evaluate_overfit(head, clips, objective)
    report = {"status": "completed_implementation_pilot", "protocol": str(cfg.protocol),
              "steps": int(cfg.steps), "clips": len(clips), "initial": initial, "final": final,
              "first_matcher_grad_after_clip": first_matcher_grad,
              "elapsed_seconds": time.monotonic() - start, "method_launched": False,
              "interpretation": "Training-clip overfit only, GT cameras, no affine alignment. NOT generalisation/RGB-only/SOTA. Review before a matched full baseline or GRU arm."}
    write_json(output / "summary.json", report)
    with open(output / "baseline_head.pth", "xb") as handle:
        torch.save({"model_state_dict": head.state_dict(), "config": OmegaConf.to_container(cfg, resolve=True),
                    "manifest": manifest, "steps": int(cfg.steps)}, handle)
    print("SUMMARY", json.dumps(report, indent=2), flush=True)
    print(f"Output directory: {output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config/stereogru/calibrated_volume_only.yaml"))
    parser.add_argument("--backbone-weights", required=True)
    parser.add_argument("--vkitti-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Real-data pilot requires an idle CUDA GPU; CPU synthetic tests are separate")
    run(OmegaConf.load(args.config), args.backbone_weights, args.vkitti_root,
        args.output, torch.device("cuda"))


if __name__ == "__main__":
    main()