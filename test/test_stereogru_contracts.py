"""Pin the legacy StereoGRU contracts while diagnosing (not changing) old runs.

These checks distinguish correct primitives from experiment-level mismatches.
They deliberately do not replace the legacy model/loss with a new method arm.
"""

import os
import json
import sys
from unittest.mock import patch

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]

import model.dpt_cost_volume_convnext as costvol
import diagnose_cost_volume_oracle as oracle
from diagnose_cost_volume_oracle import trace_decoder
from loss.videoloss import VideoDepthLoss, compute_camera_loss
from model.tools.camera import CameraHead
from model.tools.pose_enc import pose_encoding_to_extri_intri
from model.util.cost_volume import VolumeIndexer, convex_upsample
from model.util.warp import plane_sweep_warp


def test_volume_indexer_preserves_pixel_order_and_independent_volumes():
    # Per-pixel codes catch H/W/batch flattening errors; a ramp checks sub-bin lookup.
    depth, height, width = 8, 2, 3
    pixel = torch.arange(2 * height * width).reshape(2, 1, 1, height, width) * 100.
    geometry = pixel + torch.arange(depth).reshape(1, 1, depth, 1, 1)
    raw = geometry * 3 + 7
    index = torch.full((2, 1, height, width), 2.5)
    sampled = VolumeIndexer(geometry, raw, num_levels=2, radius=1)(index)
    for level in range(2):
        # Pooling a ramp creates level-one values at 0.5, 2.5, 4.5, ... .
        center = index / (2 ** level)
        for offset in range(-1, 2):
            expected = pixel[:, 0, 0] + (center[:, 0] + offset) * 2 ** level
            expected += (2 ** level - 1) / 2
            channel = level * 6 + offset + 1
            assert torch.allclose(sampled[:, channel], expected, atol=1e-4)
            assert torch.allclose(sampled[:, channel + 3], expected * 3 + 7, atol=1e-4)


def test_inverse_depth_index_upsampling_does_not_multiply_by_stride():
    low = torch.full((1, 1, 3, 4), 5.)
    weights = torch.zeros(1, 9, 12, 16)
    weights[:, 4] = 1.  # centre of the 3x3 neighbourhood
    assert torch.equal(convex_upsample(low, weights, 4), torch.full((1, 1, 12, 16), 5.))


def test_plane_sweep_handles_rotation_noncentred_K_and_nonidentity_reference():
    height, width = 16, 20
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    source = torch.stack([xx, yy]).float().unsqueeze(0)
    Kref = torch.tensor([[[22., 0., 6.], [0., 20., 5.], [0., 0., 1.]]])
    Ksrc = torch.tensor([[[21., 0., 9.], [0., 23., 7.], [0., 0., 1.]]])
    reference = torch.eye(4).unsqueeze(0)
    angle = torch.tensor(.07)
    reference[0, :3, :3] = torch.tensor([
        [torch.cos(angle), 0., torch.sin(angle)], [0., 1., 0.],
        [-torch.sin(angle), 0., torch.cos(angle)]])
    reference[0, :3, 3] = torch.tensor([.3, -.2, .1])
    source_pose = torch.eye(4).unsqueeze(0)
    source_pose[0, :3, 3] = torch.tensor([.6, .1, .2])
    samples = torch.full((1, 1, height, width), 5.)
    warped, valid = plane_sweep_warp(source, samples, Kref, Ksrc, reference, source_pose)
    # Independent camera->world->source derivation, not the production relative matrix.
    pix = torch.stack([xx, yy, torch.ones_like(xx)]).float().reshape(3, -1)
    cam_ref = torch.linalg.solve(Kref[0], pix) * 5.
    world = reference[0, :3, :3].T @ (cam_ref - reference[0, :3, 3:4])
    cam_src = source_pose[0, :3, :3] @ world + source_pose[0, :3, 3:4]
    projected = Ksrc[0] @ cam_src
    expected = (projected[:2] / projected[2:]).reshape(1, 2, 1, height, width)
    selected = valid.bool().expand_as(warped)
    assert selected.any()
    assert torch.allclose(warped[selected], expected[selected], atol=2e-5)


def test_legacy_ssi_does_not_anchor_the_physical_lookup_index():
    q = torch.linspace(.08, .85, 12 * 16).reshape(1, 1, 12, 16).repeat(1, 3, 1, 1)
    depth = 1. / (.0125 + q * (1. / 3. - .0125))
    wrong_q = .5 * q + .25
    criterion = VideoDepthLoss(pose_flag=False)
    args = (depth, torch.ones_like(depth), None, None, None, None)
    assert criterion(q, *args)["total_loss"] < 2e-6
    assert criterion(wrong_q, *args)["total_loss"] < 2e-6
    assert ((wrong_q - q).abs().mean() * 31) > 3.


def test_legacy_focal_rows_have_no_supervised_gradient():
    torch.manual_seed(8)
    # Attention is irrelevant to the output-row/iterative-detach contract.
    camera = CameraHead(dim_in=32, trunk_depth=0, num_heads=4)
    predictions = camera(torch.randn(1, 3, 1, 32))
    extrinsic = torch.eye(4).repeat(1, 3, 1, 1)
    extrinsic[0, :, 0, 3] = torch.tensor([0., .5, 1.])
    intrinsic = torch.tensor([[[30., 0., 8.], [0., 30., 6.], [0., 0., 1.]]])
    images = torch.ones(1, 3, 12, 16)
    compute_camera_loss(weight_focal=0.)(
        predictions, intrinsic, extrinsic, images, images)["loss_camera"].backward()
    grad = camera.pose_branch.fc2.weight.grad
    assert grad[:7].norm() > 0
    assert torch.equal(grad[7:], torch.zeros_like(grad[7:]))


def test_focal_encoding_roundtrips_but_cropped_principal_point_does_not():
    extrinsic = torch.eye(4).repeat(1, 3, 1, 1)
    intrinsic = torch.tensor([[[24., 0., 2.], [0., 24., 6.], [0., 0., 1.]]])
    encoded = compute_camera_loss().extri_intri_to_pose_encoding(
        extrinsic, intrinsic[:, None].expand(-1, 3, -1, -1), (12, 16), "absT_quaR_FoV")
    _, recovered = pose_encoding_to_extri_intri(encoded, 12, 16)
    assert torch.allclose(recovered[..., 0, 0], torch.full((1, 3), 24.))
    assert torch.equal(recovered[..., 0, 2], torch.full((1, 3), 8.))
    assert not torch.allclose(recovered[:, 0], intrinsic)


def test_legacy_lookup_wiring_and_trace_preserve_decoder_output():
    torch.manual_seed(0)
    frames, base, dims = 3, 64, [8, 16, 32, 64]
    head = costvol.DPTHeadCostVolumeConvNeXt(
        dims, patch_size=4, num_sample=8, num_groups=2,
        match_dim=8, hidden_dim=16, iters=2).eval()
    features = [torch.randn(frames, dim, base // (4 * 2 ** i), base // (4 * 2 ** i))
                for i, dim in enumerate(dims)]
    images = torch.randn(1, frames, 3, base, base)
    intrinsic = torch.tensor([[64., 0., 32.], [0., 64., 32.], [0., 0., 1.]]).repeat(1, frames, 1, 1)
    extrinsic = torch.eye(4).repeat(1, frames, 1, 1)
    extrinsic[0, :, 0, 3] = torch.arange(frames) * .3
    captured = {}
    original = costvol.VolumeIndexer

    def observe(geometry, second, **kwargs):
        captured["softmax_substitution"] = torch.allclose(second, geometry.softmax(dim=2))
        return original(geometry, second, **kwargs)

    before = {name: tensor.clone() for name, tensor in head.state_dict().items()}
    with patch.object(costvol, "VolumeIndexer", observe):
        trace = trace_decoder(head, features, images, extrinsic, intrinsic)
    # This is an audit assertion of the legacy wiring, not the desired new method.
    assert captured["softmax_substitution"]
    assert len(trace["lookups"]) == len(trace["updates"]) == 2
    assert trace["raw_volume"]["finite_fraction"] == 1.
    assert trace["final_raw_output"]["finite_fraction"] == 1.
    assert all(torch.equal(before[name], value) for name, value in head.state_dict().items())
    assert not head.volume_stem._forward_pre_hooks
    assert not head.classifier._forward_hooks
    output = head(features, base // 4, base // 4, frames,
                  images=images, extrinsics=extrinsic, intrinsics=intrinsic)
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in head.matcher.parameters())
    assert sum(p.grad.square().sum() for p in head.matcher.parameters()) > 0


def test_oracle_cli_runs_all_camera_modes_and_writes_trace(tmp_path, monkeypatch):
    """Exercise the CLI data/shape/JSON path without weights, data or a GPU."""
    from types import SimpleNamespace

    frames, size, dims = 3, 64, [8, 16, 32, 64]
    intrinsic = torch.tensor([[64., 0., 26.], [0., 64., 32.], [0., 0., 1.]])
    extrinsic = torch.eye(4).repeat(frames, 1, 1)
    extrinsic[:, 0, 3] = torch.arange(frames) * .3

    class Dataset:
        transform = {}

        def __len__(self):
            return 4

        def _getitem_inner(self, index):
            return {"image": torch.rand(frames, 3, size, size),
                    "depth": torch.full((frames, 1, size, size), 8.),
                    "mask": torch.ones(frames, 1, size, size),
                    "IntM": intrinsic, "poses": extrinsic,
                    "path": [f"scene/{index + t}.jpg" for t in range(frames)]}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = costvol.DPTHeadCostVolumeConvNeXt(
                dims, patch_size=4, num_sample=8, num_groups=2,
                match_dim=8, hidden_dim=16, iters=1)

        def forward(self, images):
            features = [torch.randn(frames, dim, size // (4 * 2 ** i), size // (4 * 2 ** i))
                        for i, dim in enumerate(dims)]
            return self.head(features, size // 4, size // 4, frames, images=images,
                             intrinsics=intrinsic[None, None].expand(1, frames, -1, -1),
                             extrinsics=extrinsic[None])

    model = Model()
    output = tmp_path / "oracle.json"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(oracle, "load_experiment_config", lambda _: SimpleNamespace(
        model=SimpleNamespace(use_gem=True), dataset=SimpleNamespace(val={})))
    monkeypatch.setattr(oracle, "build_gemdepth_from_config", lambda *args, **kwargs: model)
    monkeypatch.setattr(oracle, "_load_state", lambda _: model.state_dict())
    monkeypatch.setattr(oracle, "DepthVideoDataset", lambda **kwargs: Dataset())
    monkeypatch.setattr(sys, "argv", ["oracle", "--config", "unused", "--ckpt", "unused",
                                   "--vkitti-root", "unused", "--max-batches", "2",
                                   "--trace-head", "--output", str(output)])
    oracle.main()
    report = json.loads(output.read_text())
    assert report["diagnostic_version"] == 2
    assert report["batches"] == 2
    assert len(report["modes"]) == 8
    assert not report["failure_examples"]
    assert report["clips"][0]["head_trace"]["raw_volume"]["finite_fraction"] == 1.
    assert report["paired_gt_support_modes"]["gt_normalized"]["mode"]["eligible_pixels"] > 0