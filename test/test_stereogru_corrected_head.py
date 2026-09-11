"""Correct raw/GEV wiring, bounded physical indices and recurrent gradient contracts."""

from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "test")]

import model.dpt_calibrated_gru_convnext as corrected
from model.decoder_registry import build_decoder, get_decoder_class
from model.util.cost_volume import VolumeIndexer
from test_stereogru_calibrated_baseline import camera_fixture, small_head


def fixture(iters=2):
    torch.manual_seed(0)
    head = build_decoder(get_decoder_class("DPTHeadCalibratedGRUConvNeXt"),
                         {"num_sample": 8, "num_groups": 2, "match_dim": 8,
                          "hidden_dim": 16, "iters": iters, "depth_min": 3., "depth_max": 80.},
                         in_channels_list=[8, 16, 32, 64], patch_size=4).eval()
    images, K, E = camera_fixture()
    features = [torch.randn(3, dim, 64 // (4 * 2 ** i), 64 // (4 * 2 ** i))
                for i, dim in enumerate([8, 16, 32, 64])]
    return head, features, {"images": images, "intrinsics": K, "extrinsics": E, "geometry_gauge": "metric"}


def test_zero_iterations_replay_c1_exactly_with_shared_initial_state():
    head, features, kwargs = fixture(iters=0)
    baseline = small_head().eval()
    initial = {name: value.clone() for name, value in baseline.state_dict().items()}
    head.load_c1_initial(initial)
    with torch.no_grad():
        expected = baseline(features, 16, 16, 3, **kwargs)
        actual = head(features, 16, 16, 3, **kwargs)
    assert torch.equal(expected, actual)
    for name, value in initial.items():
        assert torch.equal(head.state_dict()[name], value)
    bad = dict(initial)
    bad.pop("classifier.weight")
    with pytest.raises(ValueError, match="common initial keys differ"):
        head.load_c1_initial(bad)


def test_lookup_reads_raw_correlation_not_softmax_and_both_branches_get_gradient():
    head, features, kwargs = fixture()
    observed = {}
    raw_handle = head.volume_stem.register_forward_pre_hook(lambda _m, args: observed.update(raw=args[0]))
    logits_handle = head.classifier.register_forward_hook(lambda _m, _args, out: observed.update(logits=out))

    def observe(geometry, raw, **options):
        assert torch.equal(geometry, observed["logits"])
        assert torch.equal(raw, observed["raw"].mean(dim=1, keepdim=True))
        assert not torch.allclose(raw, geometry.softmax(dim=2))
        geometry.retain_grad()
        raw.retain_grad()
        observed.update(geometry_lookup=geometry, raw_lookup=raw)
        return VolumeIndexer(geometry, raw, **options)

    coordinates = []
    encoder_handle = head.encoder.register_forward_pre_hook(lambda _m, args: coordinates.append(args[0].requires_grad))
    with patch.object(corrected, "VolumeIndexer", observe):
        output = head(features, 16, 16, 3, **kwargs)
        output.square().mean().backward()
    for handle in (raw_handle, logits_handle, encoder_handle):
        handle.remove()
    assert coordinates == [False, False]
    for key in ("raw_lookup", "geometry_lookup"):
        assert observed[key].grad is not None and torch.isfinite(observed[key].grad).all()
        assert observed[key].grad.abs().sum() > 0
    for module in (head.matcher, head.classifier, head.encoder, head.gru_coarse,
                   head.gru_mid, head.gru_fine, head.index_head):
        gradients = [p.grad for p in module.parameters()]
        assert all(grad is not None and torch.isfinite(grad).all() for grad in gradients)
        assert sum(grad.square().sum() for grad in gradients) > 0


def test_bound_update_identity_symmetry_boundaries_and_derivatives():
    index = torch.tensor([0., 1., 3.5, 6., 7.], requires_grad=True)
    zeros = torch.zeros_like(index)
    assert torch.equal(corrected.bounded_index_update(index, zeros, 7), index)
    delta = torch.tensor([1., -.2, .3, .5, -1.], requires_grad=True)
    actual = corrected.bounded_index_update(index, delta, 7)
    mirrored = corrected.bounded_index_update(7 - index, -delta, 7)
    assert torch.allclose(actual, 7 - mirrored, atol=1e-6)
    assert (actual >= 0).all() and (actual <= 7).all()
    actual.sum().backward()
    assert torch.isfinite(delta.grad).all() and (delta.grad > 0).all()
    extreme = corrected.bounded_index_update(index.detach(), torch.tensor([-1e6, -1e6, 1e6, 1e6, 1e6]), 7)
    assert (extreme >= 0).all() and (extreme <= 7).all()
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        corrected.bounded_index_update(index, delta * float("nan"), 7)
    with pytest.raises(ValueError, match="outside the physical"):
        corrected.bounded_index_update(index + 1, delta, 7)


def test_eight_iteration_order_and_train_eval_output_contract():
    head, features, kwargs = fixture(iters=8)
    order = []
    handles = [module.register_forward_hook(lambda _m, _a, _o, name=name: order.append(name))
               for name, module in (("coarse", head.gru_coarse), ("mid", head.gru_mid), ("fine", head.gru_fine))]
    for training in (False, True):
        head.train(training)
        out = head(features, 16, 16, 3, **kwargs)
        assert isinstance(out, torch.Tensor) and out.shape == (3, 1, 64, 64)
        assert torch.isfinite(out).all() and (out >= 0).all() and (out <= 1).all()
        assert len(head.last_iteration_diagnostics) == 8
        assert all(0 <= row["lookup_index_min"] <= row["lookup_index_max"] <= 7 for row in head.last_iteration_diagnostics)
    assert order == ["coarse", "mid", "fine"] * 16
    for handle in handles:
        handle.remove()


def test_gru_gate_equations_match_independent_convolution_reference():
    head, _, _ = fixture()
    cell = head.gru_fine
    hidden = torch.randn(2, 16, 5, 6)
    inputs = [torch.randn(2, 16, 5, 6), torch.randn(2, 16, 5, 6)]
    joined = torch.cat([hidden, *inputs], dim=1)
    conv = lambda tensor, layer: F.conv2d(tensor, layer.weight, layer.bias, padding=1)
    z = torch.sigmoid(conv(joined, cell.convz))
    r = torch.sigmoid(conv(joined, cell.convr))
    candidate = torch.tanh(conv(torch.cat([r * hidden, *inputs], dim=1), cell.convq))
    expected = (1 - z) * hidden + z * candidate
    assert torch.equal(cell(hidden, *inputs), expected)


def test_bad_camera_refused_before_sweep_and_raw_features_affect_output():
    head, features, kwargs = fixture()
    with torch.no_grad():
        first = head(features, 16, 16, 3, **kwargs)
        changed = head([feature.flip(-1) for feature in features], 16, 16, 3, **kwargs)
    assert not torch.equal(first, changed)
    kwargs["intrinsics"][..., 1, 1] = float("inf")
    with patch.object(head, "_build_volume") as build:
        with pytest.raises(ValueError, match="Nonfinite"):
            head(features, 16, 16, 3, **kwargs)
        build.assert_not_called()