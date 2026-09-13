"""Verify the actual computation, not merely output shapes or a finite loss."""

import copy
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "test")]

from loss.objective_calibrated_index import CalibratedIndexL1Objective
from loss.objective_calibrated_sequence import CalibratedSequenceIndexL1Objective, igev_relative_weights
from loss.objective_registry import build_objective
from model.decoder_registry import build_decoder, get_decoder_class
import model.dpt_calibrated_gru_sequence_convnext as sequence
from model.util.cost_volume import VolumeIndexer
from test_stereogru_corrected_head import fixture


def sequence_fixture(iters=8, baseline=False):
    old, features, cameras = fixture(iters=iters)
    name = "DPTHeadCalibratedVolumeSequenceConvNeXt" if baseline else "DPTHeadCalibratedGRUSequenceConvNeXt"
    head = build_decoder(get_decoder_class(name),
                         {"num_sample": 8, "num_groups": 2, "match_dim": 8,
                          "hidden_dim": 16, "iters": iters, "depth_min": 3., "depth_max": 80.},
                         in_channels_list=[8, 16, 32, 64], patch_size=4)
    state = old.state_dict()
    if baseline:
        state = {k: v for k, v in state.items() if not k.startswith(old.recurrent_state_prefixes)}
    head.load_state_dict(state, strict=True)
    return head.eval(), old, features, cameras


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("iters", [0, 8])
def test_final_output_state_and_bn_updates_match_old_g1(training, iters):
    head, old, features, cameras = sequence_fixture(iters)
    head.train(training)
    old.train(training)
    expected = old(features, 16, 16, 3, **cameras)
    actual = head(features, 16, 16, 3, **cameras)
    if training:
        assert isinstance(actual, tuple) and len(actual) == iters + 1
        actual = actual[-1]
    assert torch.equal(actual, expected)
    assert set(head.state_dict()) == set(old.state_dict())
    assert all(torch.equal(v, old.state_dict()[k]) for k, v in head.state_dict().items())


def test_initial_readout_uses_initial_hidden_and_matches_volume_baseline():
    head, old, features, cameras = sequence_fixture()
    baseline, _, _, _ = sequence_fixture(0, baseline=True)
    initial = {k: v for k, v in old.state_dict().items() if not k.startswith(old.recurrent_state_prefixes)}
    baseline.load_c1_initial(initial)
    with torch.no_grad():
        outputs = head(features, 16, 16, 3, return_sequence=True, **cameras)
        expected = baseline(features, 16, 16, 3, **cameras)
        final = head(features, 16, 16, 3, **cameras)
    assert len(outputs) == 9 and torch.equal(outputs[0], expected)
    assert torch.equal(outputs[-1], final)
    assert all(q.shape == (3, 1, 64, 64) for q in outputs)
    assert all(torch.isfinite(q).all() and (q >= 0).all() and (q <= 1).all() for q in outputs)


@pytest.mark.parametrize("objective_name,expected_direct", [
    ("calibrated_igev_sequence_index_l1", [True] * 9),
    ("calibrated_final_sequence_index_l1", [False] * 8 + [True]),
])
def test_direct_gradients_reach_initial_index_and_each_update_before_detach(objective_name, expected_direct):
    head, _, features, cameras = sequence_fixture()
    head.train()
    indices, deltas, coordinates = [], [], []
    original = head._checked_readout

    def readout(index, hidden, *shape):
        index.retain_grad()
        indices.append(index)
        return original(index, hidden, *shape)

    def delta_hook(_module, _args, output):
        output.retain_grad()
        deltas.append(output)

    handles = [head.index_head.register_forward_hook(delta_hook),
               head.encoder.register_forward_pre_hook(lambda _m, args: coordinates.append(args[0].requires_grad))]
    with patch.object(head, "_checked_readout", readout):
        outputs = head(features, 16, 16, 3, **cameras)
    for q in outputs:
        q.retain_grad()
    target = torch.full_like(outputs[0], 8.)
    loss = build_objective(objective_name, {"depth_min": 3., "depth_max": 80., "iterations": 8})
    report = loss(outputs, target, torch.ones_like(target))
    report["total_loss"].backward()
    assert coordinates == [False] * 8
    assert [q.grad is not None and float(q.grad.abs().sum()) > 0 for q in outputs] == expected_direct
    assert [q.grad is not None and float(q.grad.abs().sum()) > 0 for q in indices] == expected_direct
    assert [q.grad is not None and float(q.grad.abs().sum()) > 0 for q in deltas] == expected_direct[1:]
    assert all(torch.isfinite(q.grad).all() for q in outputs if q.grad is not None)
    for name in head.required_gradient_modules:
        gradients = [p.grad for p in getattr(head, name).parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients), name
        assert sum(float(g.square().sum()) for g in gradients) > 0, name
    for handle in handles:
        handle.remove()


def test_final_only_sequence_gradients_and_optimizer_update_equal_legacy():
    head, old, features, cameras = sequence_fixture()
    head.train()
    old.train()
    target = torch.full((3, 1, 64, 64), 8.)
    mask = torch.ones_like(target)
    objective = build_objective("calibrated_final_sequence_index_l1",
                                {"depth_min": 3., "depth_max": 80., "iterations": 8})
    objective(head(features, 16, 16, 3, **cameras), target, mask)["total_loss"].backward()
    CalibratedIndexL1Objective(3., 80.)(old(features, 16, 16, 3, **cameras), target, mask)["total_loss"].backward()
    for (name, p), (old_name, q) in zip(head.named_parameters(), old.named_parameters()):
        assert name == old_name
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            assert torch.equal(p.grad, q.grad), name
    for model in (head, old):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.).step()
    assert all(torch.equal(v, old.state_dict()[k]) for k, v in head.state_dict().items())


def test_raw_and_gev_lookup_branches_not_detached_or_softmax_substituted():
    head, _, features, cameras = sequence_fixture()
    observed = {}
    handles = [head.volume_stem.register_forward_pre_hook(lambda _m, args: observed.update(raw=args[0])),
               head.classifier.register_forward_hook(lambda _m, _args, out: observed.update(logits=out))]

    def lookup(geometry, raw, **options):
        assert torch.equal(geometry, observed["logits"])
        assert torch.equal(raw, observed["raw"].mean(1, keepdim=True))
        assert not torch.allclose(raw, geometry.softmax(2))
        geometry.retain_grad()
        raw.retain_grad()
        observed.update(geometry=geometry, direct_raw=raw)
        return VolumeIndexer(geometry, raw, **options)

    with patch.object(sequence, "VolumeIndexer", lookup):
        head(features, 16, 16, 3, **cameras).square().mean().backward()
    for name in ("geometry", "direct_raw"):
        assert observed[name].grad is not None and observed[name].grad.abs().sum() > 0
        assert torch.isfinite(observed[name].grad).all()
    for handle in handles:
        handle.remove()


def test_final_loss_reaches_every_recurrent_hidden_state_without_coordinate_gradient():
    head, _, features, cameras = sequence_fixture()
    head.train()
    hidden = {name: [] for name in ("gru_coarse", "gru_mid", "gru_fine")}

    def retain(name):
        def observe(_module, _args, output):
            output.retain_grad()
            hidden[name].append(output)
        return observe

    handles = [getattr(head, name).register_forward_hook(retain(name)) for name in hidden]
    try:
        predictions = head(features, 16, 16, 3, **cameras)
        target = torch.full_like(predictions[-1], 8.)
        objective = build_objective("calibrated_final_sequence_index_l1",
                                    {"depth_min": 3., "depth_max": 80., "iterations": 8})
        objective(predictions, target, torch.ones_like(target))["total_loss"].backward()
        for name, states in hidden.items():
            assert len(states) == 8, name
            for state in states:
                assert state.grad is not None and torch.isfinite(state.grad).all(), name
                assert state.grad.abs().sum() > 0, name
    finally:
        for handle in handles:
            handle.remove()


def test_sequence_weight_formula_and_loss_gradient_match_independent_reference():
    raw = (1.,) + tuple(.9 ** (15 * (7 - i) / 7) for i in range(8))
    assert igev_relative_weights(8) == raw
    objective = CalibratedSequenceIndexL1Objective(3., 80., raw)
    assert sum(objective.weights) == pytest.approx(1.)
    predictions = tuple(torch.full((1, 1, 2, 2), .1 + i * .03, requires_grad=True) for i in range(9))
    target = torch.tensor([[[[4., 8.], [float("nan"), 81.]]]])
    mask = torch.tensor([[[[1., 1.], [0., 1.]]]])
    truth = (target[0, 0, 0].reciprocal() - 1. / 80) / (1. / 3 - 1. / 80)
    expected_parts = torch.stack([(p[0, 0, 0] - truth).abs().mean() for p in predictions])
    expected = sum(v * w for v, w in zip(expected_parts, raw)) / sum(raw)
    result = objective(predictions, target, mask)
    assert torch.allclose(result["total_loss"], expected)
    assert torch.equal(result["per_prediction_index_l1"], expected_parts)
    assert result["index_l1"] is not result["total_loss"]
    assert int(result["valid_pixels"]) == 2
    result["total_loss"].backward()
    for p, w in zip(predictions, objective.weights):
        assert torch.allclose(p.grad[0, 0, 0], (p.detach()[0, 0, 0] - truth).sign() * w / 2)
        assert torch.equal(p.grad[0, 0, 1], torch.zeros(2))


def test_baseline_sequence_loss_and_gradients_exactly_match_single_output():
    q = torch.tensor([[[[.2, .3]]]], requires_grad=True)
    target = torch.full_like(q, 8.)
    mask = torch.ones_like(q)
    new = build_objective("calibrated_igev_sequence_index_l1", {"depth_min": 3., "depth_max": 80., "iterations": 0})
    old = CalibratedIndexL1Objective(3., 80.)
    a, b = new((q,), target, mask)["total_loss"], old(q, target, mask)["total_loss"]
    assert torch.equal(a, b)
    assert torch.equal(torch.autograd.grad(a, q)[0], torch.autograd.grad(b, q)[0])


@pytest.mark.parametrize("weights", [[], [0., 0.], [-1., 2.], [float("nan")], [float("inf")]])
def test_invalid_weights_refused(weights):
    with pytest.raises(ValueError):
        CalibratedSequenceIndexL1Objective(3., 80., weights)


@pytest.mark.parametrize("defect", ["tensor", "short", "long", "nonfinite", "shape", "empty_mask"])
def test_malformed_sequence_never_silently_drops_predictions(defect):
    objective = CalibratedSequenceIndexL1Objective(3., 80., [1., 1.])
    q = torch.full((1, 1, 2, 2), .2)
    values, target, mask = [q, q.clone()], torch.full_like(q, 8.), torch.ones_like(q)
    if defect == "tensor":
        values = q
    elif defect == "short":
        values.pop()
    elif defect == "long":
        values.append(q)
    elif defect == "nonfinite":
        values[0][0, 0, 0, 0] = float("nan")
    elif defect == "shape":
        values[0] = q[..., :1]
    else:
        mask.zero_()
    with pytest.raises((ValueError, FloatingPointError)):
        objective(values, target, mask)


def test_bad_camera_rejected_without_touching_geometry_builder():
    head, _, features, cameras = sequence_fixture()
    cameras = copy.deepcopy(cameras)
    cameras["intrinsics"][..., 1, 1] = float("inf")
    with patch.object(head, "_build_volume") as build:
        with pytest.raises(ValueError, match="Nonfinite"):
            head(features, 16, 16, 3, **cameras)
        build.assert_not_called()


def test_production_dimensions_legacy_gradient_parity_and_live_sequence_verification():
    """Real head settings + 4x256 synthetic cached features, NOT real-data accuracy."""
    from omegaconf import OmegaConf
    import stereogru_sequence_experiment as experiment

    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        cfg = OmegaConf.to_container(OmegaConf.load(
            ROOT / "config/stereogru/matched_c0_c1_30train.yaml"), resolve=True)
        backbone = {"embed_dims": [96, 192, 384, 768], "patch_size": 4}
        experiment.controls.seed_everything(0)
        old = build_decoder(get_decoder_class("DPTHeadCalibratedGRUConvNeXt"),
                            {**cfg["decoder"]["kwargs"], "iters": 8},
                            in_channels_list=backbone["embed_dims"], patch_size=4)
        head = build_decoder(get_decoder_class("DPTHeadCalibratedGRUSequenceConvNeXt"),
                             {**cfg["decoder"]["kwargs"], "iters": 8},
                             in_channels_list=backbone["embed_dims"], patch_size=4)
        head.load_state_dict(old.state_dict(), strict=True)
        features = [torch.randn(4, dim, 256 // stride, 256 // stride)
                    for dim, stride in zip(backbone["embed_dims"], [4, 8, 16, 32])]
        K = torch.tensor([[220., 0., 127.5], [0., 220., 127.5], [0., 0., 1.]]).repeat(1, 4, 1, 1)
        E = torch.eye(4).repeat(1, 4, 1, 1)
        E[0, :, 0, 3] = torch.arange(4) * .3
        clip = {"images": torch.rand(1, 4, 3, 256, 256), "features": features,
                "intrinsics": K, "extrinsics": E, "depth": torch.full((1, 4, 1, 256, 256), 8.),
                "mask": torch.ones(1, 4, 1, 256, 256)}
        objective = build_objective("calibrated_igev_sequence_index_l1",
                                    {"depth_min": 3., "depth_max": 80., "iterations": 8})
        report = experiment.verify_model(head, {"config": cfg, "contract": {"backbone": backbone}}, clip, objective)
        assert all(v > 0 for v in report["output_gradient_norms"])
        assert len(report["output_gradient_norms"]) == 9 and len(report["update_gradient_norms"]) == 8
        assert all(v == 0. for v in report["parity"].values())
        head.train()
        old.train()
        q = experiment.predict_sequence(head, clip)
        expected = experiment.controls.predict(old, clip)
        assert torch.equal(q[-1], expected)
        final_only = build_objective("calibrated_final_sequence_index_l1",
                                     {"depth_min": 3., "depth_max": 80., "iterations": 8})
        final_only(q, clip["depth"], clip["mask"])["total_loss"].backward()
        CalibratedIndexL1Objective(3., 80.)(expected, clip["depth"], clip["mask"])["total_loss"].backward()
        for (name, p), (old_name, reference) in zip(head.named_parameters(), old.named_parameters()):
            assert name == old_name
            assert (p.grad is None) == (reference.grad is None), name
            if p.grad is not None:
                assert torch.equal(p.grad, reference.grad), name
        for model in (head, old):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.).step()
        assert all(torch.equal(value, old.state_dict()[name]) for name, value in head.state_dict().items())
    finally:
        torch.set_num_threads(previous)