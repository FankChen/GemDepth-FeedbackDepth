"""Smoke test for geometric temporal (cycle) consistency loss."""
import torch
from model.util.temporal import geometric_temporal_consistency, align_pred_metric

torch.manual_seed(0)
B, T, H, W = 2, 4, 24, 32

# intrinsics at depth resolution
K = torch.tensor([[30., 0., W / 2], [0., 30., H / 2], [0., 0., 1.]]).view(1, 1, 3, 3).expand(B, T, 3, 3).contiguous()

# camera trajectory: small translation per frame (world->cam extrinsics)
ext = torch.eye(4).view(1, 1, 4, 4).expand(B, T, 4, 4).contiguous()
for t in range(T):
    ext[:, t, 0, 3] = 0.05 * t  # slide along x

# ---- 1) geometry self-consistency: identity poses + identical depth => ~0 ----
ext_id = torch.eye(4).view(1, 1, 4, 4).expand(B, T, 4, 4).contiguous()
depth_const = torch.full((B, T, 1, H, W), 5.0)
l0 = geometric_temporal_consistency(depth_const, K, ext_id, offsets=(1,))
print(f"[1] identity-pose consistent loss = {l0.item():.6e} (expect ~0)")
assert l0.item() < 1e-4, "consistent scene should give ~0 loss"

# ---- 2) perturb neighbour frame depth => loss > 0 ----
depth_pert = depth_const.clone()
depth_pert[:, 1:] += 1.5
l1 = geometric_temporal_consistency(depth_pert, K, ext_id, offsets=(1,))
print(f"[2] perturbed loss = {l1.item():.6e} (expect > 0)")
assert l1.item() > l0.item(), "perturbation should raise loss"

# ---- 3) gradient flows through prediction ----
pred_disp = torch.rand(B, T, 1, H, W, requires_grad=True)
gt_depth = 3.0 + 4.0 * torch.rand(B, T, 1, H, W)
mask = torch.ones(B, T, 1, H, W)
depth_m = align_pred_metric(pred_disp, gt_depth, mask)
print(f"[3] aligned metric depth range = [{depth_m.min().item():.3f}, {depth_m.max().item():.3f}]")
loss = geometric_temporal_consistency(depth_m, K, ext, offsets=(1,))
loss.backward()
print(f"[3] cycle loss = {loss.item():.6e}, grad_norm = {pred_disp.grad.norm().item():.6e}")
assert pred_disp.grad is not None and pred_disp.grad.norm().item() > 0, "grad must flow to prediction"

# ---- 4) multi-offset ----
l2 = geometric_temporal_consistency(depth_m.detach(), K, ext, offsets=(1, 2))
print(f"[4] multi-offset (1,2) loss = {l2.item():.6e}")

print("ALL OK")
