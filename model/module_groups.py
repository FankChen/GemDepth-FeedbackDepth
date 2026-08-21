"""Named groups of model parameters, shared by the optimizer and freeze policies.

These prefixes describe *what GemDepth adds on top of a pretrained depth model*
(DAv2 / VDA). They are consumed in two places that must never disagree:

  * ``train.py`` puts them in the ``optimizer.dec_lr`` group, because they are
    always randomly initialised and need the paper's 1e-4 rather than the 1e-6
    used for pretrained weights.
  * ``model/freeze_policies.py`` freezes GEM alone for the paper's stage 2.

Keeping them here means neither module has to import the other.
"""

# ASTT - alternating spatio-temporal transformer.
ASTT_PREFIXES = ('spatial_blocks', 'time_blocks', 'dec_norm')

# GEM - geometry embedding module (camera pose + geometric features).
GEM_PREFIXES = (
    'global_blocks', 'frame_blocks', 'camera_token', 'register_token',
    'camera_head', 'cam_rot_encoder', 'cam_trans_encoder',
    'cam_trans_scale_encoder',
)

# Everything GemDepth adds; always random-init, hence the "new module" rate.
NEW_MODULE_PREFIXES = ASTT_PREFIXES + GEM_PREFIXES
