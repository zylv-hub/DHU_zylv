"""Role-Matched detector — matched (ImageNet) setting.

Same as role_matched.py but:
  - No nuImages domain pretraining (ImageNet init only)
  - lambda_e = 0 (no image backbone distillation)
  - load_from: trained ImageNet baseline checkpoint
"""
_base_ = ['./role_matched.py']

model = dict(
    distill_cfg=dict(
        student_bev_dim=256,
        teacher_bev_dim=256,
        student_img_dims=[128, 256, 512],
        teacher_img_dims=[512, 1024, 2048],
        lambda_g=1.0,
        lambda_m=1.0,
        lambda_b=0.5,
        lambda_e=0.0,  # no backbone distillation in matched setting
    ),
    teacher_ckpt=None,  # path to trained ImageNet baseline
)

load_from = None  # path to trained ImageNet baseline for warm-start
