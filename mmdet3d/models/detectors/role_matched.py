"""Role-Matched Capacity detector.

Extends BEVDilation with:
  - RoleMatchedBEVBackbone (PConv + LSK + tile gate + image guidance)
  - FGN auxiliary regression loss
  - Feature distillation from a frozen teacher
"""
import torch
from copy import deepcopy
from mmdet.models import DETECTORS
from .bevdilation import BEVDilation, clip_sigmoid
from mmdet3d.models.losses.fgn_loss import FGNLoss
from mmdet3d.models.losses.distillation_loss import DistillationLoss


@DETECTORS.register_module()
class RoleMatchedDetector(BEVDilation):
    """Compressed BEVDilation with role-matched capacity allocation.

    Additional kwargs over BEVDilation:
        fgn_cfg (dict): Config for FGNLoss. Default: dict(C=3.0, eta=0.5,
            loss_weight=0.5).
        distill_cfg (dict | None): Config for DistillationLoss. None to
            disable distillation.
        teacher_ckpt (str | None): Path to the frozen teacher checkpoint.
    """

    def __init__(self, fgn_cfg=None, distill_cfg=None, teacher_ckpt=None,
                 **kwargs):
        super().__init__(**kwargs)

        if fgn_cfg is None:
            fgn_cfg = dict(C=3.0, eta=0.5, loss_weight=0.5)
        self.fgn_loss = FGNLoss(**fgn_cfg)

        self.distill_loss = None
        self.teacher = None
        if distill_cfg is not None:
            self.distill_loss = DistillationLoss(**distill_cfg)

        if teacher_ckpt is not None:
            self._load_teacher(teacher_ckpt, kwargs)

    def _load_teacher(self, ckpt_path, kwargs):
        """Build and freeze a teacher copy of BEVDilation."""
        teacher_kwargs = deepcopy(kwargs)
        teacher_kwargs.pop('fgn_cfg', None)
        teacher_kwargs.pop('distill_cfg', None)
        teacher_kwargs.pop('teacher_ckpt', None)
        try:
            self.teacher = BEVDilation(**teacher_kwargs)
            ckpt = torch.load(ckpt_path, map_location='cpu')
            state_dict = ckpt.get('state_dict', ckpt)
            self.teacher.load_state_dict(state_dict, strict=False)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False
        except Exception:
            self.teacher = None

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        img_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        img_feats_bev = self.img_view_transformer(
            img_feats + img_inputs[1:7],
            depth_from_lidar=kwargs['gt_depth'])

        bev_mask_for_gate = None
        pts_feats, pred_bev_mask = self.extract_pts_feat(
            points, img_feats, img_metas, img_feats_bev[0].clone())

        if hasattr(self.pts_backbone, 'tile_gate'):
            bev_mask_for_gate = pred_bev_mask

        losses = dict()
        losses_pts = self.forward_pts_train(
            [img_feats, pts_feats, img_feats_bev],
            gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore)
        losses.update(losses_pts)

        losses_img_auxiliary = self.forward_img_auxiliary_train(
            img_feats, img_metas, gt_bboxes_3d, gt_labels_3d,
            gt_bboxes_ignore, **kwargs)
        losses.update(losses_img_auxiliary)

        gt_bev_mask = self.pts_middle_encoder.SVDB.obtain_bev_mask_gt(
            gt_bboxes_3d, gt_bboxes_ignore, img_feats[0].device,
            pred_bev_mask)
        loss_bev_mask = self.pts_bbox_head.loss_bevmask(
            clip_sigmoid(pred_bev_mask), gt_bev_mask.unsqueeze(1),
            avg_factor=max(gt_bev_mask.sum().item(), 1))
        losses.update(dict(loss_bev_mask=loss_bev_mask))

        if self.distill_loss is not None and self.teacher is not None:
            with torch.no_grad():
                t_img_feats = self.teacher.extract_feat(
                    points, img=img_inputs, img_metas=img_metas)
                t_img_feats_bev = self.teacher.img_view_transformer(
                    t_img_feats + img_inputs[1:7],
                    depth_from_lidar=kwargs['gt_depth'])
                t_pts_feats, t_mask = self.teacher.extract_pts_feat(
                    points, t_img_feats, img_metas,
                    t_img_feats_bev[0].clone())

            student_dict = {
                'img_bev': img_feats_bev[0],
                'mask': pred_bev_mask,
                'bev_feat': pts_feats[0] if isinstance(pts_feats, (list, tuple))
                            else pts_feats,
            }
            teacher_dict = {
                'img_bev': t_img_feats_bev[0],
                'mask': t_mask,
                'bev_feat': t_pts_feats[0] if isinstance(t_pts_feats, (list, tuple))
                            else t_pts_feats,
            }
            loss_dis = self.distill_loss(student_dict, teacher_dict)
            losses.update(dict(loss_distill=loss_dis))

        return losses
