"""Focaler-GIoU-NWD (FGN) loss for BEV box regression.

Combines Focaler reweighting of GIoU (Zhang et al., 2024) with the
Normalized Gaussian Wasserstein Distance (Wang et al., 2021).

Reference: Lyu et al., "Lightweight RT-DETR with small-object feature
enhancement for traffic sign detection", JEI 2026.
"""
import torch
import torch.nn as nn


def _giou_2d(pred, target):
    """Axis-aligned 2-D GIoU for (cx, cy, w, l) boxes in meters."""
    px, py, pw, pl = pred.unbind(-1)
    tx, ty, tw, tl = target.unbind(-1)

    p_x1, p_x2 = px - pw / 2, px + pw / 2
    p_y1, p_y2 = py - pl / 2, py + pl / 2
    t_x1, t_x2 = tx - tw / 2, tx + tw / 2
    t_y1, t_y2 = ty - tl / 2, ty + tl / 2

    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)

    inter = (inter_x2 - inter_x1).clamp(min=0) * \
            (inter_y2 - inter_y1).clamp(min=0)
    area_p = pw * pl
    area_t = tw * tl
    union = area_p + area_t - inter + 1e-7

    iou = inter / union

    encl_x1 = torch.min(p_x1, t_x1)
    encl_y1 = torch.min(p_y1, t_y1)
    encl_x2 = torch.max(p_x2, t_x2)
    encl_y2 = torch.max(p_y2, t_y2)
    encl_area = (encl_x2 - encl_x1) * (encl_y2 - encl_y1) + 1e-7

    giou = iou - (encl_area - union) / encl_area
    return iou, giou


def _nwd(pred, target, C=3.0):
    """Normalized Wasserstein Distance between Gaussian-modeled 2-D boxes."""
    px, py, pw, pl = pred.unbind(-1)
    tx, ty, tw, tl = target.unbind(-1)

    diff = torch.stack([px - tx, py - ty, pw / 2 - tw / 2, pl / 2 - tl / 2],
                       dim=-1)
    dist = torch.norm(diff, dim=-1)
    return torch.exp(-dist / C)


class FGNLoss(nn.Module):
    """Focaler-GIoU-NWD loss.

    Args:
        d (float): Focaler lower bound. Default 0.0.
        u (float): Focaler upper bound. Default 0.95.
        C (float): NWD normalizing constant in meters. Default 3.0.
        eta (float): Mixing weight (eta * FG + (1-eta) * (1-NWD)). Default 0.5.
        loss_weight (float): Overall loss weight. Default 0.5.
    """

    def __init__(self, d=0.0, u=0.95, C=3.0, eta=0.5, loss_weight=0.5):
        super().__init__()
        self.d = d
        self.u = u
        self.C = C
        self.eta = eta
        self.loss_weight = loss_weight

    def forward(self, pred_bev, target_bev, weight=None, avg_factor=None):
        """
        Args:
            pred_bev: (N, 4) predicted BEV boxes (cx, cy, w, l) in meters.
            target_bev: (N, 4) target BEV boxes (cx, cy, w, l) in meters.
        """
        iou, giou = _giou_2d(pred_bev, target_bev)
        iou_f = ((iou - self.d) / (self.u - self.d)).clamp(0, 1)
        loss_fg = (1 - giou) + iou - iou_f

        nwd = _nwd(pred_bev, target_bev, self.C)
        loss_nwd = 1.0 - nwd

        loss = self.eta * loss_fg + (1 - self.eta) * loss_nwd

        if weight is not None:
            loss = loss * weight
        if avg_factor is not None:
            loss = loss.sum() / max(avg_factor, 1)
        else:
            loss = loss.mean()

        return loss * self.loss_weight
