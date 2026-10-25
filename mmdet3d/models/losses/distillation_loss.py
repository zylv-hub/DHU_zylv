"""Feature distillation loss for the compressed detector (Eq. 10).

Matches three BEV-level quantities between the student and the frozen
teacher:
  - f_I: 32-channel image BEV
  - M:   foreground mask
  - F:   BEV backbone output (adapted via 1x1 when widths differ)
  - E_k: image backbone stage outputs (optional, lambda_e)
"""
import torch
import torch.nn as nn


class DistillationLoss(nn.Module):
    """Multi-term feature distillation.

    Args:
        student_bev_dim (int): Student BEV backbone output channels.
        teacher_bev_dim (int): Teacher BEV backbone output channels.
        student_img_dims (list[int]): Student image backbone stage dims.
        teacher_img_dims (list[int]): Teacher image backbone stage dims.
        lambda_g (float): Weight for image BEV. Default 1.0.
        lambda_m (float): Weight for mask. Default 1.0.
        lambda_b (float): Weight for BEV backbone feature. Default 0.5.
        lambda_e (float): Weight for image backbone stages. Default 0.0
            (only used in domain-pretrained setting).
    """

    def __init__(self,
                 student_bev_dim=256,
                 teacher_bev_dim=256,
                 student_img_dims=None,
                 teacher_img_dims=None,
                 lambda_g=1.0,
                 lambda_m=1.0,
                 lambda_b=0.5,
                 lambda_e=0.0):
        super().__init__()
        self.lambda_g = lambda_g
        self.lambda_m = lambda_m
        self.lambda_b = lambda_b
        self.lambda_e = lambda_e

        self.phi = None
        if student_bev_dim != teacher_bev_dim:
            self.phi = nn.Conv2d(student_bev_dim, teacher_bev_dim, 1,
                                 bias=False)

        self.psi = nn.ModuleList()
        if student_img_dims and teacher_img_dims and lambda_e > 0:
            for s_dim, t_dim in zip(student_img_dims, teacher_img_dims):
                if s_dim != t_dim:
                    self.psi.append(nn.Conv2d(s_dim, t_dim, 1, bias=False))
                else:
                    self.psi.append(nn.Identity())

    def forward(self, student, teacher):
        """
        Args:
            student (dict): keys 'img_bev', 'mask', 'bev_feat',
                            optionally 'img_stages' (list of tensors).
            teacher (dict): same keys.
        Returns:
            torch.Tensor: scalar distillation loss.
        """
        loss = 0.0

        if self.lambda_g > 0:
            loss += self.lambda_g * torch.mean(
                (student['img_bev'] - teacher['img_bev']) ** 2)

        if self.lambda_m > 0:
            loss += self.lambda_m * torch.mean(
                (student['mask'] - teacher['mask']) ** 2)

        if self.lambda_b > 0:
            s_feat = student['bev_feat']
            t_feat = teacher['bev_feat']
            if self.phi is not None:
                s_feat = self.phi(s_feat)
            loss += self.lambda_b * torch.mean((s_feat - t_feat) ** 2)

        if self.lambda_e > 0 and 'img_stages' in student:
            for psi, s_stage, t_stage in zip(
                    self.psi, student['img_stages'], teacher['img_stages']):
                loss += self.lambda_e * torch.mean(
                    (psi(s_stage) - t_stage) ** 2)

        return loss
