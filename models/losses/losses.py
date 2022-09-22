"""
    Function: combination of various losses.

    a. CE Loss, Boundary Loss, and Lovasz Softmax Loss.
    b. Cross Entropy Loss, Online Hard Example Mining.

    Date: August 31, 2022.
"""

import torch
import torch.nn as nn
from .boundary_loss import BoundaryLoss
from .lovasz_loss import LovaszSoftmax


# *****************************************************************************************
# class CombineCELovaszSoftmaxBoundaryLoss(nn.Module):
#     """combination of CE Loss, Boundary Loss, and Lovasz Softmax Loss.
#     Default reduction mode is 'mean', because the Boundary Loss only supports the 'mean' reduction.
#     """
#     def __init__(self, weights, ignore_index, coef):
#         super(CombineCELovaszSoftmaxBoundaryLoss, self).__init__()
#         self.coef = coef
#         self.ce = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index, reduction='mean')
#         self.bd = BoundaryLoss(theta0=3, theta=5)  # only 'mean' reduction mode.
#         self.ls = LovaszSoftmax(ignore_label=255, reduction='mean')
#
#     def forward(self, pred, gt):
#         output, head2, head4, head8 = pred
#         bd_loss = self.bd(output, gt) + self.coef * (self.bd(head2, gt) + self.bd(head4, gt) + self.bd(head8, gt))
#         loss_output = self.ce(output, gt) + 1.5 * self.ls(output, gt)
#         loss_head2 = self.ce(head2, gt) + 1.5 * self.ls(head2, gt)
#         loss_head4 = self.ce(head4, gt) + 1.5 * self.ls(head4, gt)
#         loss_head8 = self.ce(head8, gt) + 1.5 * self.ls(head8, gt)
#         loss = loss_output + self.coef * (loss_head2 + loss_head4 + loss_head8) + bd_loss
#         return loss

class CombineCELovaszSoftmaxBoundaryLoss(nn.Module):
    """combination of CE Loss, Boundary Loss, and Lovasz Softmax Loss.
    Default reduction mode is 'mean', because the Boundary Loss only supports the 'mean' reduction.
    """
    def __init__(self, weights, ignore_index, coef):
        super(CombineCELovaszSoftmaxBoundaryLoss, self).__init__()
        self.coef = coef
        self.ce = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index, reduction='mean')
        self.bd = BoundaryLoss(theta0=3, theta=5)  # only 'mean' reduction mode.
        self.ls = LovaszSoftmax(ignore_label=255, reduction='mean')

    def forward(self, pred, gt_img, gt_pt, num_valid_pts, p2ri_lut):
        """ pred:
            # main_head_out:  torch.Size([2, 19, N])
            # head1_out:  torch.Size([2, 19, 64, 2048])
            # head2_out:  torch.Size([2, 19, 64, 2048])
            # head3_out:  torch.Size([2, 19, 64, 2048])
            # head4_out:  torch.Size([2, 19, 64, 2048])

            gt_img: B x H x W
            gt_pt: B x N
            num_valid_pts: B x 1
            p2ri_lut: B x M x 3, and N != M.
        """
        main_head_out, output, head2, head4, head8 = pred

        assert main_head_out.shape[-1] == gt_pt.shape[-1], "must be same size."
        assert output.shape[2:] == gt_img.shape[1:], "must be same size."

        # for range images.
        bd_loss = self.bd(output, gt_img) + self.coef * (self.bd(head2, gt_img) + self.bd(head4, gt_img) + self.bd(head8, gt_img))
        loss_output = self.ce(output, gt_img) + 1.5 * self.ls(output, gt_img)
        loss_head2 = self.ce(head2, gt_img) + 1.5 * self.ls(head2, gt_img)
        loss_head4 = self.ce(head4, gt_img) + 1.5 * self.ls(head4, gt_img)
        loss_head8 = self.ce(head8, gt_img) + 1.5 * self.ls(head8, gt_img)
        loss = loss_output + self.coef * (loss_head2 + loss_head4 + loss_head8) + bd_loss

        # for final points.
        B = main_head_out.shape[0]
        tmp_main_losses = torch.zeros(B).type_as(main_head_out)
        for b in range(B):
            b_num_valid_pts = num_valid_pts[b]
            b_p2ri_lut = p2ri_lut[b][0:b_num_valid_pts, :]
            b_idx_pt = b_p2ri_lut[:, 0]
            b_main_head_out = main_head_out[b][:, b_idx_pt].unsqueeze(dim=0).unsqueeze(dim=2)  # [1, C, 1, 'N']
            b_gt_pt = gt_pt[b][b_idx_pt].unsqueeze(dim=0).unsqueeze(dim=0)  # [1, 1, 'N']
            tmp_main_loss = self.ce(b_main_head_out, b_gt_pt) + 1.5 * self.ls(b_main_head_out, b_gt_pt)
            tmp_main_losses[b] = tmp_main_loss

        main_loss = torch.mean(tmp_main_losses)
        loss = main_loss + self.coef * loss
        return loss


# *****************************************************************************************
class OhemCELoss(nn.Module):
    """
    default value of thresh is 0.7, or at least 1/16 of all pixels in each batchsize.
    """
    def __init__(self, weights=None, thresh=0.7, ignore_lb=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float))  # .cuda()
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.sum(loss_hard)


class CombineOhemCELoss(nn.Module):
    """combination of Cross Entropy Loss and Online Hard Example Mining."""
    def __init__(self, weights, ignore_index, coef):
        super(CombineOhemCELoss, self).__init__()
        self.coef = coef
        self.ce_ohem = OhemCELoss(weights=weights, thresh=0.7, ignore_lb=ignore_index)

    def forward(self, pred, gt):
        output, head2, head4, head8 = pred
        loss_output = self.ce_ohem(output, gt)
        loss_head2 = self.ce_ohem(head2, gt)
        loss_head4 = self.ce_ohem(head4, gt)
        loss_head8 = self.ce_ohem(head8, gt)
        loss = loss_output + self.coef * (loss_head2 + loss_head4 + loss_head8)  # + bd_loss
        return loss
