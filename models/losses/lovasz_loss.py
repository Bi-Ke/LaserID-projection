"""
    Function: Lovasz-Softmax Loss.

    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/LovaszSoftmax/lovasz_loss.py

    Adding the consideration of ignore_label=255.

    Date: August 31, 2022.
"""

import torch
import torch.nn as nn


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmax(nn.Module):
    def __init__(self, ignore_label=255, reduction='mean'):
        super(LovaszSoftmax, self).__init__()
        self.ignore_label = ignore_label
        self.reduction = reduction

    def prob_flatten(self, input, target):
        # [1, C, 1, 'N'], [1, 1, 'N']
        assert input.dim() in [4, 5]  # N x C x H x W
        num_class = input.size(1)
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1).contiguous()  # N x C x H x W -> N x H x W x C
            input_flatten = input.view(-1, num_class)
        elif input.dim() == 5:
            input = input.permute(0, 2, 3, 4, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        target_flatten = target.view(-1)

        # remove ignore labels and corresponding inputs.
        mask = target_flatten != self.ignore_label
        input_flatten = input_flatten[mask, :]
        target_flatten = target_flatten[mask]
        return input_flatten, target_flatten

    def lovasz_softmax_flat(self, inputs, targets):
        num_classes = inputs.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (targets == c).float()
            if num_classes == 1:
                input_c = inputs[:, 0]
            else:
                input_c = inputs[:, c]
            loss_c = (torch.autograd.Variable(target_c) - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(lovasz_grad(target_c_sorted))))
        losses = torch.stack(losses)

        if self.reduction == 'none':
            loss = losses
        elif self.reduction == 'sum':
            loss = losses.sum()
        else:
            loss = losses.mean()
        return loss

    def forward(self, inputs, targets):
        # print(inputs.shape, targets.shape) # (batch size, class_num, x, y, z), (batch size, 1, x, y, z)
        # [1, C, 1, 'N'], [1, 1, 'N']
        assert len(inputs.shape) == 4, "shape: N x C x H x W."
        inputs = torch.softmax(inputs, dim=1)
        inputs, targets = self.prob_flatten(inputs, targets)
        # print(inputs.shape, targets.shape)
        losses = self.lovasz_softmax_flat(inputs, targets)
        return losses



