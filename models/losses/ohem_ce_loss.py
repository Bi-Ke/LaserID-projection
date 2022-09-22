"""
    Function: OHEM + CrossEntropy Loss.

    Copy from: https://github.com/CoinCheung/BiSeNet
               https://github.com/MichaelFan01/STDC-Seg
               https://github.com/HRNet/HRNet-Semantic-Segmentation

    Date: November 17, 2021.
    Updated: November 23, 2021. adding OhemCELossV1&V2, WeightedOhemCELoss, SoftmaxFocalLoss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import enet_weighing


#  import ohem_cpp
#  class OhemCELoss(nn.Module):
#
#      def __init__(self, thresh, ignore_lb=255):
#          super(OhemCELoss, self).__init__()
#          self.score_thresh = thresh
#          self.ignore_lb = ignore_lb
#          self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='mean')
#
#      def forward(self, logits, labels):
#          n_min = labels[labels != self.ignore_lb].numel() // 16
#          labels = ohem_cpp.score_ohem_label(
#                  logits, labels, self.ignore_lb, self.score_thresh, n_min).detach()
#          loss = self.criteria(logits, labels)
#          return loss


class OhemCELoss(nn.Module):
    """
    default value of thresh is 0.7, or at least 1/16 of all pixels in each batchsize.
    """
    def __init__(self, thresh, ignore_lb=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        # return torch.mean(loss_hard)
        return torch.sum(loss_hard)


class OhemCELossV1(nn.Module):
    """
    different implementation from the above OhemCELoss.
    """
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELossV1, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class WeightedOhemCELoss(nn.Module):
    """
    considering class imbalance weights in the loss function.
    """
    def __init__(self, thresh, n_min, num_classes, ignore_lb=255, *args, **kwargs):
        super(WeightedOhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.num_classes = num_classes
        # self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        criteria = nn.CrossEntropyLoss(weight=enet_weighing(labels, self.num_classes).cuda(),
                                       ignore_index=self.ignore_lb, reduction='none')
        loss = criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class OhemCELossV2(nn.Module):
    """
    different implementation from the above OhemCELoss.
    In addition, parts of branches use traditional Cross Entropy Loss.
    """
    def __init__(self, ignore_label=-1, thres=0.7, min_kept=100000, weight=None):
        super(OhemCELossV2, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _ce_forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.criterion(score, target)
        return loss

    def _ohem_forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)  # Cross Entropy Loss.
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target, weights):
        """
        :param score: outputs from the model.
        :param target: ground truth.
        :param weights: weights for losses of auxiliary branches.
        :return:
        """
        if len(score) == 1:
            score = [score]
        assert len(weights) == len(score)

        functions = [self._ce_forward] * (len(weights) - 1) + [self._ohem_forward]
        return sum([w * func(x, target) for (w, x, func) in zip(weights, score, functions)])


if __name__ == '__main__':
    torch.manual_seed(15)
    criteria1 = OhemCELossV1(thresh=0.7, n_min=16*20*20//16).cuda()
    criteria2 = OhemCELossV1(thresh=0.7, n_min=16*20*20//16).cuda()
    net1 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net1.cuda()
    net1.train()
    net2 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net2.cuda()
    net2.train()

    with torch.no_grad():
        inten = torch.randn(16, 3, 20, 20).cuda()
        lbs = torch.randint(0, 19, [16, 20, 20]).cuda()
        lbs[1, :, :] = 255

    logits1 = net1(inten)
    logits1 = F.interpolate(logits1, inten.size()[2:], mode='bilinear', align_corners=True)
    logits2 = net2(inten)
    logits2 = F.interpolate(logits2, inten.size()[2:], mode='bilinear', align_corners=True)

    loss1 = criteria1(logits1, lbs)
    loss2 = criteria2(logits2, lbs)
    loss = loss1 + loss2
    print(loss.detach().cpu())
    loss.backward()
