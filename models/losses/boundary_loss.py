"""
    Function: Boundary Loss.

    Copy from:
    https://github.com/yiskw713/boundary_loss_for_remote_sensing

    Proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852

    Revising the generation of one-hot, supporting removing the ignore label 255
    However, note that (cannot consider other ignore labels less than num_classes).

    Default: mean reduction.

    Date: August 31, 2022.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryLoss(nn.Module):
    def __init__(self, theta0=3, theta=5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (WARNING: before softmax), shape (N, C, H, W)
            - gt: ground truth map, shape (N, H, W)
        Return:
            - boundary loss, averaged over mini-batch (Revised).
        """
        n, c, _, _ = pred.shape

        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred, dim=1)  # N x C x H x W

        # one-hot vector of ground truth
        one_hot_gt = torch.stack([(gt == cls).float() for cls in range(c)], dim=1)  # N x C x H x W (ignore label 255).

        # boundary map
        gt_b = F.max_pool2d(1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)
        pred_b_ext = F.max_pool2d(pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, c, -1)  # N x C x H*W
        pred_b = pred_b.view(n, c, -1)  # N x C x H*W
        gt_b_ext = gt_b_ext.view(n, c, -1)  # N x C x H*W
        pred_b_ext = pred_b_ext.view(n, c, -1)  # N x C x H*W

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)  # N x C x H*W
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)  # N x C x H*W

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)  # N x C x H*W

        # summing BF1 Score for each class and average over mini-batch.
        loss = torch.mean(1 - BF1)
        return loss


# class BoundaryLoss(nn.Module):
#     def __init__(self, theta0=3, theta=5):
#         super().__init__()
#         self.theta0 = theta0
#         self.theta = theta
#
#     def forward(self, pred, gt):
#         """
#         Input:
#             - pred: the output from model (WARNING: before softmax), shape (N, C, H, W)
#             - gt: ground truth map, shape (N, H, w)
#         Return:
#             - boundary loss, averaged over mini-batch (Revised).
#         """
#         n, c, _, _ = pred.shape
#
#         # softmax so that predicted map can be distributed in [0, 1]
#         pred = torch.softmax(pred, dim=1)  # N x C x H x W
#
#         # one-hot vector of ground truth
#         one_hot_gt = one_hot(gt, c)  # N x C x H x W
#
#         # boundary map
#         gt_b = F.max_pool2d(1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
#         gt_b -= 1 - one_hot_gt
#
#         pred_b = F.max_pool2d(1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
#         pred_b -= 1 - pred
#
#         # extended boundary map
#         gt_b_ext = F.max_pool2d(gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)
#         pred_b_ext = F.max_pool2d(pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)
#
#         # reshape
#         gt_b = gt_b.view(n, c, -1)  # N x C x H*W
#         pred_b = pred_b.view(n, c, -1)  # N x C x H*W
#         gt_b_ext = gt_b_ext.view(n, c, -1)  # N x C x H*W
#         pred_b_ext = pred_b_ext.view(n, c, -1)  # N x C x H*W
#
#         # Precision, Recall
#         P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)  # N x C x H*W
#         R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)  # N x C x H*W
#
#         # Boundary F1 Score
#         BF1 = 2 * P * R / (P + R + 1e-7)  # N x C x H*W
#
#         # summing BF1 Score for each class and average over mini-batch.
#         loss = torch.mean(1 - BF1)
#         return loss
#
#
# def one_hot(label, n_classes, requires_grad=True):
#     """Return One Hot Label"""
#     device = label.device
#     one_hot_label = torch.eye(n_classes, device=device, requires_grad=requires_grad)[label]
#     one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)
#     return one_hot_label


# for debug
if __name__ == "__main__":
    import torch.optim as optim
    from torchvision.models import segmentation

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img = torch.randn(8, 3, 224, 224).to(device)
    gt = torch.randint(0, 10, (8, 224, 224)).to(device)

    gt[:, 0:10, 0:10] = 255

    model = segmentation.fcn_resnet50(num_classes=10).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = BoundaryLoss()

    y = model(img)

    loss = criterion(y['out'], gt)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("loss: ", loss)
