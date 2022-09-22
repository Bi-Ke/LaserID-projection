"""
    Function: Semantic Segmentation evaluation criteria.

        Please note that 'SegmentationMetric' only supports numpy data format (CPU) while 'SegmentationMetricV1'
        supports torch data format (CPU/GPU).

          Confusion Matrix,
          Pixel Accuracy (PA) (commonly used),
          Class Pixel Accuracy (Precision) (CPA),
          Mean Pixel Accuracy (MPA),
          Intersection over Union (IoU) (commonly used),
          Mean Intersection over Union (MIoU) (commonly used).

          Confusion Matrix:
                        |   Prediction
           Ground Truth ----------------------
                        | Positive | Negative
          ------------------------------------
           Positive     |    TP    |   FN
          ------------------------------------
           Negative     |    FP    |   TN

           Accuracy: (TP + TN) / (TP + TN + FP + FN)
           Precision: TP / (TP + FN) (or: TN / (TN + FP))
           Recall: TP / (TP + FP) (or: TN / (TN + FN)) (not commonly used.)
           Mean Pixel Accuracy: sum(Precision) / C, C is the number of classes.
           Intersection over Union (IoU): TP / (TP + FP + FN)
           Mean Intersection over Union: sum(IoU) / C,
           Frequency Weight Intersection over Union (FWIoU):
                        [TP / (TP + FN)] * [TP / (TP + FP + FN)] + [TN / (TN + FP)] * [TN / (TN + FP + FN)]

    Date: June 28, 2021
"""

import numpy as np
import torch
from utils.utils import nanmean


class SegmentationMetric(object):
    """
    This is based on the numpy library.
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes  # the number of classes.
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def reset(self, confusion_matrix):
        if confusion_matrix is None:
            self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes))
        else:
            assert confusion_matrix.shape[0] == self.num_classes
            self.confusion_matrix = confusion_matrix

    def generate_confusion_matrix(self, gt, prediction):
        mask = (gt >= 0) & (gt < self.num_classes)  # remove classes from unlabeled pixels in gt and prediction.
        confusion_matrix = np.bincount(self.num_classes * gt[mask] + prediction[mask],
                                       minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
        return confusion_matrix

    def get_confusion_matrix(self):
        return self.confusion_matrix

    def add_batch(self, gt, prediction):
        assert gt.shape == prediction.shape
        gt = gt.flatten()  # if gt and prediction are not 1 dimension.
        prediction = prediction.flatten()
        self.confusion_matrix += self.generate_confusion_matrix(gt=gt, prediction=prediction)

    def pixel_accuracy(self):
        """
        Accuracy: (TP + TN) / (TP + TN + FP + FN)
        return: all classes overall pixel accuracy (overall accuracy).
        """
        accuracy = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return accuracy

    def class_pixel_accuracy(self):
        """
        Class Pixel Accuracy, Precision: TP / (TP + FN) (or: TN / (TN + FP))
        :return: each category pixel accuracy (A more accurate way to call it precision)(per-class accuracy)
        """
        class_accuracy = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return class_accuracy

    def mean_pixel_accuracy(self):
        """
        Mean Pixel Accuracy: sum(Precision) / C, C is the number of classes. (mean accuracy)
        """
        class_accuracy = self.class_pixel_accuracy()
        mean_accuracy = np.nanmean(class_accuracy)
        return mean_accuracy

    def intersection_over_union(self):
        """
        Intersection over Union (IoU): TP / (TP + FP + FN)
        """
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - intersection
        iou = intersection / union
        return iou

    def mean_intersection_over_union(self):
        """
        Mean Intersection over Union: sum(IoU) / C. (mean IoU)
        """
        iou = self.intersection_over_union()
        miou = np.nanmean(iou)
        return miou

    def frequency_weighted_intersection_over_union(self):
        """
        Frequency Weight Intersection over Union (FWIoU):
                        [TP / (TP + FN)] * [TP / (TP + FP + FN)] + [TN / (TN + FP)] * [TN / (TN + FP + FN)]
        """
        frequency = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iou = self.intersection_over_union()
        fwiou = (frequency[frequency > 0] * iou[frequency > 0]).sum()
        return fwiou


class SegmentationMetricV1(object):
    """
    This is based on the PyTorch platform.
    """
    def __init__(self, num_classes, device):
        self.num_classes = num_classes  # the number of classes.
        self.device = device
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes)).to(device)

    def reset(self, confusion_matrix=None):
        if confusion_matrix is None:
            self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes)).to(self.device)
        else:
            assert confusion_matrix.shape[0] == self.num_classes
            self.confusion_matrix = confusion_matrix.to(self.device)

    def generate_confusion_matrix(self, gt, prediction):
        mask = (gt >= 0) & (gt < self.num_classes)  # remove unlabeled pixels (i.e., 255) in gt and prediction.
        confusion_matrix = torch.bincount(self.num_classes * gt[mask] + prediction[mask],
                                          minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
        return confusion_matrix

    def get_confusion_matrix(self):
        return self.confusion_matrix

    def add_batch(self, gt, prediction):
        assert gt.shape == prediction.shape
        gt = gt.flatten()  # if gt and prediction are not 1 dimension.
        prediction = prediction.flatten()
        self.confusion_matrix += self.generate_confusion_matrix(gt=gt, prediction=prediction)

    def add_metric(self, metric):
        self.confusion_matrix += metric.confusion_matrix

    def pixel_accuracy(self):
        """
        Accuracy: (TP + TN) / (TP + TN + FP + FN)
        return: all classes overall pixel accuracy (overall accuracy).
        """
        accuracy = torch.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return accuracy

    def class_pixel_accuracy(self):
        """
        Class Pixel Accuracy, Precision: TP / (TP + FN) (or: TN / (TN + FP))
        :return: each category pixel accuracy (A more accurate way to call it precision)(per-class accuracy)
        """
        class_accuracy = torch.diag(self.confusion_matrix) / self.confusion_matrix.sum(dim=1)
        return class_accuracy

    def mean_pixel_accuracy(self):
        """
        Mean Pixel Accuracy: sum(Precision) / C, C is the number of classes. (mean accuracy)
        """
        class_accuracy = self.class_pixel_accuracy()
        mean_accuracy = nanmean(class_accuracy)
        return mean_accuracy

    def intersection_over_union(self):
        """
        Intersection over Union (IoU): TP / (TP + FP + FN)
        """
        intersection = torch.diag(self.confusion_matrix)
        union = torch.sum(self.confusion_matrix, dim=1) + torch.sum(self.confusion_matrix, dim=0) - intersection
        iou = intersection / union
        return iou

    def mean_intersection_over_union(self):
        """
        Mean Intersection over Union: sum(IoU) / C. (mean IoU)
        """
        iou = self.intersection_over_union()
        miou = nanmean(iou)
        return miou

    def frequency_weighted_intersection_over_union(self):
        """
        Frequency Weight Intersection over Union (FWIoU):
                        [TP / (TP + FN)] * [TP / (TP + FP + FN)] + [TN / (TN + FP)] * [TN / (TN + FP + FN)]
        """
        frequency = torch.sum(self.confusion_matrix, dim=1) / torch.sum(self.confusion_matrix)
        iou = self.intersection_over_union()
        fwiou = (frequency[frequency > 0] * iou[frequency > 0]).sum()
        return fwiou


if __name__ == "__main__":
    # SegmentationMetric
    gt = np.array([0, 1, 0, 2, 1, 0, 2, 2, 1])
    prediction = np.array([0, 1, 0, 2, 1, 0, 1, 2, 1])
    metric = SegmentationMetric(num_classes=3)
    # confusion_matrix = metric.generate_confusion_matrix(gt=gt, prediction=prediction)
    # print(confusion_matrix)

    metric.add_batch(gt=gt, prediction=prediction)

    pa = metric.pixel_accuracy()
    cpa = metric.class_pixel_accuracy()
    mpa = metric.mean_pixel_accuracy()
    mIoU = metric.mean_intersection_over_union()

    print("pa = {0}, \t cpa = {1}, \t mpa = {2}, \t mIoU = {3}".format(pa, cpa, mpa, mIoU))

    # SegmentationMetricV1
    device = torch.device("cuda")
    gt = torch.LongTensor([0, 1, 0, 2, 1, 0, 2, 2, 1]).to(device)
    prediction = torch.LongTensor([0, 1, 0, 2, 1, 0, 1, 2, 1]).to(device)
    metric = SegmentationMetricV1(num_classes=3, device=device)
    metric.add_batch(gt=gt, prediction=prediction)
    pa = metric.pixel_accuracy()
    cpa = metric.class_pixel_accuracy()
    mpa = metric.mean_pixel_accuracy()
    mIoU = metric.mean_intersection_over_union()
    print("pa = {0}, \t cpa = {1}, \t mpa = {2}, \t mIoU = {3}".format(pa, cpa, mpa, mIoU))

