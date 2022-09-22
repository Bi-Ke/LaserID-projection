"""
    Function: data augmentation especially for SemanticKITTI.

    RandomNoiseXYZ, (p = 0.25, jitter X, Y, Z are [-5, 5), [-3, 3), [-1, 0))
    RandomHorizontalFlip, (p = 0.25)
    RandomRotate, (p = 0.25, rotation angle is [0, 180) degrees)
    RandomDrop, (p = 0.5, drop ratio is [0, 0.5))
    ToTensor
    Compose

    TODO:
    RandomResizedCrop, (ratios of 0.5 ~ 2.0)
    RandomVerticalFlip, (p = 0.5)
    RandomResize, (ratios of 0.5 ~ 2.0)
    CenterCrop
    RandomCrop

    Recommend usage 1: RandomResizedCrop -> RandomHorizontalFlip -> ColorJitter.
    Recommend usage 2: RandomResize -> RandomCrop -> RandomHorizontalFlip -> ColorJitter.

    Theoretically, RandomResizedCrop = RandomResize + RandomCrop.

    Date: August 29, 2022.
"""

import random
import math
import numpy as np
import cv2
import numbers
import torch


class RandomHorizontalFlip(object):
    """random horizontal flip with the probability of p=0.25"""
    def __init__(self, p=0.25):
        self.p = p

    def __call__(self, img_gt):
        """
        img_gt['img']: (H, W, C); img_gt['gt']: (H, W); img_gt['p2ri_lut']: (N, 3), [[index, y, x],...]
        """
        if np.random.random() > self.p:
            return img_gt

        img, gt, p2ri_lut = img_gt['img'], img_gt['gt'], img_gt['p2ri_lut']
        assert img.shape[:2] == gt.shape[:2]

        img = img[:, ::-1, :]
        if gt is not None:
            gt = gt[:, ::-1]
        if p2ri_lut is not None:
            w = img.shape[1] - 1  # 2048 - 1 = 2047. [0, 1, 2, ..., 2047] -> [2047, ..., 2, 1, 0]
            p2ri_lut[:, 2] = w - p2ri_lut[:, 2]
        return dict(img=img, gt=gt, p2ri_lut=p2ri_lut)

    def __repr__(self):
        return self.__class__.__name__ + '(probability = {0})'.format(self.p)


class RandomDrop(object):
    """randomly dropout points with the probability of p=0.5"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_gt):
        """
        img_gt['img']: (H, W, C); img_gt['gt']: (H, W); img_gt['p2ri_lut']: (N, 3), [[index, y, x],...]
        """
        if np.random.random() > self.p:
            return img_gt

        img, gt, p2ri_lut = img_gt['img'], img_gt['gt'], img_gt['p2ri_lut']
        assert img.shape[:2] == gt.shape[:2]

        drop_ratio = random.uniform(0, 0.5)
        h, w = img.shape[:2]
        drop_points = np.random.randint(0, h*w, int(h * w * drop_ratio))
        img = img.reshape(h*w, -1)  # (H*W, C)
        img[drop_points, :] = 0
        img = img.reshape(h, w, -1)  # H x W x C
        if gt is not None:
            gt = gt.reshape(h*w, -1)
            gt[drop_points, 0] = 255  # label
            gt[drop_points, 1] = 0  # mask
            gt = gt.reshape(h, w, -1)  # H x W x 2
        if p2ri_lut is not None:  # WARNING: only be used under the condition of existing a ground truth.
            y_coords = p2ri_lut[:, 1]  # vertical
            x_coords = p2ri_lut[:, 2]  # horizontal
            p2ri_lut = np.delete(p2ri_lut, gt[y_coords, x_coords, 1] == 0, axis=0)
        return dict(img=img, gt=gt, p2ri_lut=p2ri_lut)

    def __repr__(self):
        return self.__class__.__name__ + '(probability = {0})'.format(self.p)


class RandomVerticalFlip(object):
    """TODO: random vertical flip with the probability of p=0.5"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_gt):
        """
        img: (H, W, C)
        """
        if np.random.random() < self.p:
            return img_gt

        img, gt = img_gt['img'], img_gt['gt']
        assert img.shape[:2] == gt.shape[:2]
        img = img[::-1, :, :]
        gt = gt[::-1, :]
        return dict(img=img, gt=gt)

    def __repr__(self):
        return self.__class__.__name__ + '(probability = {0})'.format(self.p)


class RandomRotate(object):
    """randomly rotate the range image along width with the probability of 0.25"""
    def __init__(self, p=0.25):
        self.p = p

    def __call__(self, img_gt):
        """
        img_gt['img']: (H, W, C); img_gt['gt']: (H, W); img_gt['p2ri_lut']: (N, 3), [[index, y, x],...]
        """
        if np.random.random() > self.p:
            return img_gt

        img, gt, p2ri_lut = img_gt['img'], img_gt['gt'], img_gt['p2ri_lut']
        assert img.shape[:2] == gt.shape[:2]

        w = img.shape[1]
        random_angle = np.random.random()  # [0, 1)
        rotate_w = int(w // 2 * random_angle)  # 2048 // 2 = 1024 equals to 360 // 2 = 180 degrees.

        img = np.roll(img, shift=rotate_w, axis=1)
        if gt is not None:
            gt = np.roll(gt, shift=rotate_w, axis=1)
        if p2ri_lut is not None:
            # p2ri_lut[:, 2] = np.roll(p2ri_lut[:, 2], shift=rotate_w)  # there are multiple x values.
            p2ri_lut[:, 2] = p2ri_lut[:, 2] + rotate_w
            p2ri_lut[:, 2][p2ri_lut[:, 2] >= w] -= w

        return dict(img=img, gt=gt, p2ri_lut=p2ri_lut)

    def __repr__(self):
        return self.__class__.__name__ + '(Randomly rotate the range image with the probability = {0})'.format(self.p)


class RandomResize(object):
    """ TODO:
    Please note that using cv.INTER_NEAREST will harm the ground truth.
    One effective way to end this problem is to directly scale the points of polygon surrounding objects.
    """
    def __init__(self, scales=(0.5, 2.0)):
        self.scales = scales

    def __call__(self, img_gt):
        """
        :param img_gt: img_gt['img'] (H, W, C); img_gt['gt'] (H, W)
        """
        img, gt = img_gt['img'], img_gt['gt']
        assert img.shape[:2] == gt.shape[:2]

        scale = np.random.uniform(min(self.scales), max(self.scales))  # a scale in [0.5, 2.0)
        img_h, img_w = [math.ceil(el * scale) for el in img.shape[:2]]
        img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(gt, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        return dict(img=img, gt=gt)

    def __repr__(self):
        return self.__class__.__name__+" (random scales = {0})".format(self.scales)


class CenterCrop(object):
    """ TODO:
    crop an image from the image center.
    """
    def __init__(self, size=(64, 2048)):
        """
        :param size: (h, w)
        """
        if isinstance(size, numbers.Number):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img_gt):
        """
        :param img_gt: img_gt['img'] (H, W, C); img_gt['gt'] (H, W)
        """
        img, gt = img_gt['img'], img_gt['gt']
        img_h, img_w, _ = img.shape
        crop_h, crop_w = self.size

        if (img_h, img_w) == (crop_h, crop_w):
            return dict(img=img, gt=gt)

        pad_h, pad_w = 0, 0
        if img_h < crop_h:
            pad_h = (crop_h - img_h) // 2 + 1

        if img_w < crop_w:
            pad_w = (crop_w - img_w) // 2 + 1

        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), 'constant')
            gt[:, :, 0] = np.pad(gt[:, :, 0], ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=255)
            gt[:, :, 1] = np.pad(gt[:, :, 1], ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=0)  # mask for SemanticKITTI.

        img_h, img_w, _ = img.shape
        sh = (img_h - crop_h) // 2
        sw = (img_w - crop_w) // 2
        img = img[sh:sh+crop_h, sw:sw+crop_w, :].copy()
        gt = gt[sh:sh+crop_h, sw:sw+crop_w].copy()
        return dict(img=img, gt=gt)

    def __repr__(self):
        return self.__class__.__name__+"(cropping size = {0})".format(self.size)


class RandomCrop(object):
    """ TODO:
    crop a given image at a random location.
    """
    def __init__(self, crop_size=(64, 2048)):
        """
        padding: img, constant, 0; gt, constant, 255.
        crop_size: (h, w)
        """
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size

    def __call__(self, img_gt):
        """
        :param img_gt: img_gt['img'] (H, W, C); img_gt['gt'] (H, W).
        """
        img, gt = img_gt['img'], img_gt['gt']  # img: HWC
        assert img.shape[:2] == gt.shape[:2]

        img_h, img_w, _ = img.shape
        crop_h, crop_w = self.crop_size
        if (img_h, img_w) == (crop_h, crop_w):
            return dict(img=img, gt=gt)

        padw = (crop_w - img_w) // 2 + 1 if img_w < crop_w else 0
        padh = (crop_h - img_h) // 2 + 1 if img_h < crop_h else 0

        if padw > 0 or padh > 0:
            img = np.pad(img, ((padh, padh), (padw, padw), (0, 0)), 'constant')
            gt[:, :, 0] = np.pad(gt[:, :, 0], ((padh, padh), (padw, padw)), 'constant', constant_values=255)
            gt[:, :, 1] = np.pad(gt[:, :, 1], ((padh, padh), (padw, padw)), 'constant', constant_values=0)  # mask for SemanticKITTI.

        img_h, img_w = img.shape[:2]
        x = random.randint(0, img_w - crop_w)
        y = random.randint(0, img_h - crop_h)

        img = img[y:y+crop_h, x:x+crop_w, :].copy()
        gt = gt[y:y+crop_h, x:x+crop_w].copy()
        return dict(img=img, gt=gt)

    def __repr__(self):
            return self.__class__.__name__ + '(random cropping size={0})'.format(self.crop_size)


class RandomResizedCrop(object):
    def __init__(self, scales=(0.5, 2.0), size=(64, 2048)):
        """TODO:
        :param scales: random resize with ratio of [0.5, 2.0).
        :param size: should be a tuple of (H, W), or (S, S).
        """
        self.scales = scales
        if isinstance(size, numbers.Number):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img_gt):
        """
        img_gt['img'] (H x W x C) and img_gt['gt'] (H x W): images and corresponding labels.
        """
        if self.size is None:
            return img_gt

        img, gt = img_gt['img'], img_gt['gt']
        assert img.shape[:2] == gt.shape[:2]

        crop_h, crop_w = self.size
        scale = np.random.uniform(min(self.scales), max(self.scales))  # a scale in [0.5, 2.0)
        img_h, img_w = [math.ceil(el * scale) for el in img.shape[:2]]
        img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(gt, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        if (img_h, img_w) == (crop_h, crop_w):
            return dict(img=img, gt=gt)

        pad_h, pad_w = 0, 0
        if img_h < crop_h:
            pad_h = (crop_h - img_h) // 2 + 1

        if img_w < crop_w:
            pad_w = (crop_w - img_w) // 2 + 1

        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), 'constant')
            gt[:, :, 0] = np.pad(gt[:, :, 0], ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=255)
            gt[:, :, 1] = np.pad(gt[:, :, 1], ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=0)  # mask for SemanticKITTI.

        img_h, img_w, _ = img.shape
        sh, sw = np.random.random(2)
        sh, sw = int(sh * (img_h - crop_h)), int(sw * (img_w - crop_w))

        img = img[sh:sh+crop_h, sw:sw+crop_w, :].copy()
        gt = gt[sh:sh+crop_h, sw:sw+crop_w].copy()

        return dict(img=img, gt=gt)

    def __repr__(self):
        return self.__class__.__name__ + '(random scales = {0}, cropping size = {1})'.format(self.scales, self.size)


class RandomNoiseXYZ(object):
    """
    adding random noise to X, Y, Z values. probability=0.25.
    """
    def __init__(self, p=0.25):
        self.p = p

    def __call__(self, img_gt):
        """
        :param img_gt: img_gt['img']: (H, W, C); img_gt['gt']: (H, W); img_gt['p2ri_lut']: (N, 3), [[index, y, x],...]
        """
        if np.random.random() > self.p:
            return img_gt

        img, gt, p2ri_lut = img_gt['img'], img_gt['gt'], img_gt['p2ri_lut']
        assert img.shape[:2] == gt.shape[:2]

        jitter_x = random.uniform(-5, 5)
        jitter_y = random.uniform(-3, 3)
        jitter_z = random.uniform(-1, 0)
        img[:, :, 1] += jitter_x
        img[:, :, 2] += jitter_y
        img[:, :, 3] += jitter_z
        return dict(img=img, gt=gt, p2ri_lut=p2ri_lut)

    def __repr__(self):
        return self.__class__.__name__ + '({} probability of adding random noise to XYZ)'.format(0.5)


class ToTensor(object):
    """
    Converts a numpy.ndarray image (H x W x C) to a torch.FloatTensor of shape (C x H x W) with normalization.
    """
    def __init__(self, mean=(0, 0, 0, 0, 0), std=(1.0, 1.0, 1.0, 1.0, 1.0)):
        """
        :param mean: range_image, x, y, z, remission.
        :param std: range_image, x, y, z, remission.
        """
        self.mean = mean
        self.std = std

    def __call__(self, img_gt):
        """
        :param img_gt: img_gt['img']: (H, W, C); img_gt['gt']: (H, W).
        """
        img, gt, p2ri_lut = img_gt['img'], img_gt['gt'], img_gt['p2ri_lut']
        img = img.transpose(2, 0, 1).copy()  # HWC -> CHW
        img = torch.from_numpy(img)
        dtype, device = img.dtype, img.device

        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)[:, None, None]
        std = torch.as_tensor(self.std, dtype=dtype, device=device)[:, None, None]

        img = img.sub_(mean).div_(std).clone()  # CHW

        if gt is not None:
            gt = torch.from_numpy(gt.astype(np.int64).copy()).clone()

        if p2ri_lut is not None:
            p2ri_lut = torch.from_numpy(p2ri_lut.astype(np.int64).copy()).clone()

        return dict(img=img, gt=gt, p2ri_lut=p2ri_lut)


class Compose(object):
    """
    Composes several transforms together.
    Example:
        >>> transforms.Compose([
        >>>     transforms.RandomDrop(p = 0.5),
        >>>     transforms.RandomHorizontalFlip(p = 0.25),
        >>>     transforms.RandomNoiseXYZ(p = 0.25),
        >>>     transforms.RandomRotate(p = 0.25),
        >>>     transforms.ToTensor(mean=(0, 0, 0, 0, 0), std=(1.0, 1.0, 1.0, 1.0, 1.0)),
        >>> ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img_gt):
        for t in self.transforms:
            img_gt = t(img_gt)
        return img_gt

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "   {0}".format(t)
        format_string += "\n)"
        return format_string


if __name__ == "__main__":
    pass
