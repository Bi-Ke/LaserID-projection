"""
    Function: some useful functions.

    reference: https://github.com/pytorch/examples/tree/master/imagenet

    Date: July 5, 2021
    Updated: October 18, 2021. Adding Kullback-Leibler Divergence.
"""
import shutil
import torch
from PIL import Image
import json
import numpy as np
import os.path as osp
import glob
import torch.nn.functional as F


def accuracy(prediction, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = prediction.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeterV1(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')


def nanmean(v, *args, inplace=False, **kwargs):
    """ This is like np.nanmean """
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


def imresize(im, size, interp='bilinear'):
    """
    resize an image using the PIL.Image library.
    """
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)


# useful functions for loading ADE20K and COCO-Stuff datasets.
def parse_input_list(data_file, max_sample=-1, start_idx=-1, end_idx=-1):
    """
    :param self:
    :param data_file: like the followings:

    {img_path: xxx0.jpg, gt_path: xxx0.png, img_width: xxx0, img_height: xxx0}
    {img_path: xxx1.jpg, gt_path: xxx1.png, img_width: xxx1, img_height: xxx1}
                ......
    {img_path: xxxN.jpg, gt_path: xxxN.png, img_width: xxxN, img_height: xxxN}

    :param max_sample:
    :param start_idx:
    :param end_idx:
    :return:
    """
    list_sample = None
    if isinstance(data_file, list):
        list_sample = data_file
    elif isinstance(data_file, str):
        list_sample = [json.loads(x.rstrip()) for x in open(data_file, 'r')]

    if max_sample > 0:
        list_sample = list_sample[0:max_sample]

    if start_idx >= 0 and end_idx >= 0:     # divide file list
        list_sample = list_sample[start_idx:end_idx]

    num_sample = len(list_sample)
    assert num_sample > 0
    print('# samples: {}'.format(num_sample))
    return list_sample


def round2nearest_multiple(x, p):
    """
    Round x to the nearest x' which is multiple of p and x' >= x so as to avoid segmentation label misalignment.
    :param self: 
    :param x: 
    :param p: 
    :return: 
    """
    return ((x - 1) // p + 1) * p


def split_list(lst, num):
    split_lst = []
    for i in range(0, len(lst), num):
        split_lst.append(lst[i:i+num])
    return split_lst


def color_encode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3), dtype=np.uint8)
    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
                        np.tile(colors[label], (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


def kl_divergence(P, Q, get_softmax=True, reduction="batchmean"):
    """
    Kullback-Leibler divergence Loss, KL(P || Q) = (P * torch.log(P / Q)).sum()
    :param P: (*, dims)
    :param Q: (*, dims)
    :param get_softmax:
    :param reduction: "batchmean", "sum", default: "sum".
    :return:
    """
    if get_softmax:
        P = F.softmax(P, dim=-1)
        Q = F.softmax(Q, dim=-1)

    p_size = P.size(-2)
    q_size = Q.size(-2)
    tmp_P = torch.repeat_interleave(P, q_size, dim=-2).view(p_size, q_size, -1)

    if reduction == "batchmean":
        kl_div = (tmp_P * torch.log(tmp_P / Q)).mean(dim=-1)
    else:
        kl_div = (tmp_P * torch.log(tmp_P / Q)).sum(dim=-1)  # default.
    return kl_div


def js_divergence(P, Q, get_softmax=True, reduction="batchmean"):
    """
    Jensen Shannon Divergence. JS(P || Q) = 1/2 * KL(P || M) + 1/2 * KL(Q || M), M = 1/2 * (P + Q).
                               KL(P || Q) = (P * torch.log(P / Q)).sum()
    :param P: (*, dims)
    :param Q: (*, dims)
    :param get_softmax:
    :param reduction:
    :return:
    """
    if get_softmax:
        P = F.softmax(P, dim=-1)
        Q = F.softmax(Q, dim=-1)

    p_size = P.size(-2)
    q_size = Q.size(-2)
    tmp_P = torch.repeat_interleave(P, q_size, dim=-2)  # [p_size * q_size, 3]
    tmp_Q = Q.repeat(p_size, 1)  # [p_size * q_size, 3]
    sum_PQ = (tmp_P + tmp_Q) * 0.5  # sum_PQ:[p_size * q_size, 3]

    if reduction == "batchmean":
        js_div = 0.5 * (P * torch.log(P / sum_PQ)).mean(dim=-1) + 0.5 * (Q * torch.log(Q / sum_PQ)).mean(dim=-1)
    else:
        js_div = 0.5 * (tmp_P * torch.log(tmp_P / sum_PQ)).sum(dim=-1) + \
                 0.5 * (tmp_Q * torch.log(tmp_Q / sum_PQ)).sum(dim=-1)

    attn = 1.0 - js_div
    return attn.view(p_size, q_size)


def load_imgs(path):
    """
    load all of images from a path.
    :param path, single image: "path/to/xxx.jpg" or multiple images: "path/to/dir"
    """
    img_paths = []
    if osp.isfile(path=path):  # single file.
        img_paths.append(path)
    elif osp.isdir(path):
        paths = glob.glob(path + "/*")
        paths.sort()
        img_paths = paths
    return img_paths


class ToTensor(object):
    """
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, mean=(0, 0, 0), std=(1.0, 1.0, 1.0)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        img (PIL Image or numpy.ndarray): Image to be converted to tensor.
        """
        img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)  # bgr->rgb, HWC -> CHW
        img = torch.from_numpy(img).div_(255.0)
        dtype, device = img.dtype, img.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)[:, None, None]
        std = torch.as_tensor(self.std, dtype=dtype, device=device)[:, None, None]
        img = img.sub_(mean).div_(std).clone()
        return img


if __name__ == "__main__":
    # P = torch.ones(2, 3) * torch.Tensor([[1, 2, 3], [2, 4, 6]])
    # print(P)
    # tensor([[1., 2., 3.],
    #         [1., 2., 3.]])

    # Q = torch.randn(4, 3)
    # print(Q)
    # tensor([[ 0.4468, -0.7075, -0.6142],
    #         [ 0.3937, -0.7542,  0.0175],
    #         [-1.7459, -0.3096, -0.5978],
    #         [ 1.1521, -0.9023,  0.5080]])

    # P = torch.tensor([[0.4, 0.4, 0.2], [0.4, 0.4, 0.2]], dtype=torch.float32)
    # Q = torch.tensor([[0.5, 0.1, 0.4], [0.5, 0.1, 0.4], [0.4, 0.4, 0.2]], dtype=torch.float32)

    P = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    Q = torch.tensor([[0.05, 0.05, 0.9], [0.15, 0.15, 0.7], [0.15, 0.15, 0.7]], dtype=torch.float32)

    kl_div = kl_divergence(P, Q, reduction="sum")
    print("kl_div = ", kl_div)

    js_div = js_divergence(P, Q, reduction="sum")
    print("js_div = ", js_div)
    # js_div = js_divergence(Q, P, reduction="sum")
    # print(js_div)

    print(F.softmax(js_div, dim=-1))

    attn = similarity_func(P, Q)
    print("attn = ", attn)

    """
    tensor([[0.3330, 0.3330, 0.3340],
        [0.3330, 0.3330, 0.3340]])
    attn =  tensor([[0.3308, 0.3308, 0.3385],
            [0.3308, 0.3308, 0.3385]])
    """


    # device = torch.device("cuda")
    # v1 = torch.randint(10, (2, 2)).to(device)
    # mean_v1 = nanmean(v1)
    # print(v1)
    # print(mean_v1)

    # lst = [1, 2, 3, 4, 5, 6, 7, 6, 5, 3]
    # num = 3
    # split_lst = split_list(lst, num)
    # print(split_lst)

    # x = 10
    # p = 1
    # y = round2nearest_multiple(x, p)
    # print(y)

