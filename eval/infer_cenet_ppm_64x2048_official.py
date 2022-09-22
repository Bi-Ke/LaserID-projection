"""
    Function: evaluation functions. (support multi-GPU testing.)

    Date: September 2, 2022.
"""

import sys
sys.path.insert(0, '.')
import tqdm
import time
import datetime
import numpy as np
import random
import logging
import os
import os.path as osp
import argparse

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

from utils.generate_semantickitti_lut import get_label_remap_lut_color_lut
from utils.config import load_cfg_from_cfg_file
from models.architectures.resnet import ResNet34
# from data.semantic_kitti_range_image_data_constructor import SemanticKITTITestData
from data.semantic_kitti_range_image_data_constructorV1 import SemanticKITTITestData


# setting random seeds.
torch.manual_seed(123)  # cpu
torch.cuda.manual_seed_all(123)  # parallel gpu.
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.deterministic = True  # cpu/gpu consistent.
torch.backends.cudnn.benchmark = False
# torch.multiprocessing.set_sharing_strategy('file_system')


def get_parser():
    parser = argparse.ArgumentParser(description='Range Image Based Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, required=False, default='configs/laserid_cenet_ppm_64x2048.yaml',
                        help='configurations.')
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    return cfg


def get_logger():
    logger_name = "laserid_cenet_ppm_64x2048-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


class multi_scale_flip_eval(object):
    def __init__(self, scales=(0.5, 0.75, 1.0, 1.25, 1.5, 1.75), flip=False):
        self.scales = scales
        self.flip = flip

    @torch.no_grad()
    def __call__(self, net, test_loader, num_classes, save_dir, remap_lut_inv):
        net.eval()

        if dist.is_initialized() and dist.get_rank() == 0:  # multi-GPU
            logger.info("=> evaluation ...")
            logger.info("=> scales: {}".format(self.scales))
            logger.info("=> flip: {}".format(self.flip))
            progressbar = tqdm.tqdm(range(len(test_loader)))
        if not dist.is_initialized():  # single-GPU
            logger.info("=> evaluation ...")
            logger.info("=> scales: {}".format(self.scales))
            logger.info("=> flip: {}".format(self.flip))
            progressbar = tqdm.tqdm(range(len(test_loader)))

        for range_xyz_remission_img, range_xyz_raw_data, align_unproj_range_xyz_data, align_p2ri_lut, \
            num_valid_points, sequences_index, filename in test_loader:
            # directly use the proposed PPM method
            # projected
            img = range_xyz_remission_img.cuda()
            range_xyz_raw_data = range_xyz_raw_data.cuda()

            # unprojected
            align_unproj_range_xyz_data = align_unproj_range_xyz_data.cuda()

            # 'link'
            align_p2ri_lut = align_p2ri_lut.cuda()
            num_valid_points = num_valid_points.cuda()

            with torch.no_grad():
                probs = net(x=img,
                            proj_range_xyz=range_xyz_raw_data,
                            unproj_range_xyz=align_unproj_range_xyz_data,
                            p2ri_lut=align_p2ri_lut,
                            num_valid_pts=num_valid_points)[0]  # B x C x N.
                proj_argmax = torch.argmax(probs, dim=1)  # B x N

                B = align_p2ri_lut.shape[0]  # B x N x 3.
                for b in range(B):
                    b_num_valid_points = num_valid_points[b]
                    b_proj_argmax = proj_argmax[b][0:b_num_valid_points]   # (N,)
                    preds = b_proj_argmax.cpu().numpy().reshape(-1).astype(np.int32)  # (range: 0-18)

                    # remap to original labels.
                    preds = preds + 1  # 0, 1~19.
                    preds = remap_lut_inv[preds]

                    # saved the predicted labels.
                    sequences_dir = osp.join(save_dir, sequences_index[b], 'predictions')

                    try:
                        os.makedirs(sequences_dir)
                    except OSError:
                        pass

                    save_filename = osp.join(sequences_dir, filename[b]+'.label')
                    preds.tofile(save_filename)

            if dist.is_initialized() and dist.get_rank() == 0:  # multi-GPU.
                progressbar.update(1)
            if not dist.is_initialized():  # single-GPU
                progressbar.update(1)

        if dist.is_initialized() and dist.get_rank() == 0:  # multi-GPU.
            progressbar.close()
        if not dist.is_initialized():  # single-GPU.
            progressbar.close()

        torch.cuda.empty_cache()


def convert_relu_to_softplus(model, act):
    """convert ReLU or LeakyReLU to Hardswish or SiLU."""
    for child_name, child in model.named_children():
        if isinstance(child, nn.LeakyReLU):
            setattr(model, child_name, act)
        else:
            convert_relu_to_softplus(child, act)


def main_worker_dist(local_rank, args):
    """ multi-GPU evaluation. """
    # initialization. (1/5)
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:{}'.format(args["port"]),
                            world_size=args['num_gpus'],
                            rank=local_rank,
                            timeout=datetime.timedelta(seconds=300))

    # setting one process one gpu. (2/5)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    # logger.
    if dist.get_rank() == 0:
        global logger
        logger = get_logger()
        logger.info(args)

    if dist.get_rank() == 0:
        logger.info("******************** Start Time **************************")
        logger.info(time.strftime('%H:%M:%S', time.localtime()))

    # dataset.
    data_root = args['dataset']['data_root']
    # test_sequences = args['dataset']['test_sequences']
    test_sequences = args['dataset']['val_sequences']

    _, remap_lut_inv, _ = get_label_remap_lut_color_lut(data_config=args['dataset']['data_config'])

    if dist.get_rank() == 0:
        logger.info("=> data root: {}".format(data_root))
        logger.info("=> test sequences: {}".format(test_sequences))

    test_dataset = SemanticKITTITestData(data_root=data_root, sequences=test_sequences, CFG=args)

    # using DistributedSampler. (3/5)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    test_loader = Data.DataLoader(dataset=test_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  sampler=test_sampler,
                                  pin_memory=True,
                                  num_workers=2)

    # model.
    net = ResNet34(nclasses=args['dataset']['num_classes'],
                   aux=args['train']['aux_loss'],
                   search=args['post']['KNN']['params']['search'])  # 7x7.

    # convert ReLU (or LeakyReLU) to Hardswish or SiLU.
    if args['train']['act'] == 'Hardswish':
        convert_relu_to_softplus(net, nn.Hardswish())
    elif args['train']['act'] == 'SiLU':
        convert_relu_to_softplus(net, nn.SiLU())

    if dist.get_rank() == 0:
        logger.info("=> creating the model ...")
        logger.info(net)

    # save_best_val_model_path or save_best_train_model.pt
    if dist.get_rank() == 0:
        logger.info("=> Loading the best model: {}".format(args['train']['save_best_val_model_path']))
    saved_dict = torch.load(args['train']['save_best_val_model_path'], map_location='cpu')
    net.load_state_dict(saved_dict['net'])

    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net.to(device)
    net.eval()
    net = DistributedDataParallel(module=net,
                                  device_ids=[local_rank],
                                  output_device=local_rank)

    # metric (just save predicted labels)
    num_classes = args['dataset']['num_classes']  # 19
    # save_dir = args['dataset']['data_root']
    save_dir = 'Datasets/SemanticKitti/dataset/predictions_sequences_laserid_range_images_2048/sequences'

    if dist.get_rank() == 0:
        logger.info("=> save predicted labels to {}".format(save_dir))

    # WARNING: multi-scale testing might cause problems. Please check that again.
    if dist.get_rank() == 0:
        logger.info("=> ss: single scale evaluation ...")
    single_scale = multi_scale_flip_eval(scales=(1., ), flip=False)
    single_scale(net=net, test_loader=test_loader, num_classes=num_classes, save_dir=save_dir, remap_lut_inv=remap_lut_inv)

    if dist.get_rank() == 0:
        logger.info("******************* End Time **********************")
        logger.info(time.strftime('%H:%M:%S', time.localtime()))
        logger.info("Successfully !")
    dist.destroy_process_group()


if __name__ == "__main__":
    args = get_parser()

    # multi-GPU evaluation.
    num_gpus = args['num_gpus']
    mp.spawn(main_worker_dist, nprocs=num_gpus, args=(args,))
    print("Successfully.")

