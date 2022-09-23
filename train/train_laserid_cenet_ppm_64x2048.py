"""
    Function: training CENet+PPM on LaserID Projected Range Images.

    Date: August 29, 2022.
"""

import sys
sys.path.insert(0, '.')
import random
import numpy as np
import tqdm
import pickle
import time
import datetime
import logging
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

from utils.config import load_cfg_from_cfg_file
from utils.metrics import SegmentationMetricV1
from utils.lr_scheduler import WarmupPolyLrScheduler, WarmupCosineAnnealingWarmRestartsPolyLrScheduler

from models.architectures.resnet import ResNet34
from models.losses.losses import CombineCELovaszSoftmaxBoundaryLoss
# from models.losses.losses import CombineOhemCELoss
# from data.semantic_kitti_range_image_data_constructor import SemanticKITTITrainData
from data.semantic_kitti_range_image_data_constructorV1 import SemanticKITTITrainData

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


def train(net, train_loader, device, optimizer, criterion, scheduler, epoch, metric):
    # train
    net.train()
    losses = []
    metric.reset()

    if dist.get_rank() == 0:
        progressbar = tqdm.tqdm(range(len(train_loader)))

    for train_img, train_gt, range_xyz_raw_data, align_unproj_range_xyz_data, align_unproj_range_xyz_label, align_p2ri_lut, num_valid_points in train_loader:
        # projected
        img = train_img.to(device)
        gt = train_gt.to(device)
        range_xyz_raw_data = range_xyz_raw_data.to(device)

        # unprojected.
        align_unproj_range_xyz_data = align_unproj_range_xyz_data.to(device)
        align_unproj_range_xyz_label = align_unproj_range_xyz_label.to(device)

        # 'link'
        align_p2ri_lut = align_p2ri_lut.to(device)
        num_valid_points = num_valid_points.to(device)

        optimizer.zero_grad()
        predict = net(x=img,
                      proj_range_xyz=range_xyz_raw_data,
                      unproj_range_xyz=align_unproj_range_xyz_data,
                      p2ri_lut=align_p2ri_lut,
                      num_valid_pts=num_valid_points)

        loss = criterion(pred=predict,
                         gt_img=gt,
                         gt_pt=align_unproj_range_xyz_label,
                         num_valid_pts=num_valid_points,
                         p2ri_lut=align_p2ri_lut)
        loss.backward()

        optimizer.step()
        torch.cuda.synchronize()
        scheduler.step()

        # collect losses from multiple processes.
        losses.append(loss.detach().item())

        if dist.get_rank() == 0:
            lr = scheduler.get_lr()
            lr = sum(lr) / len(lr)
            progressbar.set_description("epoch {0},  loss {1},  lr {2} \n".format(epoch, loss.detach().item(), lr))
            progressbar.update(1)

        # calculate the mIoU of semantic segmentation. (B x C x N)
        indices = torch.argmax(predict[0], dim=1)  # predict[0], main output.

        B = num_valid_points.shape[0]
        for b in range(B):
            b_num_valid_pts = num_valid_points[b]
            b_p2ri_lut = align_p2ri_lut[b][0:b_num_valid_pts, :]
            b_idx_pt = b_p2ri_lut[:, 0]
            metric.add_batch(gt=align_unproj_range_xyz_label[b][b_idx_pt],
                             prediction=indices.detach()[b][b_idx_pt])

    if dist.get_rank() == 0:
        progressbar.close()

    # synchronize and compute loss.
    loss = torch.mean(torch.Tensor(losses).to(device))
    dist.all_reduce(loss, dist.ReduceOp.SUM)  # sum all losses from all processes.

    # synchronize and compute miou.
    confusion_matrix = metric.get_confusion_matrix()
    dist.all_reduce(confusion_matrix, dist.ReduceOp.SUM)  # sum all confusion matrix from all processes.
    metric.reset(confusion_matrix=confusion_matrix)
    miou = metric.mean_intersection_over_union().cpu().item()

    del indices, metric
    del img, gt, predict, losses
    torch.cuda.empty_cache()
    return net, optimizer, scheduler, loss.cpu().item(), miou


def validate(net, val_loader, device, criterion, epoch, metric):
    # validate
    net.eval()
    losses = []
    metric.reset()

    if dist.get_rank() == 0:
        progressbar = tqdm.tqdm(range(len(val_loader)))

    for val_img, val_gt, range_xyz_raw_data, align_unproj_range_xyz_data, align_unproj_range_xyz_label, align_p2ri_lut, num_valid_points in val_loader:
        # projected
        img = val_img.to(device)
        gt = val_gt.to(device)
        range_xyz_raw_data = range_xyz_raw_data.to(device)

        # unprojected.
        align_unproj_range_xyz_data = align_unproj_range_xyz_data.to(device)
        align_unproj_range_xyz_label = align_unproj_range_xyz_label.to(device)

        # 'link'
        align_p2ri_lut = align_p2ri_lut.to(device)
        num_valid_points = num_valid_points.to(device)

        with torch.no_grad():
            predict = net(x=img,
                          proj_range_xyz=range_xyz_raw_data,
                          unproj_range_xyz=align_unproj_range_xyz_data,
                          p2ri_lut=align_p2ri_lut,
                          num_valid_pts=num_valid_points)

            # loss.
            loss = criterion(pred=predict,
                             gt_img=gt,
                             gt_pt=align_unproj_range_xyz_label,
                             num_valid_pts=num_valid_points,
                             p2ri_lut=align_p2ri_lut)
            losses.append(loss.detach().item())

            # mIoU.
            indices = torch.argmax(predict[0], dim=1)  # predict[0], main output.

            B = num_valid_points.shape[0]
            for b in range(B):
                b_num_valid_pts = num_valid_points[b]
                b_p2ri_lut = align_p2ri_lut[b][0:b_num_valid_pts, :]
                b_idx_pt = b_p2ri_lut[:, 0]
                metric.add_batch(gt=align_unproj_range_xyz_label[b][b_idx_pt],
                                 prediction=indices.detach()[b][b_idx_pt])

        if dist.get_rank() == 0:
            progressbar.set_description("epoch {0} \t, loss {1}  \n".format(epoch, loss.detach().item()))
            progressbar.update(1)

    if dist.get_rank() == 0:
        progressbar.close()

    # synchronize and compute loss.
    loss = torch.mean(torch.Tensor(losses).to(device))
    dist.all_reduce(loss, dist.ReduceOp.SUM)  # sum all losses from all processes.

    # synchronize and compute miou.
    confusion_matrix = metric.get_confusion_matrix()
    dist.all_reduce(confusion_matrix, dist.ReduceOp.SUM)  # sum all confusion matrix from all processes.
    metric.reset(confusion_matrix=confusion_matrix)
    miou = metric.mean_intersection_over_union().cpu().item()

    del indices, metric
    del img, gt, predict, losses
    torch.cuda.empty_cache()
    return loss.cpu().item(), miou


def get_data_loader(args):
    data_root = args['dataset']['data_root']
    train_sequences = args['dataset']['train_sequences']
    val_sequences = args['dataset']['val_sequences']

    if dist.get_rank() == 0:
        logger.info("=> data root: {}".format(data_root))
        logger.info("=> train sequences: {}".format(train_sequences))
        logging.info("=> val sequences: {}".format(val_sequences))

    train_dataset = SemanticKITTITrainData(data_root=data_root, sequences=train_sequences, CFG=args, is_train=True)
    val_dataset = SemanticKITTITrainData(data_root=data_root, sequences=val_sequences, CFG=args, is_train=False)

    # using DistributedSampler (3/5)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = Data.DataLoader(dataset=train_dataset,
                                   batch_size=args['train']['train_batch_size'],  # for single GPU.
                                   shuffle=False,
                                   sampler=train_sampler,
                                   pin_memory=True,
                                   num_workers=args['train']['train_num_workers'])
    val_loader = Data.DataLoader(dataset=val_dataset,
                                 batch_size=args['train']['val_batch_size'],
                                 shuffle=False,
                                 sampler=val_sampler,
                                 pin_memory=True,
                                 num_workers=args['train']['val_num_workers'])
    return train_loader, val_loader, train_dataset, val_dataset, train_sampler, val_sampler


def set_model(args, device):
    net = ResNet34(args['dataset']['num_classes'], args['train']['aux_loss'],
                   search=args['post']['KNN']['params']['search'])
    # convert ReLU (or LeakyReLU) to Hardswish or SiLU.
    if args['train']['act'] == 'Hardswish':
        convert_relu_to_softplus(net, nn.Hardswish())
    elif args['train']['act'] == 'SiLU':
        convert_relu_to_softplus(net, nn.SiLU())

    if dist.get_rank() == 0:
        logger.info("=> creating the model ...")
        logger.info(net)

    # loading a pretrained model?
    # assert args['train']['pretrained_model_path'] is not None
    # if dist.get_rank() == 0:
    #     logger.info("Loading the pretrained model: {} ...".format(args['train']['pretrained_model_path']))
    # saved_dict = torch.load(args['train']['pretrained_model_path'], map_location='cpu')
    # net.load_state_dict(saved_dict['net'])

    # is continuous to train (resume)?
    saved_dict = None
    if args['train']['continue_to_train']:
        assert args['train']['save_last_model_path'] is not None
        if dist.get_rank() == 0:
            logger.info("=> resume? {}".format(args['train']['continue_to_train']))
            logger.info("=> Loading the last model: {}".format(args['train']['save_last_model_path']))
        saved_dict = torch.load(args['train']['save_last_model_path'], map_location='cpu')
        net.load_state_dict(saved_dict['net'])

    # Synchronize BatchNorm2D
    net = set_syncbn(net)

    if dist.get_rank() == 0:
        total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        logger.info("The number of parameters: {} M".format(total_params/1000000))

    net.to(device)
    net.train()
    return net, saved_dict


def set_syncbn(net):
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    return net


def convert_relu_to_softplus(model, act):
    """convert ReLU or LeakyReLU to Hardswish or SiLU."""
    for child_name, child in model.named_children():
        if isinstance(child, nn.LeakyReLU):
            setattr(model, child_name, act)
        else:
            convert_relu_to_softplus(child, act)


def set_optimizer(net, saved_dict, args):
    # AdamW
    optimizer = optim.AdamW(params=net.parameters(),
                            lr=args['train']['learning_rate'],
                            betas=(0.9, 0.999),
                            weight_decay=args['train']['weight_decay'])
    # SGD
    # optimizer = optim.SGD(params=net.parameters(),
    #                       lr=args["train"]["consine"]["min_lr"],
    #                       momentum=args["train"]["momentum"],
    #                       weight_decay=args["train"]["w_decay"])

    if args['train']['continue_to_train']:
        optimizer.load_state_dict(saved_dict['optim'])  # load pretrained parameters.

    return optimizer


def set_model_dist(net, local_rank):
    net = DistributedDataParallel(module=net,
                                  device_ids=[local_rank],
                                  output_device=local_rank)
    return net


def set_criterion(args, device):
    """set loss functions."""
    DATA = load_cfg_from_cfg_file(args['dataset']['data_config'])

    # transform original labels to 0-19.
    remap_dict = DATA["learning_map"]
    max_key = max(remap_dict.keys())
    remap_lut = np.zeros((max_key + 100), dtype=np.int32)
    remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

    epsilon_w = args["train"]["epsilon_w"]
    num_classes = len(DATA['learning_map_inv'])  # 20
    content = torch.zeros(num_classes, dtype=torch.float)

    for cls, freq in DATA["content"].items():
        x_cls = remap_lut[cls]  # map actual class to 0~19.
        content[x_cls] += freq
    weights = 1 / (content + epsilon_w)  # get weights

    for x_cls, weight in enumerate(weights):  # ignore the ones necessary to ignore
        if DATA["learning_ignore"][x_cls]:
            # weights[x_cls] = 0  # don't weight
            weights = np.delete(weights, x_cls)  # don't weight

    coef = args['train']['coef']
    criterion = CombineCELovaszSoftmaxBoundaryLoss(weights=weights, ignore_index=255, coef=coef).to(device)
    # criterion = CombineOhemCELoss(weights=weights, ignore_index=255, coef=coef).to(device)
    return criterion


def main_worker(local_rank, args):
    # initialization. (1/5)
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:{}'.format(args['port']),
                            world_size=args['num_gpus'],
                            rank=local_rank,
                            timeout=datetime.timedelta(seconds=300))

    # setting one process one gpu (2/5)
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

    # datasets
    train_loader, val_loader, train_dataset, val_dataset, train_sampler, val_sampler = get_data_loader(args=args)

    # model.
    net, saved_dict = set_model(args=args, device=device)

    # optimizer.
    optimizer = set_optimizer(net, saved_dict, args=args)

    # DistributedDataParallel training.
    net = set_model_dist(net, local_rank=local_rank)

    # lr scheduler.
    iters_per_epoch = args['dataset']['num_train_samples'] // (args['num_gpus'] * args['train']['train_batch_size']) + 1
    # max_iteration = args['train']['max_iteration']
    # max_epoch = max_iteration // iters_per_epoch + 1
    max_epoch = args['train']['max_epoch']

    s_max_iter = max_epoch * iters_per_epoch
    # scheduler = WarmupPolyLrScheduler(optimizer,
    #                                   power=args['train']['warmup_power'],
    #                                   max_iter=s_max_iter,
    #                                   warmup_iter=args['train']['warmup_iters'],
    #                                   warmup_ratio=args['train']['warmup_ratio'],
    #                                   warmup=args['train']['warmup_type'],
    #                                   last_epoch=-1,)

    scheduler = WarmupCosineAnnealingWarmRestartsPolyLrScheduler(optimizer,
                                                                 max_iter=s_max_iter,
                                                                 inter_iter=80000,  # 40000.
                                                                 power=args['train']['warmup_power'],
                                                                 T_0=1000,
                                                                 T_mult=2,
                                                                 eta_ratio=0,
                                                                 warmup_iter=args['train']['warmup_iters'],
                                                                 warmup_ratio=args['train']['warmup_ratio'],
                                                                 warmup=args['train']['warmup_type'],
                                                                 last_epoch=-1)

    if dist.get_rank() == 0:
        logger.info("=> max_epoch: {}".format(max_epoch))
        logger.info("=> max iterations: {}".format(s_max_iter))

    if args['train']['continue_to_train']:
        scheduler.load_state_dict(saved_dict['scheduler'])
        scheduler.step(epoch=(saved_dict['epoch'] + 1) * iters_per_epoch)   # fixing steps.

    # criteria.
    # criterion = WeightedCELoss(CFG=args, ignore_label=255).to(device)
    criterion = set_criterion(args=args, device=device)

    # segmentation metric.
    metric = SegmentationMetricV1(num_classes=args['dataset']['num_classes'], device=device)

    # iteration.
    train_losses = []
    train_mious = []

    val_losses = []
    val_mious = []

    # current best miou.
    best_val_miou = 0.0
    best_train_miou = 0.0

    if args['train']['continue_to_train']:
        best_val_miou = saved_dict['best_val_miou']
        best_train_miou = saved_dict['best_train_miou']
        if dist.get_rank() == 0:
            logger.info("best val mIoU = {0}".format(best_val_miou))
            logger.info("best train mIoU = {}".format(best_train_miou))

    if args['train']['continue_to_train']:
        # validate.
        val_loss, val_miou = validate(net=net,
                                      val_loader=val_loader,
                                      device=device,
                                      criterion=criterion,
                                      epoch=saved_dict['epoch'],
                                      metric=metric)
        val_losses.append(val_loss / args['num_gpus'])  # averaged loss value (one process, one batchsize).
        val_mious.append(val_miou)  # it's accurate.

        # selecting the best model in terms of mIoU values obtained on the validation dataset.
        if dist.get_rank() == 0:
            if val_mious[-1] > best_val_miou:
                best_val_miou = val_mious[-1]
                logger.info("=> best Val mIoU {0:.3f}".format(best_val_miou))
                logger.info("=> Save The Best Val Model !")

                save_info = {'epoch': saved_dict['epoch'],
                             'net': net.module.state_dict(),
                             'optim': optimizer.state_dict(),
                             'scheduler': scheduler.state_dict(),
                             'best_train_miou': best_train_miou,
                             'best_val_miou': best_val_miou}

                torch.save(save_info, args['train']['save_best_val_model_path'])

    start_epoch = 0
    if args['train']['continue_to_train']:
        start_epoch = saved_dict['epoch'] + 1

    # iteration.
    for epoch in range(start_epoch, max_epoch):
        if dist.get_rank() == 0:
            logger.info(f'epoch : {epoch}')

        train_dataset.shuffle()  # is useful ?
        val_dataset.shuffle()  # is useful ?
        # train_sampler.set_epoch(epoch=epoch)  # is useful?
        # val_sampler.set_epoch(epoch=epoch)  # is useful?

        # train.
        net, optimizer, scheduler, train_loss, train_miou = train(net=net,
                                                                  train_loader=train_loader,
                                                                  device=device,
                                                                  optimizer=optimizer,
                                                                  criterion=criterion,
                                                                  scheduler=scheduler,
                                                                  epoch=epoch,
                                                                  metric=metric)
        train_losses.append(train_loss / args['num_gpus'])  # averaged loss value (one process, one batchsize).
        train_mious.append(train_miou)  # it is not accurate due to the changes of weights in each iteration.

        # save the last training model.
        if dist.get_rank() == 0:
            logger.info("=> Save The Last Model !")

            save_info = {'epoch': epoch,
                         'net': net.module.state_dict(),
                         'optim': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'best_train_miou': best_train_miou,
                         'best_val_miou': best_val_miou}

            torch.save(save_info, args['train']['save_last_model_path'])

        # save the best training model.
        if dist.get_rank() == 0:
            if train_mious[-1] > best_train_miou:
                best_train_miou = train_mious[-1]
                logger.info("=> Save The Best Training Model !")

                save_info = {'epoch': epoch,
                             'net': net.module.state_dict(),
                             'optim': optimizer.state_dict(),
                             'scheduler': scheduler.state_dict(),
                             'best_train_miou': best_train_miou,
                             'best_val_miou': best_val_miou}
                torch.save(save_info, args['train']['save_best_train_model_path'])

        # validate.
        val_loss, val_miou = validate(net=net,
                                      val_loader=val_loader,
                                      device=device,
                                      criterion=criterion,
                                      epoch=epoch,
                                      metric=metric)
        val_losses.append(val_loss / args['num_gpus'])  # averaged loss value (one process, one batchsize).
        val_mious.append(val_miou)  # it's accurate.

        # show and save model weights.
        if dist.get_rank() == 0:
            logger.info("=> Train loss {0:.3f} \t, Val loss {1:.3f} \n".format(train_losses[-1], val_losses[-1]))
            logger.info("=> Train mIoU {0:.3f} \t, Val mIoU {1:.3f} \n".format(train_mious[-1], val_mious[-1]))
            logger.info("=> Best Train mIoU {0:.3f}".format(best_train_miou))
            logger.info("=> Best Val mIoU {0:.3f}".format(best_val_miou))

        # selecting the best model in terms of mIoU values obtained on the validation dataset.
        if dist.get_rank() == 0:
            if val_mious[-1] > best_val_miou:
                best_val_miou = val_mious[-1]
                logger.info("=> Save The Best Model !")

                save_info = {'epoch': epoch,
                             'net': net.module.state_dict(),
                             'optim': optimizer.state_dict(),
                             'scheduler': scheduler.state_dict(),
                             'best_train_miou': best_train_miou,
                             'best_val_miou': best_val_miou}

                torch.save(save_info, args['train']['save_best_val_model_path'])

        # save intermediate models.
        if dist.get_rank() == 0 and (epoch + 1) % 50 == 0:
            logger.info("=> Save The Intermediate Model !")

            save_info = {'epoch': epoch,
                         'net': net.module.state_dict(),
                         'optim': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'best_train_miou': best_train_miou,
                         'best_val_miou': best_val_miou}

            torch.save(save_info, args['train']['save_intime_model_path']+str(epoch+1)+".pt")

        # save train_loss, val_loss, train_miou, and val_miou.
        if dist.get_rank() == 0:
            with open(args['train']['save_loss_mIoU_path'], 'ab') as file:
                pickle.dump({'train_loss': train_losses[-1],
                             'val_loss': val_losses[-1],
                             'train_miou': train_mious[-1],
                             'val_miou': val_mious[-1]}, file)
    if dist.get_rank() == 0:
        logger.info("******************* End Time **********************")
        logger.info(time.strftime('%H:%M:%S', time.localtime()))
        logger.info("Successfully !")
    dist.destroy_process_group()


if __name__ == "__main__":
    args = get_parser()
    num_gpus = args['num_gpus']
    mp.spawn(main_worker, nprocs=num_gpus, args=(args,))

