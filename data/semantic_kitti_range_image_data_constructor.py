"""
    Function: data constructor for the distributed training on the SemanticKITTI dataset (i.e., range images).

    Date: August 29, 2022.
"""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import glob
from torch.utils import data
import os.path as osp
import cv2
from utils.transformations import RandomDrop, RandomHorizontalFlip, RandomNoiseXYZ, RandomRotate, ToTensor, Compose
from utils.generate_semantickitti_lut import get_label_remap_lut_color_lut


class SemanticKITTITrainData(data.Dataset):
    """
    train_sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    val_sequences = ['08']
    """
    def __init__(self, data_root, sequences, CFG, is_train=True):
        """
        :param data_root: 'Datasets/SemanticKitti/dataset/sequences'
        :param sequences: train_sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
                          val_sequences = ['08']
        """
        super(SemanticKITTITrainData, self).__init__()
        assert osp.exists(data_root), 'please provide the data_root'
        assert sequences is not None, 'please specify sequences.'

        means = CFG['dataset']['sensor']['img_means']
        std = CFG['dataset']['sensor']['img_stds']
        self.max_points = CFG['dataset']['max_points']  # 133376.
        self.remap_lut, _, _ = get_label_remap_lut_color_lut(CFG['dataset']['data_config'])

        # projected data
        self.range_images_filenames = []
        self.xyz_filenames = []
        self.remission_filenames = []
        self.labels_filenames = []
        self.range_image_2_points_mask_filenames = []
        self.points_2_range_image_lut_filenames = []
        # unprojected data
        self.unproj_points_filenames = []
        self.unproj_labels_filenames = []

        if osp.basename(data_root) == 'sequences_spherical_range_images_2048':
            unproj_points_root = data_root.replace('sequences_spherical_range_images_2048', 'sequences')
        else:  # sequences_laserid_range_images_2048
            unproj_points_root = data_root.replace('sequences_laserid_range_images_2048', 'sequences')

        for sequence in sequences:
            # projected data
            tmp_range_images_filenames = glob.glob(osp.join(data_root, sequence, 'range_image', '*.npy'))
            tmp_range_images_filenames.sort()
            self.range_images_filenames += tmp_range_images_filenames

            tmp_xyz_filenames = glob.glob(osp.join(data_root, sequence, 'range_image_2_points_matrix', '*.npy'))
            tmp_xyz_filenames.sort()
            self.xyz_filenames += tmp_xyz_filenames

            tmp_remission_filenames = glob.glob(osp.join(data_root, sequence, 'remissions', '*.npy'))
            tmp_remission_filenames.sort()
            self.remission_filenames += tmp_remission_filenames

            tmp_labels_filenames = glob.glob(osp.join(data_root, sequence, 'range_image_gt', '*.png'))
            tmp_labels_filenames.sort()
            self.labels_filenames += tmp_labels_filenames

            tmp_range_image_2_points_mask_filenames = glob.glob(osp.join(data_root, sequence, 'range_image_2_points_mask', '*.npy'))
            tmp_range_image_2_points_mask_filenames.sort()
            self.range_image_2_points_mask_filenames += tmp_range_image_2_points_mask_filenames

            tmp_points_2_range_image_lut_filenames = glob.glob(osp.join(data_root, sequence, 'points_2_range_image_lut', '*.npy'))
            tmp_points_2_range_image_lut_filenames.sort()
            self.points_2_range_image_lut_filenames += tmp_points_2_range_image_lut_filenames

            # unprojected data
            tmp_unproj_points_filenames = glob.glob(osp.join(unproj_points_root, sequence, 'velodyne', '*.bin'))
            tmp_unproj_points_filenames.sort()
            self.unproj_points_filenames += tmp_unproj_points_filenames

            tmp_unproj_labels_filenames = glob.glob(osp.join(unproj_points_root, sequence, 'labels', '*.label'))
            tmp_unproj_labels_filenames.sort()
            self.unproj_labels_filenames += tmp_unproj_labels_filenames

        assert len(self.range_images_filenames) == len(self.labels_filenames), 'must be the same.'
        assert len(self.unproj_points_filenames) == len(self.unproj_labels_filenames), 'must be the same.'

        self.num = len(self.range_images_filenames)
        self.permutation = np.random.permutation(self.num)

        if is_train:
            self.transform_function = Compose([RandomDrop(p=0.5),  # 64 x 2084
                                               RandomHorizontalFlip(p=0.25),
                                               RandomNoiseXYZ(p=0.25),
                                               RandomRotate(p=0.25),
                                               ToTensor(means, std)])
        else:
            self.transform_function = Compose([ToTensor(means, std)])

    @staticmethod
    def _get_unproj_data(filename: str):
        """read points from the .bin file."""
        scan = np.fromfile(filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        num_points = scan.shape[0]

        unproj_points = scan[:, 0:3]    # get xyz, (N, 3)
        # unproj_remissions = scan[:, 3]  # get remission, (N,)

        # num_points = unproj_remissions.shape[0]
        # unproj_remissions = unproj_remissions.reshape(num_points, 1)  # (N, 1)

        unproj_ranges = np.linalg.norm(unproj_points, 2, axis=1)  # unproj_ranges
        unproj_ranges = unproj_ranges.reshape(num_points, 1)

        # unproj_data = np.concatenate([unproj_ranges, unproj_points, unproj_remissions], axis=-1)  # (N, 5)
        unproj_data = np.concatenate([unproj_ranges, unproj_points], axis=-1)  # (N, 4)
        return unproj_data

    def _get_unproj_labels(self, filename: str):
        """read labels from the .label file."""
        label = np.fromfile(filename, dtype=np.uint32)
        label = label.reshape((-1))
        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16    # instance id in upper half
        assert((sem_label + (inst_label << 16) == label).all())

        # label mapping.
        sem_label = self.remap_lut[sem_label]  # to 0-19.
        sem_label = sem_label.astype(np.uint8)

        sem_label = sem_label - 1
        sem_label[sem_label == -1] = 255  # 0~18, 255.
        return sem_label

    def __getitem__(self, item):
        # data (augmentation)
        range_image = np.load(self.range_images_filenames[self.permutation[item]])  # H x W
        H, W = range_image.shape
        range_image = range_image.reshape((H, W, 1))  # H x W x 1
        xyz = np.load(self.xyz_filenames[self.permutation[item]])  # H x W x 3
        remission = np.load(self.remission_filenames[self.permutation[item]])
        remission = remission.reshape((H, W, 1))  # H x W x 1
        data = np.concatenate([range_image, xyz, remission], axis=-1)  # H x W x 5

        # label
        label = cv2.imread(self.labels_filenames[self.permutation[item]], cv2.IMREAD_GRAYSCALE)  # H x W
        label = label - 1  # 0~19 -> -1~18
        label[label == -1] = 255  # 0~18, 255 (ignore)

        # data (raw)
        range_xyz_raw_data = torch.from_numpy(np.concatenate([range_image, xyz], axis=-1))  # H x W x 4.
        range_xyz_raw_data = range_xyz_raw_data.permute(2, 0, 1).contiguous()  # 4 x H x W.

        # unprojected data
        unproj_range_xyz_data = self._get_unproj_data(self.unproj_points_filenames[self.permutation[item]])  # (N, 4)
        unproj_range_xyz_data = torch.from_numpy(unproj_range_xyz_data)  # (N, 4)

        # unprojected label
        unproj_range_xyz_label = self._get_unproj_labels(self.unproj_labels_filenames[self.permutation[item]])
        unproj_range_xyz_label = torch.from_numpy(unproj_range_xyz_label).long()

        # other
        mask = np.load(self.range_image_2_points_mask_filenames[self.permutation[item]])  # H x W, (0 or 1)
        p2ri_lut = np.load(self.points_2_range_image_lut_filenames[self.permutation[item]])  # N x 3.

        # data augmentation.
        label_mask = np.stack([label, mask], axis=-1)  # H x W x 2.
        img_gt = dict(img=data, gt=label_mask, p2ri_lut=p2ri_lut)
        img_gt = self.transform_function(img_gt)
        data, label_mask, p2ri_lut = img_gt['img'], img_gt['gt'], img_gt['p2ri_lut']
        range_xyz_remission_label = label_mask[:, :, 0]
        mask = label_mask[:, :, 1]
        range_xyz_remission_img = data * mask  # only keep valid values.

        # get the number of valid points.
        num_valid_points = p2ri_lut.shape[0]

        # align all data.
        align_p2ri_lut = torch.zeros(self.max_points, p2ri_lut.shape[1])  # 'N' x 3.
        align_p2ri_lut[0:num_valid_points, :] = p2ri_lut

        align_unproj_range_xyz_data = torch.zeros(self.max_points, unproj_range_xyz_data.shape[1])
        align_unproj_range_xyz_data[p2ri_lut[:, 0], :] = unproj_range_xyz_data[p2ri_lut[:, 0], :]  # 'N' x 4

        align_unproj_range_xyz_label = torch.full((self.max_points,), 255).long()
        align_unproj_range_xyz_label[p2ri_lut[:, 0]] = unproj_range_xyz_label[p2ri_lut[:, 0]]  # 'N'

        num_valid_points = torch.LongTensor([num_valid_points])

        return range_xyz_remission_img, range_xyz_remission_label.long(), \
               range_xyz_raw_data, \
               align_unproj_range_xyz_data, align_unproj_range_xyz_label.long(), \
               align_p2ri_lut.long(), \
               num_valid_points

    def __len__(self):
        return self.num

    def shuffle(self):
        self.permutation = np.random.permutation(self.num)


class SemanticKITTITestData(data.Dataset):
    def __init__(self, data_root, sequences, CFG):
        """
        :param data_root: 'Datasets/SemanticKitti/dataset/sequences'
        :param sequences: test_sequences = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
                          val_sequences = ['08']
        """
        super(SemanticKITTITestData, self).__init__()
        assert osp.exists(data_root), 'please provide the data_root'
        assert sequences is not None, 'please specify sequences.'

        means = CFG['dataset']['sensor']['img_means']
        std = CFG['dataset']['sensor']['img_stds']
        self.max_points = CFG['dataset']['max_points']  # 133376.

        # projected data
        self.range_images_filenames = []
        self.xyz_filenames = []
        self.remission_filenames = []
        self.range_image_2_points_mask_filenames = []
        self.points_2_range_image_lut_filenames = []
        # unprojected data
        self.unproj_points_filenames = []
        # other information
        self.sequences_indices = []
        self.filenames = []

        if osp.basename(data_root) == 'sequences_spherical_range_images_2048':
            unproj_points_root = data_root.replace('sequences_spherical_range_images_2048', 'sequences')
        else:  # sequences_laserid_range_images_2048
            unproj_points_root = data_root.replace('sequences_laserid_range_images_2048', 'sequences')

        for sequence in sequences:
            # projected data
            tmp_range_images_filenames = glob.glob(osp.join(data_root, sequence, 'range_image', '*.npy'))
            tmp_range_images_filenames.sort()
            self.range_images_filenames += tmp_range_images_filenames

            tmp_xyz_filenames = glob.glob(osp.join(data_root, sequence, 'range_image_2_points_matrix', '*.npy'))
            tmp_xyz_filenames.sort()
            self.xyz_filenames += tmp_xyz_filenames

            tmp_remission_filenames = glob.glob(osp.join(data_root, sequence, 'remissions', '*.npy'))
            tmp_remission_filenames.sort()
            self.remission_filenames += tmp_remission_filenames

            tmp_range_image_2_points_mask_filenames = glob.glob(osp.join(data_root, sequence, 'range_image_2_points_mask', '*.npy'))
            tmp_range_image_2_points_mask_filenames.sort()
            self.range_image_2_points_mask_filenames += tmp_range_image_2_points_mask_filenames

            tmp_points_2_range_image_lut_filenames = glob.glob(osp.join(data_root, sequence, 'points_2_range_image_lut', '*.npy'))
            tmp_points_2_range_image_lut_filenames.sort()
            self.points_2_range_image_lut_filenames += tmp_points_2_range_image_lut_filenames

            # unprojected data
            tmp_unproj_points_filenames = glob.glob(osp.join(unproj_points_root, sequence, 'velodyne', '*.bin'))
            tmp_unproj_points_filenames.sort()
            self.unproj_points_filenames += tmp_unproj_points_filenames

            for tmp_range_images_filename in tmp_range_images_filenames:
                self.sequences_indices.append(sequence)
                self.filenames.append(osp.basename(tmp_range_images_filename).split('.')[0])

        self.num = len(self.range_images_filenames)
        self.transform_function = Compose([ToTensor(means, std)])

    @staticmethod
    def _get_unproj_data(filename: str):
        """read points from the .bin file."""
        scan = np.fromfile(filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        num_points = scan.shape[0]

        unproj_points = scan[:, 0:3]    # get xyz, (N, 3)
        # unproj_remissions = scan[:, 3]  # get remission, (N,)

        # num_points = unproj_remissions.shape[0]
        # unproj_remissions = unproj_remissions.reshape(num_points, 1)  # (N, 1)

        unproj_ranges = np.linalg.norm(unproj_points, 2, axis=1)  # unproj_ranges
        unproj_ranges = unproj_ranges.reshape(num_points, 1)

        # unproj_data = np.concatenate([unproj_ranges, unproj_points, unproj_remissions], axis=-1)  # (N, 5)
        unproj_data = np.concatenate([unproj_ranges, unproj_points], axis=-1)  # (N, 4)
        return unproj_data

    def __getitem__(self, item):
        # data (augmentation).
        range_image = np.load(self.range_images_filenames[item])  # H x W
        H, W = range_image.shape
        range_image = range_image.reshape((H, W, 1))  # H x W x 1
        xyz = np.load(self.xyz_filenames[item])  # H x W x 3
        remission = np.load(self.remission_filenames[item])
        remission = remission.reshape((H, W, 1))  # H x W x 1
        data = np.concatenate([range_image, xyz, remission], axis=-1)  # H x W x 5

        # data (raw)
        range_xyz_raw_data = torch.from_numpy(np.concatenate([range_image, xyz], axis=-1))  # H x W x 4.
        range_xyz_raw_data = range_xyz_raw_data.permute(2, 0, 1).contiguous()  # 4 x H x W.

        # unprojected data
        unproj_range_xyz_data = self._get_unproj_data(self.unproj_points_filenames[item])  # (N, 4)
        unproj_range_xyz_data = torch.from_numpy(unproj_range_xyz_data)  # (N, 4)

        # other.
        mask = torch.from_numpy(np.load(self.range_image_2_points_mask_filenames[item]))  # H x W, (0 or 1)
        p2ri_lut = torch.from_numpy(np.load(self.points_2_range_image_lut_filenames[item]).astype(np.int32)).long() # N x 3
        sequences_index = self.sequences_indices[item]
        filename = self.filenames[item]

        # data augmentation.
        img_gt = dict(img=data, gt=None, p2ri_lut=None)
        img_gt = self.transform_function(img_gt)
        data = img_gt['img']
        range_xyz_remission_img = data * mask  # only keep valid values.

        # get the number of valid points.
        num_valid_points = p2ri_lut.shape[0]

        # align all data.
        align_p2ri_lut = torch.zeros(self.max_points, p2ri_lut.shape[1]).long()  # 'N' x 3.
        align_p2ri_lut[0:num_valid_points, :] = p2ri_lut

        align_unproj_range_xyz_data = torch.zeros(self.max_points, unproj_range_xyz_data.shape[1])
        align_unproj_range_xyz_data[p2ri_lut[:, 0], :] = unproj_range_xyz_data[p2ri_lut[:, 0], :]  # 'N' x 4

        num_valid_points = torch.LongTensor([num_valid_points])

        return range_xyz_remission_img, range_xyz_raw_data, align_unproj_range_xyz_data, align_p2ri_lut.long(), \
               num_valid_points, sequences_index, filename

    def __len__(self):
        return self.num


if __name__ == "__main__":
    pass

