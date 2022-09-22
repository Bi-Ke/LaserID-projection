"""
    Function: saving range images and other information.

        'range_image': H x W (True)
        'remissions': H x W x 1 (True)
        'range_image_2_points_matrix': H x W x 3, xyz coordinates in the range image (True)
        'range_image_2_points_mask': H x W x 1 (True)
        'range_image_2_points_indices': H x W x 1 (False)
        'points_2_range_image_lut': N x 3 (True)
        'range_image_gt': H x W x 1 (True)
        'range_image_color': H x W x 3 (True)

    Date: May 3, 2022.
    Updated: June 20, 2022.
"""

import sys
sys.path.insert(0, '.')
import glob
import os
import os.path as osp
import numpy as np
import cv2
import tqdm
from tools.generate_range_image_kitti_lidar import *


def save_range_images_other_info(img_resolution, data_config, sequences, data_root, save_root, train_test='train'):
    """save range images and other information in the .npy format."""
    remap_lut, remap_lut_inv, color_lut = get_label_remap_lut_color_lut(data_config)

    for sequence in sequences:
        print("----------------- Processing the sequence {} --------------------------".format(sequence))
        data_dir = osp.join(data_root, sequence, "velodyne", "*")
        data_filenames = glob.glob(data_dir)
        data_filenames.sort()

        range_image_out = osp.join(save_root, sequence, "range_image")
        remissions_out = osp.join(save_root, sequence, "remissions")
        range_image_2_points_matrix_out = osp.join(save_root, sequence, "range_image_2_points_matrix")
        range_image_2_points_mask_out = osp.join(save_root, sequence, "range_image_2_points_mask")
        points_2_range_image_lut_out = osp.join(save_root, sequence, "points_2_range_image_lut")
        os.makedirs(range_image_out) if not osp.exists(range_image_out) else None
        os.makedirs(remissions_out) if not osp.exists(remissions_out) else None
        os.makedirs(range_image_2_points_matrix_out) if not osp.exists(range_image_2_points_matrix_out) else None
        os.makedirs(range_image_2_points_mask_out) if not osp.exists(range_image_2_points_mask_out) else None
        os.makedirs(points_2_range_image_lut_out) if not osp.exists(points_2_range_image_lut_out) else None

        if train_test == 'train':
            label_dir = osp.join(data_root, sequence, "labels", "*")
            label_filenames = glob.glob(label_dir)
            label_filenames.sort()
            range_image_gt_out = osp.join(save_root, sequence, "range_image_gt")
            range_image_color_out = osp.join(save_root, sequence, "range_image_color")
            os.makedirs(range_image_gt_out) if not osp.exists(range_image_gt_out) else None
            os.makedirs(range_image_color_out) if not osp.exists(range_image_color_out) else None

        progressbar = tqdm.tqdm(data_filenames)
        for idx, data_filename in enumerate(data_filenames):
            name = osp.basename(data_filename).split('.')[0]

            points, remissions = read_points(data_filename)
            each_row_points, \
            each_row_points_range, \
            each_row_points_remissions, \
            each_row_points_azimuth, \
            each_row_points_elevation, \
            each_row_points_index = get_each_row_points(points, remissions)

            range_image, \
            remissions, \
            range_image_2_points_matrix, \
            range_image_2_points_mask, \
            points_2_range_image_lut = generate_range_image_LUT(each_row_points,
                                                                each_row_points_range,
                                                                each_row_points_remissions,
                                                                each_row_points_azimuth,
                                                                each_row_points_elevation,
                                                                each_row_points_index,
                                                                img_resolution)

            if train_test == 'train':
                assert name == osp.basename(label_filenames[idx]).split('.')[0]
                range_image_gt, \
                range_image_color = generate_range_image_gt(label_path=label_filenames[idx],
                                                            remap_lut=remap_lut,
                                                            remap_lut_inv=remap_lut_inv,
                                                            color_lut=color_lut,
                                                            points_2_range_image_lut=points_2_range_image_lut,
                                                            img_resolution=img_resolution)

            np.save(osp.join(range_image_out, name), range_image)
            np.save(osp.join(remissions_out, name), remissions)
            np.save(osp.join(range_image_2_points_matrix_out, name), range_image_2_points_matrix)
            np.save(osp.join(range_image_2_points_mask_out, name), range_image_2_points_mask)
            np.save(osp.join(points_2_range_image_lut_out, name), points_2_range_image_lut)

            if train_test == 'train':
                cv2.imwrite(osp.join(range_image_gt_out, name+".png"), range_image_gt)
                cv2.imwrite(osp.join(range_image_color_out, name+".png"), range_image_color)
                # cv2.imshow("range image", range_image_color)
                # cv2.waitKey(1000)
            progressbar.update(1)
        progressbar.close()


if __name__ == "__main__":
    img_resolution = {'width': 512, 'height': 64}
    train_sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    test_sequences = ['11',	'12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
    data_config = "libs/semantic_kitti_api/config/semantic-kitti.yaml"
    data_root = "Datasets/SemanticKitti/dataset/sequences"
    save_root = "Datasets/SemanticKitti/dataset/sequences_laserid_range_images_{}".format(img_resolution['width'])
    save_range_images_other_info(img_resolution=img_resolution,
                                 data_config=data_config,
                                 sequences=train_sequences,
                                 data_root=data_root,
                                 save_root=save_root,
                                 train_test='train')
    save_range_images_other_info(img_resolution=img_resolution,
                                 data_config=data_config,
                                 data_root=data_root,
                                 save_root=save_root,
                                 sequences=test_sequences,
                                 train_test='test')


