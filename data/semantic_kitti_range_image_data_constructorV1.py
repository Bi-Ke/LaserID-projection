"""
    Function: data constructor for the distributed training on the SemanticKITTI dataset (i.e., range images).

    Date: August 30, 2022.
"""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import glob
from torch.utils import data
import os.path as osp
from utils.transformations import RandomDrop, RandomHorizontalFlip, RandomNoiseXYZ, RandomRotate, ToTensor, Compose
from utils.generate_semantickitti_lut import get_label_remap_lut_color_lut


def read_points(filename: str):
    """read points from the .bin file."""
    scan = np.fromfile(filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    points = scan[:, 0:3]    # get xyz, (N, 3)
    remissions = scan[:, 3]  # get remission, (N, 1)
    return points, remissions


def radian2degree(radian: np.array):
    """convert radian values to corresponding degrees."""
    degree = radian * 180 / np.pi
    return degree


def find_best_AZIMUTH_GAP_THRESHOLD(azimuth_gaps, biggest_azimuth_gaps_positions, AZIMUTH_GAP_THRESHOLD=300.0):
    """find the best azimuth gap threshold. The smallest scale is 0.0001"""
    scales = [1.0, 0.1, 0.01, 0.001, 0.0001]
    while len(biggest_azimuth_gaps_positions)+1 < 64:
        AZIMUTH_GAP_THRESHOLD -= scales[0]
        biggest_azimuth_gaps_positions = np.where(azimuth_gaps >= AZIMUTH_GAP_THRESHOLD)[0]
        if len(biggest_azimuth_gaps_positions)+1 == 64:
            return biggest_azimuth_gaps_positions, AZIMUTH_GAP_THRESHOLD

    while len(biggest_azimuth_gaps_positions)+1 > 64:
        AZIMUTH_GAP_THRESHOLD += scales[1]
        biggest_azimuth_gaps_positions = np.where(azimuth_gaps >= AZIMUTH_GAP_THRESHOLD)[0]
        if len(biggest_azimuth_gaps_positions)+1 == 64:
            return biggest_azimuth_gaps_positions, AZIMUTH_GAP_THRESHOLD

    while len(biggest_azimuth_gaps_positions)+1 < 64:
        AZIMUTH_GAP_THRESHOLD -= scales[2]
        biggest_azimuth_gaps_positions = np.where(azimuth_gaps >= AZIMUTH_GAP_THRESHOLD)[0]
        if len(biggest_azimuth_gaps_positions)+1 == 64:
            return biggest_azimuth_gaps_positions, AZIMUTH_GAP_THRESHOLD

    while len(biggest_azimuth_gaps_positions)+1 > 64:
        AZIMUTH_GAP_THRESHOLD += scales[3]
        biggest_azimuth_gaps_positions = np.where(azimuth_gaps >= AZIMUTH_GAP_THRESHOLD)[0]
        if len(biggest_azimuth_gaps_positions)+1 == 64:
            return biggest_azimuth_gaps_positions, AZIMUTH_GAP_THRESHOLD

    while len(biggest_azimuth_gaps_positions)+1 < 64:
        AZIMUTH_GAP_THRESHOLD -= scales[4]
        biggest_azimuth_gaps_positions = np.where(azimuth_gaps >= AZIMUTH_GAP_THRESHOLD)[0]
        if len(biggest_azimuth_gaps_positions)+1 == 64:
            return biggest_azimuth_gaps_positions, AZIMUTH_GAP_THRESHOLD
    return biggest_azimuth_gaps_positions, AZIMUTH_GAP_THRESHOLD


def get_each_row_points(points: np.array, remissions: np.array):
    """ Theoretically, there are 64 rows because of 64 lasers for Velodyne HDL-64E. One revolution is from 0 to 360
    degrees. Start azimuth is the 0 degree and the end azimuth is the 360 degrees.
    1) The gap between the start azimuth and the end azimuth is more than 300 degrees (i.e., theoretically 360 degrees).
    This can be used to find all start azimuth and end azimuth.
    2) For each revolution, each laser captures at least 1000 points (i.e., theoretically 2083). This can be used to
    filter some outliers.
    Here we cluster points for each row. """
    AZIMUTH_GAP_THRESHOLD = 300

    azimuth = radian2degree(np.arctan2(points[:, 1], points[:, 0]))
    mask = azimuth < 0
    azimuth[mask] = 360 + azimuth[mask]

    azimuth_rotate = radian2degree(-np.arctan2(points[:, 1], points[:, 0])) + 180  # rotate range images.

    # find the 64 biggest azimuth gaps between data.
    azimuth_gaps = np.abs(azimuth[1:] - azimuth[:-1])
    biggest_azimuth_gaps_positions = np.where(azimuth_gaps >= AZIMUTH_GAP_THRESHOLD)[0]

    if len(biggest_azimuth_gaps_positions)+1 != 64:
        biggest_azimuth_gaps_positions, AZIMUTH_GAP_THRESHOLD = \
            find_best_AZIMUTH_GAP_THRESHOLD(azimuth_gaps, biggest_azimuth_gaps_positions, AZIMUTH_GAP_THRESHOLD)

    assert len(biggest_azimuth_gaps_positions)+1 == 64, "should be 64 lasers. Now = {}". \
        format(len(biggest_azimuth_gaps_positions)+1)

    num_rows = 0
    each_row_start_end_positions = []
    for idx in range(len(biggest_azimuth_gaps_positions)):
        if idx == 0:
            each_row_start_end_positions.append([0, biggest_azimuth_gaps_positions[idx]+1])
            num_rows += 1
        else:
            each_row_start_end_positions.append([biggest_azimuth_gaps_positions[idx-1]+1,
                                                 biggest_azimuth_gaps_positions[idx]+1])
            num_rows += 1
    each_row_start_end_positions.append([biggest_azimuth_gaps_positions[-1]+1, len(azimuth)])
    num_rows += 1
    assert num_rows == 64, "At least 64 lines."
    assert len(each_row_start_end_positions) == 64, "At least 64 intervals."
    points_range = np.linalg.norm(points, 2, axis=1)
    points_z_coords = points[:, 2]
    elevation = radian2degree(np.arcsin(points_z_coords/points_range))

    # reorganize all points.
    each_row_points = []
    each_row_points_range = []
    each_row_points_remissions = []
    each_row_points_azimuth = []
    each_row_points_elevation = []
    each_row_points_index = []
    for idx, start_end_pos in enumerate(each_row_start_end_positions):
        each_row_points.append(points[start_end_pos[0]:start_end_pos[1]])
        each_row_points_range.append(points_range[start_end_pos[0]:start_end_pos[1]])
        each_row_points_remissions.append(remissions[start_end_pos[0]:start_end_pos[1]])
        # each_row_points_azimuth.append(azimuth[start_end_pos[0]:start_end_pos[1]])
        each_row_points_azimuth.append(azimuth_rotate[start_end_pos[0]:start_end_pos[1]])
        each_row_points_elevation.append(elevation[start_end_pos[0]:start_end_pos[1]])
        each_row_points_index.append(np.arange(start_end_pos[0], start_end_pos[1]))
    return each_row_points, each_row_points_range, each_row_points_remissions, each_row_points_azimuth, \
           each_row_points_elevation, each_row_points_index


def generate_range_image_LUT(each_row_points, each_row_points_range, each_row_points_remissions,
                             each_row_points_azimuth, each_row_points_elevation, each_row_points_index,
                             img_resolution):
    """generate range images and corresponding Look Up Table.

    Velodyne HDL-64E S2 information:
    ----------------------------------------------------------------------------------------
    RPM | RPS  | Points Per Laser | Angular Resolution | Vertical Field of View
        | (Hz) |                  | (degree)           | (degree)
    ----------------------------------------------------------------------------------------
    600 | 10   | 2083             | 0.1728             | +2 -- -8.33: 1/3 degree spacing.
        |      |                  |                    |-8.83 -- -24.33: 1/2 degree spacing.
    -----------------------------------------------------------------------------------------
    According to the basic LiDAR information, we can know the size of a range image should be (64, 2084), i.e.,
    np.ceil(360 / 0.1728) = 2084.

    :param each_row_points: [[[x1, y1, z1], [x2, y2, z2], ...], ...]
    :param each_row_points_range: [[ra1, ra2, ...], ...]
    :param each_row_points_remissions: [[re1, re2, ...], ...]
    :param each_row_points_azimuth: [[a1, a2, ...], ...]
    :param each_row_points_elevation: [[e1, e2, ...], ...]
    :param each_row_points_index: [[idx1, idx2, ...], ...]
    :param img_resolution: {'width':2084, 'height':64}

    :return
    range_image: shape (64, 2084), each pixel value in the range image is the range value of a point.
    remissions: shape (64, 2048), each pixel value is a signal of a point.
    range_image_2_points_matrix: shape (64, 2084, 3), each pixel means a point with the (x, y, z) coordinates.
    range_image_2_points_mask: shape (64, 2084), each pixel indicates whether there is a point.
                               0 indicates no point, and 1 indicates having a point. If all points are projected onto
                               a unique position in the range image,
                               range_image_2_points_matrix[range_image_2_points_mask].reshape(-1, 3) should be all points.
    points_2_range_image_lut: shape (N, 3), values in each row are in the format of (index, u, v), where index means the
                              index of a point, and u, v mean the corresponding position in the range image.
    """
    # NUM_LASERS = 64
    POINTS_PER_LASER = img_resolution['width']
    ANGULAR_RESOLUTION = 0.17280043

    # checking the order of each laser.
    """
    mean_elevation = []
    for el in each_row_points_elevation:
        mean_elevation.append(np.mean(el))
    mean_elevation = np.array(mean_elevation)
    mean_gap_elevation = mean_elevation[:-1] - mean_elevation[1:]
    assert np.sum(mean_gap_elevation < 0) == 0, "lasers are unordered {}: {}!".\
        format(mean_gap_elevation, mean_elevation)
    """

    # filling data.
    num_points = each_row_points_index[-1][-1]+1
    range_image = np.zeros((64, POINTS_PER_LASER), dtype=np.float32)  # range value of a point.
    remissions = np.zeros((64, POINTS_PER_LASER), dtype=np.float32)  # signal of a point.
    range_image_2_points_matrix = np.zeros((64, POINTS_PER_LASER, 3), dtype=np.float32)  # (x, y, z)
    range_image_2_points_mask = np.zeros((64, POINTS_PER_LASER), dtype=np.uint8)  # 0 or 1
    points_2_range_image_lut = np.zeros((num_points, 3), dtype=np.uint32)  # (index, u, v)

    # missing_points = []
    for u_coord in range(64):
        # v_coords = np.floor(each_row_points_azimuth[u_coord] / ANGULAR_RESOLUTION).astype(np.int16)
        # Paper: Scan-based Semantic Segmentation of LiDAR Point Clouds: A Experimental Study.
        v_coords = np.floor(each_row_points_azimuth[u_coord] / 360 * (POINTS_PER_LASER-1)).astype(np.int16)
        range_image[u_coord, v_coords] = each_row_points_range[u_coord]
        remissions[u_coord, v_coords] = each_row_points_remissions[u_coord]
        # if len(np.unique(v_coords)) != len(each_row_points_azimuth[u_coord]):
        #     print("The number of the duplicate data is {}".format(len(each_row_points_azimuth[u_coord]) -
        #                                                           len(np.unique(v_coords))))
        #     missing_points.append(len(each_row_points_azimuth[u_coord]) - len(np.unique(v_coords)))
        range_image_2_points_matrix[u_coord, v_coords] = each_row_points[u_coord]
        range_image_2_points_mask[u_coord, v_coords] = 1
        # print(len(each_row_points_azimuth[u_coord]) - np.sum(range_image_2_points_mask[u_coord]))
        points_2_range_image_lut[each_row_points_index[u_coord], 0] = each_row_points_index[u_coord]
        points_2_range_image_lut[each_row_points_index[u_coord], 1] = np.repeat(u_coord, len(v_coords))
        points_2_range_image_lut[each_row_points_index[u_coord], 2] = v_coords
        # print(points_2_range_image_lut[0:10, :])
    # print(np.sum(missing_points))
    return range_image, remissions, range_image_2_points_matrix, range_image_2_points_mask, points_2_range_image_lut


def read_labels(filename: str):
    """read labels from the .label file."""
    label = np.fromfile(filename, dtype=np.uint32)
    label = label.reshape((-1))
    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16    # instance id in upper half
    assert((sem_label + (inst_label << 16) == label).all())
    return sem_label


def generate_range_image_gt(semantic_labels, remap_lut, remap_lut_inv, color_lut, points_2_range_image_lut, img_resolution):
    """
    :param semantic_labels: raw labels.
    :param remap_lut: remapping labels to 0-19, where 0 indicates the ignored label.
    :param remap_lut_inv: 0-19 to original labels.
    :param color_lut: raw labels to colors.
    :param points_2_range_image_lut: (index, u, v) used to map point indexes to the corresponding image positions.
    :param img_resolution: {'width': 2084, 'height': 64}

    :return:
    range_image_gt: shape (64, 2084), corresponding labels.
    range_image_color: shape (64, 2084, 3), each pixel indicates R, G, and B values.
    """
    POINTS_PER_LASER = img_resolution['width']

    range_image_gt = np.zeros((64, POINTS_PER_LASER), dtype=np.uint8)
    range_image_color = np.zeros((64, POINTS_PER_LASER, 3), dtype=np.uint8)

    # label mapping.
    semantic_labels = remap_lut[semantic_labels]  # to 0-19.
    semantic_labels = semantic_labels.astype(np.uint8)

    # filling data.
    u_coords = points_2_range_image_lut[:, 1]
    v_coords = points_2_range_image_lut[:, 2]
    range_image_gt[u_coords, v_coords] = semantic_labels

    raw_labels = remap_lut_inv[semantic_labels]
    range_image_color[u_coords, v_coords] = color_lut[raw_labels]

    # cv2.imwrite("range_image_gt.png", range_image_gt)
    # cv2.imwrite("range_image_color.png", range_image_color)
    # cv2.imshow("range image", range_image_color)
    # cv2.waitKey(0)
    return range_image_gt, range_image_color


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

        self.img_resolution = CFG['dataset']['sensor']['img_prop']
        means = CFG['dataset']['sensor']['img_means']
        std = CFG['dataset']['sensor']['img_stds']
        self.max_points = CFG['dataset']['max_points']  # 133376.
        self.remap_lut, self.remap_lut_inv, self.color_lut = get_label_remap_lut_color_lut(CFG['dataset']['data_config'])

        # unprojected data
        self.unproj_points_filenames = []
        self.unproj_labels_filenames = []

        for sequence in sequences:
            # unprojected data
            tmp_unproj_points_filenames = glob.glob(osp.join(data_root, sequence, 'velodyne', '*.bin'))
            tmp_unproj_points_filenames.sort()
            self.unproj_points_filenames += tmp_unproj_points_filenames

            tmp_unproj_labels_filenames = glob.glob(osp.join(data_root, sequence, 'labels', '*.label'))
            tmp_unproj_labels_filenames.sort()
            self.unproj_labels_filenames += tmp_unproj_labels_filenames

        assert len(self.unproj_points_filenames) == len(self.unproj_labels_filenames), 'must be the same.'

        self.num = len(self.unproj_points_filenames)
        self.permutation = np.random.permutation(self.num)

        if is_train:
            self.transform_function = Compose([RandomDrop(p=0.5),  # 64 x 2084
                                               RandomHorizontalFlip(p=0.25),
                                               RandomNoiseXYZ(p=0.25),
                                               RandomRotate(p=0.25),
                                               ToTensor(means, std)])
        else:
            self.transform_function = Compose([ToTensor(means, std)])

    def __getitem__(self, item):
        # data (augmentation)
        points, remissions = read_points(self.unproj_points_filenames[self.permutation[item]])
        each_row_points, \
        each_row_points_range, \
        each_row_points_remissions, \
        each_row_points_azimuth, \
        each_row_points_elevation, \
        each_row_points_index = get_each_row_points(points, remissions)

        range_image, \
        remission_image, \
        range_image_2_points_matrix, \
        range_image_2_points_mask, \
        points_2_range_image_lut = generate_range_image_LUT(each_row_points,
                                                            each_row_points_range,
                                                            each_row_points_remissions,
                                                            each_row_points_azimuth,
                                                            each_row_points_elevation,
                                                            each_row_points_index,
                                                            self.img_resolution)
        sem_labels = read_labels(self.unproj_labels_filenames[self.permutation[item]])
        range_image_gt, \
        range_image_color = generate_range_image_gt(semantic_labels=sem_labels,
                                                    remap_lut=self.remap_lut,
                                                    remap_lut_inv=self.remap_lut_inv,
                                                    color_lut=self.color_lut,
                                                    points_2_range_image_lut=points_2_range_image_lut,
                                                    img_resolution=self.img_resolution)

        H, W = range_image.shape
        range_image = range_image.reshape((H, W, 1))  # H x W x 1
        xyz = range_image_2_points_matrix  # H x W x 3
        remission = remission_image
        remission = remission.reshape((H, W, 1))  # H x W x 1
        data = np.concatenate([range_image, xyz, remission], axis=-1)  # H x W x 5

        # label
        label = range_image_gt  # H x W
        label = label - 1  # 0~19 -> -1~18
        label[label == -1] = 255  # 0~18, 255 (ignore)

        # data (raw)
        range_xyz_raw_data = torch.from_numpy(np.concatenate([range_image, xyz], axis=-1))  # H x W x 4.
        range_xyz_raw_data = range_xyz_raw_data.permute(2, 0, 1).contiguous()  # 4 x H x W.

        # unprojected data
        num_points = points.shape[0]
        unproj_ranges = np.linalg.norm(points, 2, axis=1)  # unproj_ranges
        unproj_ranges = unproj_ranges.reshape(num_points, 1)
        unproj_range_xyz_data = np.concatenate([unproj_ranges, points], axis=-1)  # (N, 4)
        unproj_range_xyz_data = torch.from_numpy(unproj_range_xyz_data)  # (N, 4)

        # unprojected label
        sem_labels = self.remap_lut[sem_labels]  # to 0-19.
        sem_labels = sem_labels.astype(np.uint8)
        sem_labels = sem_labels - 1
        sem_labels[sem_labels == -1] = 255  # 0~18, 255.
        unproj_range_xyz_label = sem_labels
        unproj_range_xyz_label = torch.from_numpy(unproj_range_xyz_label).long()

        # other
        mask = range_image_2_points_mask  # H x W, (0 or 1)
        p2ri_lut = points_2_range_image_lut  # N x 3.

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

        self.img_resolution = CFG['dataset']['sensor']['img_prop']
        means = CFG['dataset']['sensor']['img_means']
        std = CFG['dataset']['sensor']['img_stds']
        self.max_points = CFG['dataset']['max_points']  # 133376.

        self.remap_lut, self.remap_lut_inv, self.color_lut = get_label_remap_lut_color_lut(CFG['dataset']['data_config'])

        # unprojected data
        self.unproj_points_filenames = []
        # other information
        self.sequences_indices = []
        self.filenames = []

        for sequence in sequences:
            # unprojected data
            tmp_unproj_points_filenames = glob.glob(osp.join(data_root, sequence, 'velodyne', '*.bin'))
            tmp_unproj_points_filenames.sort()
            self.unproj_points_filenames += tmp_unproj_points_filenames
            # other information
            for tmp_range_images_filename in tmp_unproj_points_filenames:
                self.sequences_indices.append(sequence)
                self.filenames.append(osp.basename(tmp_range_images_filename).split('.')[0])

        self.num = len(self.unproj_points_filenames)
        self.transform_function = Compose([ToTensor(means, std)])

    def __getitem__(self, item):
        # data (augmentation).
        points, remissions = read_points(self.unproj_points_filenames[item])
        each_row_points, \
        each_row_points_range, \
        each_row_points_remissions, \
        each_row_points_azimuth, \
        each_row_points_elevation, \
        each_row_points_index = get_each_row_points(points, remissions)

        range_image, \
        remission_image, \
        range_image_2_points_matrix, \
        range_image_2_points_mask, \
        points_2_range_image_lut = generate_range_image_LUT(each_row_points,
                                                            each_row_points_range,
                                                            each_row_points_remissions,
                                                            each_row_points_azimuth,
                                                            each_row_points_elevation,
                                                            each_row_points_index,
                                                            self.img_resolution)

        H, W = range_image.shape
        range_image = range_image.reshape((H, W, 1))  # H x W x 1
        xyz = range_image_2_points_matrix  # H x W x 3
        remission = remission_image
        remission = remission.reshape((H, W, 1))  # H x W x 1
        data = np.concatenate([range_image, xyz, remission], axis=-1)  # H x W x 5

        # data (raw)
        range_xyz_raw_data = torch.from_numpy(np.concatenate([range_image, xyz], axis=-1))  # H x W x 4.
        range_xyz_raw_data = range_xyz_raw_data.permute(2, 0, 1).contiguous()  # 4 x H x W.

        # unprojected data
        num_points = points.shape[0]
        unproj_ranges = np.linalg.norm(points, 2, axis=1)  # unproj_ranges
        unproj_ranges = unproj_ranges.reshape(num_points, 1)
        unproj_range_xyz_data = np.concatenate([unproj_ranges, points], axis=-1)  # (N, 4)
        unproj_range_xyz_data = torch.from_numpy(unproj_range_xyz_data)  # (N, 4)

        # other.
        mask = torch.from_numpy(range_image_2_points_mask)  # H x W, (0 or 1)
        p2ri_lut = torch.from_numpy(points_2_range_image_lut.astype(np.int32)).long()  # N x 3
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
    import torch.utils.data as Data
    from utils.config import load_cfg_from_cfg_file
    cfg = load_cfg_from_cfg_file('configs/laserid_cenet_ppm_64x2048.yaml')
    data_root = 'Datasets/SemanticKitti/dataset/sequences'
    train_sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10', '08']

    train_dataset = SemanticKITTITrainData(data_root=data_root, sequences=train_sequences, CFG=cfg, is_train=True)
    train_loader = Data.DataLoader(dataset=train_dataset,
                                   batch_size=2,  # for single GPU.
                                   shuffle=False,
                                   pin_memory=True,
                                   num_workers=8)

    for train_img, train_gt, range_xyz_raw_data, align_unproj_range_xyz_data, align_unproj_range_xyz_label, \
        align_p2ri_lut, num_valid_points in train_loader:
        print("train_img.shape: ", train_img.shape)
        print("train_gt.shape: ", train_gt.shape)
        print("range_xyz_raw_data.shape: ", range_xyz_raw_data.shape)
        print("align_unproj_range_xyz_data.shape: ", align_unproj_range_xyz_data.shape)
        print("align_unproj_range_xyz_label.shape: ", align_unproj_range_xyz_label.shape)
        print("align_p2ri_lut.shape: ", align_p2ri_lut.shape)
        print("num_valid_points.shape: ", num_valid_points.shape)
        print(num_valid_points)
        # train_img.shape:  torch.Size([2, 5, 64, 2048])
        # train_gt.shape:  torch.Size([2, 64, 2048])
        # range_xyz_raw_data.shape:  torch.Size([2, 4, 64, 2048])
        # align_unproj_range_xyz_data.shape:  torch.Size([2, 133376, 4])
        # align_unproj_range_xyz_label.shape:  torch.Size([2, 133376])
        # align_p2ri_lut.shape:  torch.Size([2, 133376, 3])
        #
        # num_valid_points.shape:  torch.Size([2, 1])
        # tensor([[128019],
        #         [102331]])

