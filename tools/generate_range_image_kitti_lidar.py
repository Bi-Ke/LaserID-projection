"""
    Function: generating range images of the SemanticKITTI dataset.

    Date: April 29, 2022.
    Update: June 20, 2022.
"""

import sys
sys.path.insert(0, '.')
import yaml
import os
import os.path as osp
import numpy as np
import glob
import cv2


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

    assert len(biggest_azimuth_gaps_positions)+1 == 64, "should be 64 lasers. Now = {}".\
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
    # checking points near start and end positions.
    # for idx, pos in enumerate(each_row_start_end_positions):
    #     print("------------------- {} ----------------".format(idx+1))
    #     if idx == 0:
    #         print(azimuth[0:6])
    #     elif idx == len(each_row_start_end_positions)-1:
    #         print(azimuth[-6:])
    #     else:
    #         print(azimuth[pos[0]-3:pos[0]+3])
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


def get_label_remap_lut_color_lut(data_config='semantic-kitti.yaml'):
    """(1) generate a look up table of remapping semantic labels. (2) make a dictionary supporting converting from
    labels to corresponding colors. Note that the labels should be remapped labels, i.e., 0~19.
    :return
    remap_lut: classes that are indistinguishable from single scan or inconsistent in ground truth are mapped to their
               closest equivalent. Remapped label values are from 0 to 19, where 0 indicates the ignored label.
    remap_lut_inv: 0-19 to original labels.
    color_dict: provide the mapping between labels and colors (i.e., in BGR format).
    """
    DATA = yaml.safe_load(open(data_config, 'r'))

    # transform original labels to 0-19.
    remap_dict = DATA["learning_map"]
    max_key = max(remap_dict.keys())
    remap_lut = np.zeros((max_key + 100), dtype=np.int32)
    remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

    # 0-19 to original labels.
    remap_dict_inv = DATA["learning_map_inv"]
    max_key = max(remap_dict_inv.keys())
    remap_lut_inv = np.zeros((max_key + 100), dtype=np.int32)
    remap_lut_inv[list(remap_dict_inv.keys())] = list(remap_dict_inv.values())

    # original labels to colors.
    color_dict = DATA["color_map"]
    max_key = max(color_dict.keys())
    color_lut = np.zeros((max_key + 100, 3), dtype=np.uint8)
    color_lut[list(color_dict.keys())] = list(color_dict.values())
    return remap_lut, remap_lut_inv, color_lut


def generate_range_image_gt(label_path, remap_lut, remap_lut_inv, color_lut, points_2_range_image_lut, img_resolution):
    """
    :param label_path: str
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

    # load semantic segmentation labels.
    labels = np.fromfile(label_path, dtype=np.uint32)
    labels = labels.reshape((-1))
    if labels.shape[0] == points_2_range_image_lut.shape[0]:
        semantic_labels = labels & 0xFFFF  # semantic label in lower half
        instance_labels = labels >> 16    # instance id in upper half
    else:
        print("Points shape: ", points_2_range_image_lut.shape)
        print("Label shape: ", labels.shape)
        raise ValueError("Scan and Label don't contain same number of points")
    assert((semantic_labels + (instance_labels << 16) == labels).all())  # sanity check

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


if __name__ == "__main__":
    pass

