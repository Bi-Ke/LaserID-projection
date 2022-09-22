"""
    Function: generate
        remap_lut: Remapped label values are from 0 to 19, where 0 indicates the ignored label.
        remap_lut_inv: 0-19 to original labels.
        color_lut: provide the mapping between labels and colors (i.e., in BGR format).
"""
import numpy as np
import yaml


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


if __name__ == "__main__":
    pass
