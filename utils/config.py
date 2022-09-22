"""
    Function: load hyper-parameters.

    Date: May 16, 2022.
"""
import yaml
import os.path as osp


def load_cfg_from_cfg_file(file):
    """outputs are in the json format."""
    assert osp.isfile(file) and file.endswith('.yaml'), '{} is not a yaml file'.format(file)

    with open(file, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


if __name__ == "__main__":
    file = "../configs/laserid_cenet_64x2048.yaml"
    cfg = load_cfg_from_cfg_file(file=file)
    print(cfg)

