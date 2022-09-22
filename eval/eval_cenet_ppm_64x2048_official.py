"""
    Function: only compute IoU and mIoU scores.

    Note that we here use the official implementation of iouEval.

    labels (remapped): original labels -> 0-19
    prediction: original labels -> 0-19

    Date: May 19, 2022.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import os.path as osp
import glob
import tqdm
from utils.generate_semantickitti_lut import get_label_remap_lut_color_lut
# from utils.metrics import SegmentationMetric
from libs.semantic_kitti_api.auxiliary.np_ioueval import iouEval


def compute_scores(label_data_root, prediction_data_root, sequences, remap_lut):
    """ Compute IoU and mIoU scores. """
    print("Get all label filenames ...")
    label_filenames = []
    for sequence in sequences:
        tmp_label_filenames = glob.glob(osp.join(label_data_root, sequence, 'labels', '*.label'))
        tmp_label_filenames.sort()
        label_filenames += tmp_label_filenames

    print("Get all prediction filenames ...")
    prediction_filenames = []
    for sequence in sequences:
        tmp_prediction_filenames = glob.glob(osp.join(prediction_data_root, sequence, 'predictions', '*.label'))
        tmp_prediction_filenames.sort()
        prediction_filenames += tmp_prediction_filenames

    assert len(label_filenames) == len(prediction_filenames)

    print("Initialize a confusion matrix ...")
    metric = iouEval(n_classes=19+1, ignore=0)

    progressbar = tqdm.tqdm(range(len(label_filenames)))
    for label_filename, prediction_filename in zip(label_filenames, prediction_filenames):
        labels, _ = read_labels(label_filename)  # read labels
        labels = remap_lut[labels]  # remap labels to 0-19.

        predictions, _ = read_labels(prediction_filename)  # read predictions, original labels.
        predictions = remap_lut[predictions]  # remap labels to 0-19.
        metric.addBatch(predictions, labels)

        progressbar.update(1)
    progressbar.close()


    mIoU, IoU = metric.getIoU()
    print("*"*40, "IoU", "*"*40)
    np.set_printoptions(precision=3, suppress=True)
    print(IoU)
    np.set_printoptions(precision=1, suppress=True)
    print(IoU)
    print("*"*40, "mIoU", "*"*40)
    np.set_printoptions(precision=3, suppress=True)
    print(mIoU)
    np.set_printoptions(precision=1, suppress=True)
    print(mIoU)


def read_labels(filename: str):
    label = np.fromfile(filename, dtype=np.uint32)
    label = label.reshape((-1))
    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16    # instance id in upper half
    assert((sem_label + (inst_label << 16) == label).all())
    return sem_label, inst_label


if __name__ == "__main__":
    label_data_root = 'Datasets/SemanticKitti/dataset/sequences'
    prediction_data_root = 'Datasets/SemanticKitti/dataset/predictions_sequences_laserid_range_images_2048/sequences'
    train_sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
    val_sequences = ['08']
    test_sequences = ['11',	'12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
    data_config = 'libs/semantic_kitti_api/config/semantic-kitti.yaml'
    remap_lut, _, _ = get_label_remap_lut_color_lut(data_config=data_config)
    # on the validation dataset.
    compute_scores(label_data_root=label_data_root,
                   prediction_data_root=prediction_data_root,
                   sequences=val_sequences,
                   remap_lut=remap_lut)
    # on the training dataset.
    # compute_scores(label_data_root=label_data_root,
    #                prediction_data_root=prediction_data_root,
    #                sequences=train_sequences,
    #                remap_lut=remap_lut)
    print("Successfully !")

