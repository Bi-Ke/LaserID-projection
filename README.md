# LaserID-projection
Official implementation of our work, "Effective Projection of 3D Points onto 2D Range Images for Point Cloud Segmentation".

### Abstract

Point cloud segmentation (PCS) aims to assign each point a semantic label, and hence helps autonomous driving cars and robots understand their surroundings as accurately as possible. In real-world applications, the range image representation of LiDAR data is commonly utilized to meet the efficient requirement of large-scale PCS. To generate range images, however, many existing methods directly adopt Spherical projection, ignoring the inherent attributes of a LiDAR sensor and then resulting in the loss of information. To stored raw points in produced range images as many as possible, in this paper, we propose the LaserID projection approach, which maps 3D points onto range images totally based on the properties of the LiDAR sensor. In addition, there still exists information loss in projection. To compensate for this, we propose an effective post-processing module (PPM) to instead of the usually used KNN method. The proposed PPM can be easily integrated with existing range-image-based architectures and trained in the end-to-end manner. Experimental results on SemanticKITTI demonstrate the effectiveness of the proposed LaserID projection and PPM. 

### Examples

![range_images_video](images/range_images_video.gif)

Left: **Spherical Projection**; Right: **LaserID Projection**

![ceppmnet_val](images/ceppmnet_val.gif)

CEPPMNet on the SemanticKITTI validation dataset.

![ceppmnet_test](images/ceppmnet_test.gif)

CEPPMNet on the SemanticKITTI test dataset.

### How to use the code

Download SemanticKITTI from the [official website](http://www.semantic-kitti.org/dataset.html). Then please revise the configuration file in the`configs/laserid_cenet_ppm_64x2048.yaml`.

There are two methods to train and test the model. One is that we could use the tool `tools/save_range_images_other_info_lidar.py` to generate range images and other files in advance. Then we could adopt `data/semantic_kitti_range_image_data_constructor.py` to read data as inputs. The other is that we could directly utilize `data/semantic_kitti_range_image_data_constructorV1.py` to read raw point cloud to generate range images and other files as inputs. 

#### Train

`sh train.sh`

#### Test

`sh infer.sh`

`python eval/eval_cenet_ppm_64x2048_official.py`

### Pretrained Model

The pretrained model is put in the `checkpoints/save_best_val_model.pt`

### Acknowledgements

Parts of code are derived from [CENet](https://github.com/huixiancheng/CENet) and [RangeNet++](https://github.com/PRBonn/lidar-bonnetal). Many thanks for their source code.

### Citation

I am preparing this ...

