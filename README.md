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





### Pretrained Model





### Acknowledgements

