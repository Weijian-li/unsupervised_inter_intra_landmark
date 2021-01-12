# Unsupervised Landmark Learning with Inter-Intra Subject Consistencies
[[paper]](https://arxiv.org/pdf/2004.07936.pdf)

Weijian Li, Haofu Liao, Shun Miao, Le Lu, and Jiebo Luo

IAPR International Conference on Pattern Recognition (ICPR), 2020, Oral Presentation

### Introduction
We propose an unsupervised learning approach to image landmark discovery by incorporating the inter-subject landmark consistencies on facial images.
<!-- ![alt text](http://cs.rochester.edu/u/wli69/images/projects/ICPR-20.png
"Framework") -->
<p align="center">
  <img src="http://cs.rochester.edu/u/wli69/images/projects/ICPR-20.png" width="75%"/>
</p>


The proposed core model block can be found [here](https://github.com/Weijian-li/unsupervised_inter_intra_landmark/blob/main/model.py#L184-L202)

### - Reference 
If you find our paper and repo useful, please cite our paper. Thanks!
``` 
@article{li2020unsupervised,
  title={Unsupervised Learning of Landmarks based on Inter-Intra Subject Consistencies},
  author={Li, Weijian and Liao, Haofu and Miao, Shun and Lu, Le and Luo, Jiebo},
  journal={arXiv preprint arXiv:2004.07936},
  year={2020}
}
```

### Prerequisites

* Python 3.6
* Pytorch 1.4

### Preparation

* CelebA dataset: please download the [CelebA dataset](http://www.robots.ox.ac.uk/~vgg/research/unsupervised_landmarks/resources/celeba.zip), unzip and place it under ``./celeba``. Please also copy the file ``list_landmarks_align_celeba.txt`` to this repo's path ``./``.

* AFLW dataset: please download the [AFLW dataset](http://www.robots.ox.ac.uk/~vgg/research/unsupervised_landmarks/resources/aflw_release-2.zip), unzip and place it under ``./aflw_release-2``.

* The pretrained checkpoint by [ESanchezLozano](https://github.com/ESanchezLozano) is placed in ``./checkpoint_fansoft/fan_109.pth`` which is a landmark detector pretrained on MPII human joint detection.

### Training

To train the model, first train on the CelebA dataset:

    ```
    python train.py --data_path celeba/Img/img_align_celeba_hq/ --cuda 1 --bSize 32 --num_workers 4
    ```

### Testing

The trained model is saved at ``./Exp_xxx``. To test the trained model, first we need to extract the detected results on target datasets, i.e. AFLW or MAFL, for both training and test partitions. The default number of keypoints N=10:

    ```
    python extract_data.py -f Exp_354 -e 33 -c checkpoint_fansoft/fan_109.pth -d MAFL-train --data_path celeba/Img/img_align_celeba_hq/ --cuda 1
    python extract_data.py -f Exp_354 -e 33 -c checkpoint_fansoft/fan_109.pth -d MAFL-test --data_path celeba/Img/img_align_celeba_hq/ --cuda 1
    ```

Then we can train a linear regressor and compute NME errors:

    ```
    python -f Exp_354 -e 33 -d MAFL -r 0.0001
    ```

### Related Project
``` 
Structured Landmark Detection via Topology-Adapting Deep Graph Learning
Weijian Li, Yuhang Lu, Kang Zheng, Haofu Liao, Chihung Lin, 
Jiebo Luo, Chi-Tung Cheng, Jing Xiao, Le Lu, Chang-Fu Kuo, Shun Miao
ECCV 2020
```
[[paper]](https://arxiv.org/pdf/2004.08190.pdf)

### Credits
The main structure of codes and checkpoints are provided by previous work:
[Object landmark discovery through unsupervised adaptation](https://github.com/ESanchezLozano/SAIC-Unsupervised-landmark-detection-NeurIPS2019)





