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









