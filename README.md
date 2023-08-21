# TransVOD_Lite
**by [Qianyu Zhou](https://qianyuzqy.github.io/), [Xiangtai Li](https://lxtgh.github.io/), [Lu He](https://github.com/SJTU-LuHe)**, [Yibo Yang](), [Guangliang Cheng](), [Yunhai Tong](), [Lizhuang Ma](https://dmcv.sjtu.edu.cn/people/), [Dacheng Tao]()
**modified by [Mohammad Fisal Aly Akbar](https://github.com/fisal-ITS)** 

**[[Arxiv]](https://arxiv.org/pdf/2201.05047.pdf)**
**[[Paper]](https://ieeexplore.ieee.org/document/9960850)**


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transvod-end-to-end-video-object-detection/video-object-detection-on-imagenet-vid)](https://paperswithcode.com/sota/video-object-detection-on-imagenet-vid?p=transvod-end-to-end-video-object-detection)

(TPAMI 2023) [TransVOD:End-to-End Video Object Detection with Spatial-Temporal Transformers](https://ieeexplore.ieee.org/document/9960850).


## Updates
- (August 2023) Modified for thesis purposes 
- (December 2022) Checkpoints of pretrained models are released. 
- (December 2022) Code of TransVOD Lite are released. 

## Citing TransVOD
If you find TransVOD useful in your research, please consider citing:
```bibtex
@article{zhou2022transvod,
 author={Zhou, Qianyu and Li, Xiangtai and He, Lu and Yang, Yibo and Cheng, Guangliang and Tong, Yunhai and Ma, Lizhuang and Tao, Dacheng},  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},   
 title={TransVOD: End-to-End Video Object Detection With Spatial-Temporal Transformers},   
 year={2022},   
 pages={1-16},  
 doi={10.1109/TPAMI.2022.3223955}}


@inproceedings{he2021end,
  title={End-to-End Video Object Detection with Spatial-Temporal Transformers},
  author={He, Lu and Zhou, Qianyu and Li, Xiangtai and Niu, Li and Cheng, Guangliang and Li, Xiao and Liu, Wenxuan and Tong, Yunhai and Ma, Lizhuang and Zhang, Liqing},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={1507--1516},
  year={2021}
}
```


## Abstract
Infrastructure development, particularly roads, is rapidly expanding in Indonesia. Concrete pavement is favored for its strength and durability. However, natural and human factors lead to damage, with cracking being a common initial issue that can escalate if left unaddressed. To streamline inspection, an automated crack detection system is essential. This study focuses on a Transformer Network-based model, TransVOD Lite, to detect cracks in concrete roads from video data. This architecture combines CNN for image feature extraction and Spatial-Temporal Transformer for object detection. The research involves three stages: data collection with annotated videos of cracked roads, model training for crack recognition, and performance evaluation. The TransVOD Lite achieved a 47.5% mean average precision (mAP) on validation data. For test data, it excelled in low-light and 20-30 km/h scenarios, achieving 90.46% accuracy, 80.77% precision, 94.23% recall, and 86.98% F1-score.



*Note:*
1. All models of TransVOD are trained  with pre-trained weights on COCO dataset.
2. Backbone that used are ResNet101, Swin B, and Swin S
3. Dataset that used is scrapped by self


## Installation

The codebase is built on top of [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), [TransVOD](https://github.com/SJTU-LuHe/TransVOD), and [TransVOD Lite](https://github.com/qianyuzqy/TransVOD_Lite).

### Requirements

* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7
  
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/)

    For example, if your CUDA version is 9.2, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

* Build MultiScaleDeformableAttention
    ```bash
    cd ./models/ops
    sh ./make.sh
    ```

## Usage

### Checkpoints

Below, we provide checkpoints, training logs and inference logs of TransVOD Lite for different backbones.

[DownLoad Link of Google Drive](https://drive.google.com/drive/folders/1eqpiVLAWA_oPYiGmP6CW03WJlXVDjy6e?usp=share_link)

[DownLoad Link of Baidu Netdisk](https://pan.baidu.com/s/1WAXRgXODX1tZ5PNkNOGDaA) (password:26xc)


### Dataset preparation

1. We used our own custom dataset of video footage of a concrete road with cracks. Please see the dataset and annotations at the following link: [Concrete Crack Detection Dataset by Fisal](https://drive.google.com/drive/folders/1gqiL-w3RkRptfbp_Sgsrk07jn8gYe1i2?usp=sharing). And the path structure should be as follows:

```
code_root/
└── data/
    └── vid/
        ├── Data
            ├── VID/
            └── DET/
        └── annotations/
        	  ├── custom.json
        	  └── custom_val.json

```

### Training
We train TransVOD Lite using Google Colab with Tesla T4 GPU on single node. For more detail check on [this file](https://github.com/qianyuzqy/TransVOD_Lite/blob/main/TransVODLite_full.ipynb)

## Acknowledgements

This project is based on the following open-source projects. We thank their
authors for making the source code publically available.

* [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)


## License

This project is released under the [Apache License 2.0](LICENSE), while some 
specific features in this repository are with other licenses. Please refer to 
[LICENSES.md](LICENSES.md) for the careful check, if you are using our code for 
commercial matters.




