## Corresponding Paper

This project correspond to the paper  Rui Zhang, Ziheng Jiao, Hongyuan Zhang, and Xuelong Li, "**Manifold Neural Network With Non-Gradient Optimization.**" IEEE Trans. Pattern Anal. Tntell. 45(3): 3986-3993 (2023).

## Cite

Please kindly cite our paper if you use this code in your own work:

```
@ARTICLE{9773979,
  author={Zhang, Rui and Jiao, Ziheng and Zhang, Hongyuan and Li, Xuelong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Manifold Neural Network With Non-Gradient Optimization}, 
  year={2023},
  volume={45},
  number={3},
  pages={3986-3993},
  doi={10.1109/TPAMI.2022.3174574}}
```

## Author of Code

Ziheng Jiao and Hongyuan Zhang

## Dependence

The model and comparison models in paper "**Non-Gradient Manifold Neural Network**" are all implemented with Pytorch 1.2.0, CUDA 10.0 on Windows 10 PC. The following packages you need is several well-known ones, including: 

- python==3.6
- pytorch==1.2.0
- torchvision==0.4.0
- numpy==1.19.1
- scipy==1.2.1
- scikit-learn==0.19.2

## Brief Introduction

- model.py: the framework code of the proposed model.
- load_data.py: code to load data. 
- utils.py: functions used in experiemnts.
- data: training and testing datasets. 
- train.py: the training code.

You can test the code by the following command

```shell
python train.py
```
