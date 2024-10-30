# Assignment 2 - DIP with PyTorch

# Implementation of deep learning-based DIP (Pix2Pix) with pytorch

This repository is Yuanhao Li's implementation of Assignment_02 of DIP.

## Running

To run Pix2Pix, run:
```bash
bash download_facades_dataset.sh
python train.py
```
to train the network.

The provided code will train the model on the [Facades Dataset](https://cmp.felk.cvut.cz/~tylecr1/facade/). You need to use [other datasets](https://github.com/phillipi/pix2pix#datasets) containing more images for better generalization on the validation set.

## Results

Training data:

<img src="result_1.png" alt="result_1" height="100">
<img src="result_2.png" alt="result_2" height="100">
<img src="result_3.png" alt="result_3" height="100">
<img src="result_4.png" alt="result_4" height="100">
<img src="result_5.png" alt="result_5" height="100">

Validation data:

<img src="result_1 (2).png" alt="result_1" height="100">
<img src="result_2 (2).png" alt="result_2" height="100">
<img src="result_3 (2).png" alt="result_3" height="100">
<img src="result_4 (2).png" alt="result_4" height="100">
<img src="result_5 (2).png" alt="result_5" height="100">

## Acknowledgement

- [Assignment Slides](https://rec.ustc.edu.cn/share/705bfa50-6e53-11ef-b955-bb76c0fede49)
- [Paper: Image-to-Image Translation with Conditional Adversarial Nets](https://phillipi.github.io/pix2pix/)
- [Paper: Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
- [PyTorch Installation & Docs](https://pytorch.org/)
