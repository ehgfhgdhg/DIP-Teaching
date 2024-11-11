# Assignment 3 - Play with GANs

# Implementation of deep learning-based DIP (Pix2Pix) with pytorch

This repository is Yuanhao Li's implementation of Assignment_03 of DIP.

## Running

To run Pix2Pix, run:
```bash
bash download_facades_dataset.sh
python train.py
```
to train the network.

The provided code will train the model on the [Facades Dataset](https://cmp.felk.cvut.cz/~tylecr1/facade/). You need to use [other datasets](https://github.com/phillipi/pix2pix#datasets) containing more images for better generalization on the validation set.

## Results

Results using GAN:

Training data:

<img src="result_1 (3).png" alt="result_1" height="100">
<img src="result_2 (3).png" alt="result_2" height="100">
<img src="result_3 (3).png" alt="result_3" height="100">
<img src="result_4 (3).png" alt="result_4" height="100">
<img src="result_5 (3).png" alt="result_5" height="100">

Validation data:

<img src="result_1 (4).png" alt="result_1" height="100">
<img src="result_2 (4).png" alt="result_2" height="100">
<img src="result_3 (4).png" alt="result_3" height="100">
<img src="result_4 (4).png" alt="result_4" height="100">
<img src="result_5 (4).png" alt="result_5" height="100">

Previous results without GAN:

<img src="result_1 (2).png" alt="result_1" height="100">
<img src="result_2 (2).png" alt="result_2" height="100">
<img src="result_3 (2).png" alt="result_3" height="100">
<img src="result_4 (2).png" alt="result_4" height="100">
<img src="result_5 (2).png" alt="result_5" height="100">

## Acknowledgement

- [DragGAN](https://vcai.mpi-inf.mpg.de/projects/DragGAN/): [Implementaion 1](https://github.com/XingangPan/DragGAN) & [Implementaion 2](https://github.com/OpenGVLab/DragGAN)
- [Facial Landmarks Detection](https://github.com/1adrianb/face-alignment)
- [作业03-Play_with_GANs.pptx](https://rec.ustc.edu.cn/share/705bfa50-6e53-11ef-b955-bb76c0fede49)
