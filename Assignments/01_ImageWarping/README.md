# Assignment 1 - Image Warping

## Implementation of Image Geometric Transformation

This repository is Yuanhao Li's implementation of Assignment_01 of DIP. 

<img src="pics/Screenshot 2024-09-20 at 13-55-23 Gradio.png" alt="Image Deformation Using Moving Least Squares" width="800">

## Requirements

To install requirements:

```setup
python -m pip install -r requirements.txt
```


## Running

To run basic transformation, run:

```basic
python run_global_transform.py
```

To run point guided transformation, run:

```point
python run_point_transform.py
```

## Results
### Basic Transformation
<img src="pics/global_demo.gif" alt="Basic transformation" width="800">

### Point Guided Deformation:
<img src="pics/point_demo.gif" alt="Point guided deformation (moving least squares)" width="800">

Original image:

![Original image](pics/image.png)

Warped image using moving least squares:

![Warped image using moving least squares](pics/image_warped.png)

## Acknowledgement

>📋 Thanks for the algorithms proposed by [Image Deformation Using Moving Least Squares](https://people.engr.tamu.edu/schaefer/research/mls.pdf).
