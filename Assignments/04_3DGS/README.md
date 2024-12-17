# Assignment 4 - Implement Simplified 3D Gaussian Splatting

This repository is Yuanhao Li's implementation of Assignment_04 of DIP.

This assignment covers a complete pipeline for reconstructing a 3D scene represented by 3DGS from multi-view images. The following steps use the [chair folder](data/chair); you can use any other folder by placing images/ in it.

## Running

### Step 1. Structure-from-Motion
First, we use Colmap to recover camera poses and a set of 3D points. Please refer to [11-3D_from_Multiview.pptx](https://rec.ustc.edu.cn/share/705bfa50-6e53-11ef-b955-bb76c0fede49) to review the technical details.
```
python mvs_with_colmap.py --data_dir data/chair
```

Debug the reconstruction by running:
```
python debug_mvs_by_projecting_pts.py --data_dir data/chair
```

### Step 2. A Simplified 3D Gaussian Splatting
Build your 3DGS model:
```
python train.py --colmap_dir data/chair --checkpoint_dir data/chair/checkpoints
```

## Results

<video controls autoplay muted loop width=500>
  <source src="debug_rendering.mp4" type="video/mp4">
  <p>
    你的浏览器不支持 HTML5 视频。这里有一个<a
      href="debug_rendering.mp4"
      download="debug_rendering.mp4"
      >视频</a
    >链接。
  </p>
</video>
<video controls autoplay muted loop width=500>
  <source src="debug_rendering (2).mp4" type="video/mp4">
  <p>
    你的浏览器不支持 HTML5 视频。这里有一个<a
      href="debug_rendering (2).mp4"
      download="debug_rendering (2).mp4"
      >视频</a
    >链接。
  </p>
</video>

### Compare with the original 3DGS Implementation
Since we use a pure PyTorch implementation, the training speed and GPU memory usage are far from satisfactory. Also, we do not implement some crucial parts like adaptive Gaussian densification scheme. Run the [original 3DGS implementation](https://github.com/graphdeco-inria/gaussian-splatting) with the same dataset to compare the results.

### Acknowledgement:
- [Paper: 3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [3DGS Official Implementation](https://github.com/graphdeco-inria/gaussian-splatting)
- [Colmap for Structure-from-Motion](https://colmap.github.io/index.html)
