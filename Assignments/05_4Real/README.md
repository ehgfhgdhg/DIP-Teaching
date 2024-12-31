# Assignment 5 - 4Real

This repository is Yuanhao Li's implementation of Assignment_05 of DIP.

4Real use a text-to-video diffusion model to generate a realistic 4D scene from a text prompt. They use deformable 3D Gaussian Splatting (D-3DGS) to model the scene. They first generate a reference freeze-time video to reconstruct the 3D scene using 3D Gaussian Splatting, then they use a neural network to deform 3D Gaussian Splatting to match the input video to generate the desired 4D Gaussian Splatting.

## Running

### Generate video dataset

Use text-to-video diffusion model to generate a corresponding video and a freeze-time video, then use [ffmpeg](https://ffmpeg.org/download.html) to convert them into image dataset:
```shell
mkdir -p <path to image COLMAP dataset>/images
ffmpeg -i <your video path> <path to image COLMAP dataset>/images/r_%d.png
```

Then generate COLMAP data in **freeze-time** video image dataset:
```shell
python mvs_with_colmap.py --data_dir <path to COLMAP freeze-time image dataset>
```

Finally we generate COLMAP data in **animated** video image dataset:
```shell
python mvs_with_colmap_temporal.py --freeze_data_dir <path to COLMAP freeze-time image dataset> --animated_data_dir <path to COLMAP animated image dataset>
```
You can specify `--register_image r_<num>.png` to choose the image view data to register in freeze-time image dataset.

### Spatial training of freeze-time video

First ensure your hardware and software requirements and create a conda environment as in [README_3DGS.md](README_3DGS.md#L81), then run the spatial optimizer to train 3D Gaussian Splatting in freeze-time video image dataset:
```shell
python train.py -s <path to COLMAP freeze-time image dataset> -m <path to save 3DGS model> --checkpoint_iterations 30000
```
Save 3DGS checkpoint `chkpnt30000.pth` for further temporal training.

### Temporal training of 4D Gaussian Splatting

Use this command to train in animated video:
```shell
python train_temporal.py -s <path to COLMAP animated image dataset> --start_checkpoint <path to saved 3DGS model>/chkpnt30000.pth --m <path to save temporal neural network> --iterations 36000 --checkpoint_iteration 31000 32000 33000 34000 35000 36000 --deform_scheduler_step 1000
```

### Viewer

Use `SIBR_remoteGaussian_app` compiled in `SIBR_viewers` to view 4D Gaussian Splatting as in [README_3DGS.md](README_3DGS.md#L301) when in training. Moreover, after temporal training complete, the script `train_temporal.py` will wait some time for connecting with `SIBR_remoteGaussian_app` to render.

## Results

## Acknowledgement
