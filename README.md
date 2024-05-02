# WallpaperView: a wallpaper previewer

This is a wallpaper previewing pipline, which allows input of a room image and a wallpaper sample image, and returns an augmented image with the wallpaper overlayed on the walls. Pre-trained semantic segmentation (gluoncv.model_zoo's deeplab_resnet101_ade) and monocular depth estimation (vinvino02/glpn-nyu) models are used in this pipeline. All of the code for this pipeline, plus collected ground truth data and image test sets, can be found (here)[https://github.com/kimoshinowo/wallpaper-previewer].

## Required Installations

The required conda environment can be created from `pipeline-env.txt`.

## Running the pipeline

Run the following in order to run the application:

`python wallpaperview.py`

Follow the instructions shown in the command prompt once the script is running.
