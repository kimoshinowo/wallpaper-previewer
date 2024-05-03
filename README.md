# WallpaperView: a wallpaper previewer

This is a wallpaper previewing pipline, which allows input of a room image and a wallpaper sample image, and returns an augmented image with the wallpaper overlayed on the walls. Pre-trained semantic segmentation (gluoncv.model_zoo's deeplab_resnet101_ade) and monocular depth estimation (vinvino02/glpn-nyu) models are used in this pipeline.

## Required Installations

The required conda environment can be created from `pipeline-env.yml`.

## Running the pipeline

The files needed to run the pipeline are: depth-estimation.py, edge_detection.py, general_methods.py, geometry.py, semantic_segmentation.py, transforms.py and wallpaperview.py. These should all be stored in the same directory. Then run the following in order to run the application:

`python wallpaperview.py`

Follow the instructions shown in the command prompt once the script is running. Outputs will be saved to a folder called 'outputs' in the same directory as the code.


## Evaluating the pipeline

The code used for analysis of the pipeline and an existing application is stored in the performance-testing-and-analysis directory.
