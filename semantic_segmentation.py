import numpy as np
import mxnet as mx
import gluoncv
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import PIL.Image as pil
ctx = mx.cpu(0) # using cpu

def get_pretrained_model():
    pretrained_model = gluoncv.model_zoo.get_deeplab_resnet101_ade(pretrained=True)
    return pretrained_model

def get_segementation(input_img, seg_model):
    
    img_t = gluoncv.data.transforms.presets.segmentation.test_transform(input_img, ctx)

    # Use model to get prediction of segmentation labels
    output = seg_model.predict(img_t)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

    mask = gluoncv.utils.viz.get_color_pallete(predict, 'ade20k')
    mask.save('images/outputs/segmentation-output.png')

    mmask = mpimg.imread('images/outputs/segmentation-output.png')
    return mmask

def remove_walls(input_img, other):
    # Make copy of image and set the indices of everything but the walls to black
    input_copy = input_img.copy().asnumpy()
    input_copy = input_copy.astype(float)
    input_copy[other[0], other[1]] = np.array([0, 0, 0], dtype=float)
    input_copy = np.clip(input_copy, 0, 255)

    segmented_input = pil.fromarray(input_copy.astype(np.uint8))
    segmented_input.save('images/outputs/segmentation-walls-only.png')
    return segmented_input