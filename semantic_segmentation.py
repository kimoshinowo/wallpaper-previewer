import numpy as np
import mxnet as mx
import gluoncv
import matplotlib.image as mpimg
import PIL.Image as pil

ctx = mx.cpu(0)  # using cpu


def get_pretrained_model() -> gluoncv.model_zoo.deeplabv3.DeepLabV3:
    """Returns the pretrained deeplabV3 semantic segmentation model.

    Returns
    -------
    gluoncv.model_zoo.deeplabv3.DeepLabV3
        Pretrained semantic segmentation model.
    """
    pretrained_model = gluoncv.model_zoo.get_deeplab_resnet101_ade(pretrained=True)
    return pretrained_model


def get_segementation(
    input_img: np.ndarray, seg_model: gluoncv.model_zoo.deeplabv3.DeepLabV3
) -> pil.Image:
    """Performs semantic segmentation on the input image using the provided model.

    Parameters
    ----------
    input_img : np.ndarray
        Input image on which to perform semantic segmentation.
    seg_model : gluoncv.model_zoo.deeplabv3.DeepLabV3
        Pretrained semantic segmentation model.

    Returns
    -------
    pil.Image
        The semantic segmentation map of the input image.
    """
    img_t = gluoncv.data.transforms.presets.segmentation.test_transform(input_img, ctx)

    # Use model to get prediction of segmentation labels
    output = seg_model.predict(img_t)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

    mask = gluoncv.utils.viz.get_color_pallete(predict, "ade20k")
    mask.save("images/outputs/intermediate-outputs/segmentation-output.png")

    mmask = mpimg.imread("images/outputs/intermediate-outputs/segmentation-output.png")
    return mmask


def remove_inds(input_img: np.ndarray, inds: list, colour: list) -> np.ndarray:
    """Make a copy of the image and set the indices of everything but the walls to black.

    Parameters
    ----------
    input_img : np.ndarray
        The input image of a room interior.
    other : np.ndarray
        Indices of non-wall pixels in the input image.

    Returns
    -------
    pil.Image
        The input image with everything except the walls set to black.
    """
    input_copy = input_img.copy()
    if not isinstance(input_copy, np.ndarray):
        input_copy = input_copy.asnumpy()
    input_copy = input_copy.astype(float)
    for ind_list in inds:
        input_copy[ind_list[0], ind_list[1]] = np.array(colour, dtype=float)
    input_copy = np.clip(input_copy, 0, 255)

    segmented_input = input_copy.astype(np.uint8)
    segmented_input_img = pil.fromarray(segmented_input)
    segmented_input_img.save("images/outputs/intermediate-outputs/segmentation-walls-only.png")

    return segmented_input
