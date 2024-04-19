from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import torch
import cv2
import numpy as np
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
import PIL.Image as pil


def estimate_depth(image: pil.Image) -> pil.Image:
    """Performs monocular depth estimation on an rgb image.
    Implemented as per pre-trained model instructions https://huggingface.co/vinvino02/glpn-nyu

    Parameters
    ----------
    image : pil.Image
        Image to perform depth estimation on.

    Returns
    -------
    pil.Image
        The estimated depth map of the input image.
    """

    processor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
    model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

    # prepare image for the model
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # visualize the prediction
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = pil.fromarray(formatted)

    plt.imsave("images/outputs/intermediate-outputs/depth-output.png", depth)

    return depth


def get_mean_depths(depth_image: pil.Image, other: np.ndarray) -> np.ndarray:
    """Find the mean depth for each column (1px wide) in the wall depth map and plot.

    Parameters
    ----------
    depth_image : pil.Image
        Estimated depth map image.
    other : np.ndarray
        Indices of non-wall pixels.

    Returns
    -------
    np.ndarray
        Column-wise mean non-wall depth.
    """
    test_depth = np.array(depth_image).copy().astype(float)
    test_depth[other[0], other[1]] = np.nan

    mean_depth = np.nanmean(test_depth, axis=0)
    
    try:
        window_size = np.round(len(mean_depth)/20, 0).astype(int)

        if window_size % 2 == 0:
            window_size += 1
        
        # Smooth using savgol filter
        mean_depth = savgol_filter(
            mean_depth, window_size, 3
        )
    except Exception:
        mean_depth = np.nanmean(test_depth, axis=0)

    return mean_depth


def get_harris_corners(matrix: np.ndarray) -> np.ndarray:
    """Performs and plots harris corner detection.

    Parameters
    ----------
    matrix : np.ndarray
        Array to perform corner detection on.

    Returns
    -------
    np.ndarray
        A matrix containing the input with corner points highlighted.
    """
    operatedImage = np.float32(matrix)
    operatedImage = cv2.cvtColor(operatedImage, cv2.COLOR_BGR2GRAY)
    operatedImage = np.float32(operatedImage)

    dest = cv2.cornerHarris(operatedImage, 30, 5, 0.07)
    dest = cv2.dilate(dest, None)

    # Reverting back to the original image, with optimal threshold value
    matrix[dest > 0.01 * dest.max()] = [255, 0, 0]
    pil.fromarray(matrix.astype(np.uint8)).save("images/outputs/intermediate-outputs/corners.png")

    return matrix
