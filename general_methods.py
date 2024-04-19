import cv2
import numpy as np
import PIL.Image as pil
from mxnet import image as mimage
from scipy.ndimage.filters import gaussian_filter


def import_and_resize(filename: str) -> pil.Image:
    """Open image with PIL and resave to smaller size if needed (to stop CPU error).

    Parameters
    ----------
    filename : str
        Path and filename of image to load and resize.

    Returns
    -------
    pil.Image
        The image specified by the filename, in PIL image object format.
    """

    input_img = pil.open(filename)
    input_img.thumbnail((1000, 1000))
    input_img.save("images/outputs/intermediate-outputs/resized-input.png")
    return input_img


def import_mx_image(filename: str) -> np.ndarray:
    """Loads an image using the mxnet image loader.

    Parameters
    ----------
    filename : str
        Path and filename of image to load.

    Returns
    -------
    np.ndarray
        An image stored as a numpy array (h, w, (r,g,b))
    """
    input_img = mimage.imread(filename)
    return input_img


def import_cv2_image(filename: str) -> np.ndarray:
    """Loads an image from the cv2 image reader.

    Parameters
    ----------
    filename : str
        Path and filename of image to load.

    Returns
    -------
    np.ndarray
        An image stored as a numpy array (h, w, (r,g,b))
    """
    input_img = cv2.imread(filename)
    return input_img


def get_labels_string(img: np.ndarray, rng: int) -> np.ndarray:
    """Combines RGB values of an image into a string at each pixel to use as a label.

    Parameters
    ----------
    img : np.ndarray
        An image.

    Returns
    -------
    np.ndarray
        Image shaped output where each 'pixel' is a string.
    """
    labels = []
    for i in range(rng):
        row = []
        for j in range(img.shape[1]):
            row.append(",".join(img[i, j].astype(str)))
        labels.append(row)

    labels = np.array(labels)
    return labels


def find_colour_indices(labels: np.ndarray, colour_string: str) -> np.ndarray:
    """Save the indices of every pixel that is the specified colour.

    Parameters
    ----------
    labels : np.ndarray
        Output of get_labels_string, string reduced segmentation map.
    colour_string : str
        A string containing the RGBa (a=alpha/opacity) values of the colour to match, separated by commas.

    Returns
    -------
    np.ndarray
        Indices of each pixel in the image that are the specificied colour.
    """
    inds_x, inds_y = np.where(labels == colour_string)
    inds = np.array([inds_x, inds_y])
    return inds


def find_not_colour_indices(labels: np.ndarray, colour_string: str) -> np.ndarray:
    """Save the indices of every pixel that isn't the specified colour.

    Parameters
    ----------
    labels : np.ndarray
        Output of get_labels_string, string reduced segmentation map.
    colour_string : str
        A string containing the RGBa (a=alpha/opacity) values of the colour to match, separated by commas.

    Returns
    -------
    np.ndarray
        Indices of each pixel in the image that aren't the specificied colour.
    """
    inds_x, inds_y = np.where(labels != colour_string)
    inds = np.array([inds_x, inds_y])
    return inds


# Alternative method to find indices of a certain colour (which takes longer):

# wall_colour = np.array([0.47058824, 0.47058824, 0.47058824, 1.0])
# walls_x, walls_y, other_x, other_y = [], [], [], []
# for i in range(mmask.shape[0]):
#     for j in range(mmask.shape[1]):
#         if np.allclose(mmask[i][j], wall_colour):
#             walls_x.append(i)
#             walls_y.append(j)
#         else:
#             other_x.append(i)
#             other_y.append(j)


def get_matrix(line: np.ndarray) -> np.ndarray:
    """Create a matrix image of a line graph.

    Parameters
    ----------
    line : np.ndarray
        Points for a line graph. Usually mean depth at each pixel in the width of the image.

    Returns
    -------
    np.ndarray
        Matrix version of the line graph.
    """
    y_round = np.round(line.copy(), 0).astype(int)
    y_round = y_round - np.nanmin(line)
    y_round = np.where(y_round < 0, 0, y_round)
    y_round = np.round(y_round, 0).astype(int)
    x = range(len(y_round))
    matrix = np.zeros((np.amax(y_round) + 1, len(x), 3))

    for i in x:
        matrix[y_round[i], i] = [255, 255, 255]

    matrix = matrix.astype(np.uint8)
    pil.fromarray(matrix).save("images/outputs/intermediate-outputs/matrix.png")

    return matrix
