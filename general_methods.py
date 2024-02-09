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
        The imagee specified by the filename, in pil image object format.
    """

    input_img = pil.open(filename)
    input_img.thumbnail((1000, 1000))
    input_img.save(filename)
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


def get_labels_string(mmask: np.ndarray) -> np.ndarray:
    """Combines rgb values of an image  into a string at each pixel to use as a label.

    Parameters
    ----------
    mmask : np.ndarray
        An image.

    Returns
    -------
    np.ndarray
        Image shaped output where each 'pixel' is a string.
    """
    labels = []
    for i in range(mmask.shape[0]):
        row = []
        for j in range(mmask.shape[1]):
            row.append(",".join(mmask[i, j].astype(str)))
        labels.append(row)

    labels = np.array(labels)
    return labels


def find_wall_indices(labels: np.ndarray) -> np.ndarray:
    """Save the indices of every pixel that is part of the walls.

    Parameters
    ----------
    labels : np.ndarray
        Output of get_labels_string, string reduced segmentation map.

    Returns
    -------
    np.ndarray
        Indices of each pixel in the image that belongs to a wall.
    """
    walls_x, walls_y = np.where(labels == "0.47058824,0.47058824,0.47058824,1.0")
    walls = np.array([walls_x, walls_y])
    return walls


def find_non_wall_indices(labels: np.ndarray) -> np.ndarray:
    """Save the indices of every pixel that isn't part of the walls

    Parameters
    ----------
    labels : np.ndarray
        Output of get_labels_string, string reduced segmentation map.

    Returns
    -------
    np.ndarray
        Indices of each pixel in the image that doesn't belong to a wall.
    """
    other_x, other_y = np.where(labels != "0.47058824,0.47058824,0.47058824,1.0")
    other = np.array([other_x, other_y])
    return other


# Alternative method which takes longer:

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
    """!!!

    Parameters
    ----------
    line : np.ndarray
        Mean depth at each pixel in the width of the image.

    Returns
    -------
    np.ndarray
        !!!
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
    pil.fromarray(matrix).save("images/outputs/matrix.png")

    return matrix


# !!! I've assumed this is going to be removed
# def makeShadow(image, iterations, border, offset, backgroundColour, shadowColour):
#     # image: base image to give a drop shadow
#     # iterations: number of times to apply the blur filter to the shadow
#     # border: border to give the image to leave space for the shadow
#     # offset: offset of the shadow as [x,y]
#     # backgroundCOlour: colour of the background
#     # shadowColour: colour of the drop shadow

#     # Calculate the size of the shadow's image
#     fullWidth = image.size[0] + abs(offset[0]) + 2 * border
#     fullHeight = image.size[1] + abs(offset[1]) + 2 * border

#     # Create the shadow's image. Match the parent image's mode.
#     shadow = pil.new(image.mode, (fullWidth, fullHeight), backgroundColour)

#     # Place the shadow, with the required offset
#     shadowLeft = border + max(offset[0], 0)  # if <0, push the rest of the image right
#     shadowTop = border + max(offset[1], 0)  # if <0, push the rest of the image down
#     # Paste in the constant colour
#     shadow.paste(
#         shadowColour,
#         [shadowLeft, shadowTop, shadowLeft + image.size[0], shadowTop + image.size[1]],
#     )

#     # Apply the BLUR filter repeatedly
#     for i in range(iterations):
#         shadow = shadow.filter(ImageFilter.BLUR)

#     # Paste the original image on top of the shadow
#     imgLeft = border - min(offset[0], 0)  # if the shadow offset was <0, push right
#     imgTop = border - min(offset[1], 0)  # if the shadow offset was <0, push down
#     shadow.paste(image, (imgLeft, imgTop))

#     return shadow


# !!! I've assumed this isn't finished yet
# def add_shadows(image, corner_inds, height, width, other, walls):
#     shadow_image = image.copy()

#     lines = np.ones((height, width, 3), dtype=np.uint8) * 255
#     lines[:, corner_inds] = [0, 0, 0]
#     lines = pil.fromarray(lines, "RGB")

#     line = np.zeros((height, 3, 3), dtype=np.uint8)
#     line = pil.fromarray(line, "RGB")
#     shadow = makeShadow(line, 5, 50, [0, 0], "white", "black")

#     plt.imshow(lines)
#     return shadow_image


def add_shadows(
    image: np.ndarray,
    corners: np.ndarray,
    shadow_line_width: int,
    shadow_fade: int,
    shadow_opacity: float
) -> np.ndarray:
    """Adds appearance of vertical shadow to a specific column area of an image.

    Parameters
    ----------
    image : np.ndarray
        Input image of a room interior.
    corners : np.ndarray
        Column indices of where on the room image corners are located.
    shadow_line_width : int
        The width of the shadow pre-blurring, as a percentage of image width.
    shadow_fade : int
        The fade of the shadow.
    shadow_opacity : int
        How opaque the shadow is, higher values cause less visible shadow.

    Returns
    -------
    np.ndarray
        The input image with shadow added at corner locations.
    """
    shadow_line_width = np.clip(
        np.round(image.shape[1] * (shadow_line_width / 100), 0).astype(int),
        0,
        image.shape[0],
    )

    lines = np.zeros(np.concatenate(([len(corners)], image.shape)))
    for i in range(len(corners)):
        lines[i, :, corners[i] - shadow_line_width : corners[i] + shadow_line_width] = [
            255,
            255,
            255,
        ]
        lines[i] = gaussian_filter(lines[i], sigma=image.shape[1] / shadow_fade) / shadow_opacity
    
    lines= np.amax(lines, axis=0)

    image = image - lines
    image = np.clip(image, 0, 255).astype(int)

    return image
