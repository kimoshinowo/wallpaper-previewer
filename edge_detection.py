import cv2
import numpy as np
import PIL.Image as pil
from matplotlib import pyplot as plt


def detect_edges(image: np.ndarray) -> pil.Image:
    """Implements canny edge detection and required image preprocessing.

    Parameters
    ----------
    image : np.ndarray
        Input image to perform edge detection on.

    Returns
    -------
    pil.Image
        Image showing detected edges, with the same shape as the input image.
    """
    blur = cv2.blur(image, (3, 3))  # Add blur
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    edges = cv2.Canny(gray, 15, 50, apertureSize=3)  # Use canny method
    edge_map = pil.fromarray(edges)
    edge_map.save("images/outputs/edge-detection-output.png")

    return edge_map


def get_segmented_edges(edge_map: pil.Image, walls: np.ndarray) -> np.ndarray:
    """Show only those edges which fall within wall regions.

    Parameters
    ----------
    edge_map : pil.Image
        Image containing the edge map produced by canny edge detection.
    walls : np.ndarray
        Indices of pixels which were classified as walls.

    Returns
    -------
    np.ndarray
        Image containing only edges within wall regions.
    """
    edge_map_array = np.asarray(edge_map.convert("RGB"))
    segmented_edges = np.empty((edge_map_array.shape[0], edge_map_array.shape[1], 3))
    segmented_edges[:] = np.nan
    segmented_edges[walls[0], walls[1]] = edge_map_array[walls[0], walls[1]]
    segmented_edges = segmented_edges.astype(dtype=np.uint8)

    pil.fromarray(segmented_edges).save("images/outputs/segmented-edges.png")

    return segmented_edges


def hough_transform(image: np.ndarray) -> np.ndarray:
    """Performs the hough transform.

    Parameters
    ----------
    image : np.ndarray
        Image on which to perform the hough transform.

    Returns
    -------
    np.ndarray
        An image containing all lines detected using the hough transform.
    """
    test = np.asarray(image)
    hough_img = np.empty((test.shape[0], test.shape[1], 3))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply HoughLinesP method to directly obtain line end points
    lines_list = []
    lines = cv2.HoughLinesP(
        image,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 180,  # Angle resolution in radians
        threshold=50,  # Min number of votes for valid line
        minLineLength=25,  # Min allowed length of line
        maxLineGap=15,  # Max allowed gap between line for joining them
    )

    # Iterate over points
    for points in lines:
        # Extracted points nested in the list
        x1, y1, x2, y2 = points[0]
        # Draw the lines joing the points on the original image
        cv2.line(hough_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # Maintain a simples lookup list for points
        lines_list.append([(x1, y1), (x2, y2)])

    cv2.imwrite("images/outputs/hough-output.png", hough_img)

    return hough_img


def get_vertical_lines(hough_img: np.ndarray) -> np.ndarray:
    """Get vertical lines only - https://www.youtube.com/watch?v=veoz_46gOkc

    Parameters
    ----------
    hough_img : np.ndarray
        hough_transform output image.

    Returns
    -------
    np.ndarray
        An image containing only the vertical lines of the hough transform.
    """
    kernel = np.ones((20, 1), np.uint8)
    vertical_lines = cv2.erode(hough_img, kernel, iterations=1)
    cv2.imwrite("images/outputs/hough_corners.png", vertical_lines)
    corners = plt.imread("images/outputs/hough_corners.png")[:, :, :3] * 255

    return corners


def get_hough_corners(colours: np.ndarray) -> np.ndarray:
    """Find image column indices of hough transform lines.

    Parameters
    ----------
    colours : np.ndarray
        Image with combined rgb string values at each pixel.

    Returns
    -------
    np.ndarray
        Column indices where colour is found.
    """
    _, hough_corners = np.where(colours == "255.0,0.0,0.0")

    return hough_corners
