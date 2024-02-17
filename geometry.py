import numpy as np
from sklearn.cluster import DBSCAN
import PIL.Image as pil
import cv2
from matplotlib import pyplot as plt


def find_corners(hough_corners: np.ndarray, harris_corners: np.ndarray, savgol_corners, width: int) -> np.ndarray:
    """Performs clusting on the hough and harris corners to find likely room corners.

    Parameters
    ----------
    hough_corners : np.ndarray
        Image column indices of edges found by hough transform.
    harris_corners : np.ndarray
        Image column indices of edge found by harris corner detection.

    Returns
    -------
    np.ndarray
        Wall corner locations estimated via DBSCAN.
    """
    all_corners = np.concatenate((hough_corners, harris_corners))

    min_samples = int(all_corners.size // 14)

    # if savgol_corners.size > 0:
    #     length = harris_corners.size
    #     savgol_length = savgol_corners.size
    #     div = (length//2) // savgol_length
    #     tile = np.tile(savgol_corners, int(div))

    #     all_corners = np.concatenate((harris_corners, tile))

    #     min_samples = int(all_corners.size // 3)

    # if savgol_corners.size > 0:
    #     length = all_corners.size / 2
    #     savgol_length = savgol_corners.size
    #     div = (length//2) // savgol_length
    #     tile = np.tile(savgol_corners, int(div))

    #     all_corners = np.concatenate((all_corners, tile))

    #     min_samples = int(all_corners.size // 5)

    corner_inds = []

    if all_corners.size >= 0:
        # Find only most dense clusters
        clf = DBSCAN(eps=(0.02*width), min_samples=max(min_samples, 200)).fit(
            all_corners.reshape(-1, 1)
        )

        # Find centers of clusters by taking means
        centers = []
        for i in np.unique(clf.labels_):
            if i != -1:
                ind = np.where(clf.labels_ == i)
                ind = all_corners[ind]
                centers.append(np.mean(ind))

        centers = np.round(centers, 0)
        corner_inds = centers.astype(int)

    return corner_inds


def create_wall_corner_map(
    segmented_input: pil.Image,
    other: np.ndarray,
    walls: np.ndarray,
    corner_inds: np.ndarray,
) -> np.ndarray:
    """Generate a high contrast image showing only walls, non-walls, and wall corners, that can be used to find geometry.

    Parameters
    ----------
    segmented_input : pil.Image
        The segmentation map image.
    other : np.ndarray
        Indices of non-wall columns in the image.
    walls : np.ndarray
        Indices of wall columns in the image.
    corner_inds : np.ndarray
        Indices of estimated wall corner locations.

    Returns
    -------
    np.ndarray
        Image with white wall regions and black non-walls and wall corners. 
    """
    only_walls = np.array(segmented_input.copy())
    only_walls[other[0], other[1]] = [0, 0, 0]
    only_walls[walls[0], walls[1]] = [255, 255, 255]
    only_walls[:, corner_inds] = [0, 0, 0]

    pil_image = pil.fromarray(only_walls)
    pil_image.save("images/outputs/intermediate-outputs/segmented-with-corners.png")

    return only_walls


def find_contours(only_walls: np.ndarray) -> np.ndarray:
    """Finds the geometry (contours) of shapes in the image.

    Parameters
    ----------
    only_walls : np.ndarray
        Output of create_wall_corner_map.

    Returns
    -------
    np.ndarray
        Unordered geometry of the shapes in the input image.
    """
    only_walls_grey = cv2.cvtColor(only_walls, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(only_walls_grey, 0, 1, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    final_cnt = []

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), False)
        only_walls = cv2.drawContours(only_walls, [cnt], -1, (0, 0, 255), 3)
        final_cnt.append(approx[:, 0, :])

    final_cnt = np.array(final_cnt, dtype=object)

    # Plot the found contours
    for i in range(len(final_cnt)):
        data = np.append(final_cnt[i], final_cnt[i][0]).reshape(-1, 2)

        plt.plot(np.array(data)[:, 0], -np.array(data)[:, 1])

    plt.savefig("images/outputs/intermediate-outputs/contours.png")
    plt.clf()
    return final_cnt


def find_walls(contours: np.ndarray, corner_inds: np.ndarray) -> list:
    """Keep only shapes which have at least one point on the corner wall.

    Parameters
    ----------
    contours : np.ndarray
        Geometry of shapes in the image.
    corner_inds : np.ndarray
        Estimated wall corner locations.

    Returns
    -------
    list
        Contours which are adjacent to the esimated wall corners.
    """
    corner_adj_geom = []

    for i in range(len(contours)):
        data = np.array(contours[i])[:, 0]
        limit = 5

        for ind in corner_inds:
            diff = np.sum(np.abs(data.copy() - ind) <= limit)
            if diff >= 1:
                corner_adj_geom.append(contours[i].astype(int))

    # Plot contours
    for i in range(len(corner_adj_geom)):
        data = np.append(corner_adj_geom[i], corner_adj_geom[i][0]).reshape(-1, 2)
        plt.plot(np.array(data)[:, 0], -np.array(data)[:, 1])

    plt.savefig("images/outputs/intermediate-outputs/corner-contours.png")
    plt.clf()
    return corner_adj_geom


def make_edges_parallel(convex_hull: np.ndarray, width: int) -> np.ndarray:
    """Make the vertical edges of the polygon parallel.

    Parameters
    ----------
    convex_hull : np.ndarray
        Corner points of the polygon.
    width : int
        Width of the input image.

    Returns
    -------
    np.ndarray
        Corner points of the polygon.
    """
    temp_0 = convex_hull[0][0]
    temp_1 = convex_hull[1][0]
    temp_2 = convex_hull[2][0]
    temp_3 = convex_hull[3][0]
    biggest, smallest = [], []

    # For each corner, if the horizontal difference between it and the corner it's compared to is less than
    # 20% of the width of the picture, then set them to be the same
    for ind1 in range(0, 4):
        for ind2 in range(0, 4):
            if (
                abs(convex_hull[ind1][0] - convex_hull[ind2][0])
                <= (0.2 * width)
            ) and (convex_hull[ind1][0] != convex_hull[ind2][0]):
                # save both the largest and smallest of the two points
                biggest.append(
                    max(convex_hull[ind1][0], convex_hull[ind2][0])
                )
                smallest.append(
                    min(convex_hull[ind1][0], convex_hull[ind2][0])
                )
                # Temporarily set to the largest of the two points
                convex_hull[ind1][0] = biggest[-1]
                convex_hull[ind2][0] = biggest[-1]

    count = 0
    left = set()

    for x in range(0, 4):
        for y in range(0, 4):
            if abs(convex_hull[x][0] - convex_hull[y][0]) <= 20:
                count += 1
            if (
                convex_hull[x][0] < convex_hull[y][0]
            ):  # find which is the left side of the wall
                left.add(x)

    # if a point is on the left, set it to the min of the two points rather than the max
    if len(smallest) > 0:
        for x in range(0, 4):
            if x in left and convex_hull[x][0] in biggest:
                convex_hull[x][0] = min(smallest)

    # if all 4 points have been set to the same x-value, revert the change
    if count > 8:
        convex_hull[0][0] = temp_0
        convex_hull[1][0] = temp_1
        convex_hull[2][0] = temp_2
        convex_hull[3][0] = temp_3

    return convex_hull


def remove_duplicate_walls(geom: list) -> list:
    """Remove any duplicate walls.

    Parameters
    ----------
    geom : list
        List of walls.

    Returns
    -------
    list
        New list of walls with any duplicates removed.
    """
    new_geom = []

    if len(geom) > 0:
        # Remove any duplicate walls
        new_geom = [geom[0]]

        for i in geom:
            seen = False
            for j in new_geom:
                if np.array_equal(i, j):
                    seen = True
                    break

            if seen is False:
                new_geom.append(i)
    
    return new_geom


def find_quadrilaterals(corner_adj_geom: list, width: int) -> list:
    """Estimate quadrilaterals from the polygons already found.

    Parameters
    ----------
    corner_adj_geom : list
        Contours which are adjacent to the esimated wall corners.
    width : int
        Width of the input image.

    Returns
    -------
    list
        List of walls.
    """
    geom = []

    # for each shape
    for i in range(len(corner_adj_geom)):
        # try multiple thresholds
        for j in np.linspace(0.01, 0.1):
            # find convex hull
            convex_hull = cv2.convexHull(corner_adj_geom[i])
            # approximate the polygon
            convex_hull = cv2.approxPolyDP(
                convex_hull, j * cv2.arcLength(convex_hull, True), True
            )

            # if polygon has length of 4, keep it and break loop
            if len(convex_hull) == 4:
                convex_hull = convex_hull.reshape((4, 2))
                convex_hull = make_edges_parallel(convex_hull, width)
                geom.append(convex_hull)
                break

    new_geom = remove_duplicate_walls(geom)

    # Plot new contours
    for i in range(len(new_geom)):
        data = np.append(new_geom[i], new_geom[i][0]).reshape(-1, 2)
        plt.plot(np.array(data)[:, 0], -np.array(data)[:, 1])

    plt.savefig("images/outputs/intermediate-outputs/final-contours.png")
    plt.clf()
    return new_geom
