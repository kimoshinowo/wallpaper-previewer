import numpy as np
import cv2
import math
import PIL.Image as pil


def find_hypotenuse(wall_coords: np.ndarray) -> float:
    """Find length of the horizontal edge of the wall.

    Parameters
    ----------
    wall_coords : np.ndarray
        Corner coordinates of the wall.

    Returns
    -------
    float
        The hypotenuse ('width') of the wall.
    """
    ys = []
    for i in wall_coords:
        ys.append(i[1])
    smallest_inds = np.argpartition(ys, 2)[:2]
    smallest_ys = wall_coords[smallest_inds]
    hyp = math.sqrt(
        (smallest_ys[0][0] - smallest_ys[1][0]) ** 2
        + (smallest_ys[0][1] - smallest_ys[1][1]) ** 2
    )

    return hyp


def find_wall_height(wall_coords: np.ndarray) -> int:
    """Find the height of the wall (on longest edge).

    Parameters
    ----------
    wall_coords : np.ndarray
        Corner coordinates of the wall.

    Returns
    -------
    int
        Height of the wall.
    """
    y_coords = set()
    for i in wall_coords:
        y_coords.add(i[1])
    y_coords = list(y_coords)
    if len(y_coords) >= 2:
        y = abs(max(y_coords) - min(y_coords))
    else:
        y = 0

    return y


def order_corner_points(wall_coords: np.ndarray) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    wall_coords : np.ndarray
        Corner coordinates of the wall.

    Returns
    -------
    np.ndarray
        Corner coordinates of the wall, in the required order.
    """
    xs = []
    # find 2 smallest x
    for i in wall_coords:
        xs.append(i[0])
    xs = np.array(xs)
    largest_xs = [0, 1, 2, 3]
    smallest_xs = np.argpartition(xs, 2)[:2]
    [largest_xs.remove(smallest_xs[i]) for i in range(2)]

    # of those, find biggest y and smallest y
    left_points = wall_coords[smallest_xs]
    tl = left_points[np.argmax(left_points[:, 1])]
    bl = left_points[np.argmin(left_points[:, 1])]

    # find biggest y and smallest y of the 2 largest x values
    largest_xs = np.array(largest_xs)
    right_points = wall_coords[largest_xs]
    tr = right_points[np.argmax(right_points[:, 1])]
    br = right_points[np.argmin(right_points[:, 1])]

    ordered_points = np.array([bl, tl, tr, br])

    return ordered_points


def crop_wallpaper(
    height: int, width: float, repeat_div: int, temp_tiled: np.ndarray
) -> np.ndarray:
    """Crop the tiled wallpaper image.

    Parameters
    ----------
    height : int
        Height of the wall.
    width : int
        Width of the wall.
    repeat_div : int
        Relative width of the wallpaper.
    temp_tiled : np.ndarray
        Tiled wallpaper image.

    Returns
    -------
    np.ndarray
        Cropped wallpaper image.
    """
    y_ratio = height / repeat_div
    x_ratio = width / repeat_div
    y_percentage = y_ratio / math.ceil(height / repeat_div)
    x_percentage = x_ratio / math.ceil(width / repeat_div)
    tiled = temp_tiled[
        : round(y_percentage * temp_tiled.shape[0]),
        : round(x_percentage * temp_tiled.shape[1]),
    ]

    return tiled


def get_corners(tiled: np.ndarray) -> np.ndarray:
    """Gets an array containing the four corner points of the tiled wallpaper image.

    Parameters
    ----------
    tiled : np.ndarray
        Tiled wallpaper image.

    Returns
    -------
    np.ndarray
        The corner points of the input image.
    """
    corners = np.float32(
        [
            [0, 0],
            [0, tiled.shape[0]],
            [tiled.shape[1], tiled.shape[0]],
            [tiled.shape[1], 0],
        ]
    )

    return corners


def get_transformed_wallpaper(
    new_geom: list, height: int, width: int, size: tuple, wallpaper: np.ndarray
) -> "tuple[np.ndarray, np.ndarray]":
    """Create a wallpaper image where each wall is transformed to the correct size and shape.

    Parameters
    ----------
    new_geom : list
        List of points for each wall.
    height : int
        Height of the input room image.
    width : int
        Width of the input room image.
    size : tuple
        Size of the input room image.
    wallpaper : np.ndarray
        Wallpaper input image.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Transformed wallpaper image, and the simplified version.
    """
    wall_list_1, wall_list_2 = [], []
    input_corners = np.float32([[0, 0], [0, height], [width, height], [width, 0]])
    repeat_div = 100

    if len(new_geom) > 0:
        # Find and perform transform for each wall
        for wall in range(len(new_geom)):
            wall_coords = np.float32(new_geom[wall])
            hyp = find_hypotenuse(wall_coords)
            y = find_wall_height(wall_coords)
            wall_coords = order_corner_points(wall_coords)

            # Tile and crop the wallpaper
            temp_tiled = np.tile(
                wallpaper,
                (
                    max(math.ceil(y / repeat_div), 1),
                    max(math.ceil(hyp / repeat_div), 1),
                    1,
                ),
            )
            tiled = crop_wallpaper(y, hyp, repeat_div, temp_tiled)
            corners = get_corners(tiled)

            # Get and apply perspective transform
            matrix = cv2.getPerspectiveTransform(corners, wall_coords)
            temp_result = cv2.warpPerspective(tiled, matrix, size)
            wall_list_1.append(temp_result)

    # Tile and crop the wallpaper
    temp_tiled = np.tile(
        wallpaper, (math.ceil(height / repeat_div), math.ceil(width / repeat_div), 1)
    )
    tiled = crop_wallpaper(height, width, repeat_div, temp_tiled)
    corners = get_corners(tiled)

    # Get and apply perspective transform
    matrix = cv2.getPerspectiveTransform(corners, input_corners)
    temp_result = cv2.warpPerspective(tiled, matrix, size)
    wall_list_2.append(temp_result)

    result_2 = np.sum(wall_list_2, axis=0)

    # If there are no walls found, set both results to the default 'simple' version
    if len(wall_list_1) < 1:
        result_1 = result_2
    else:
        result_1 = np.sum(wall_list_1, axis=0)

    pil.fromarray(result_1.astype(np.uint8)).save("images/outputs/intermediate-outputs/wallpaper.png")

    return (result_1, result_2)


def get_wall_mask(new_geom: list, height: int, width: int, walls: np.ndarray) -> np.ndarray:
    """Get mask of walls.

    Parameters
    ----------
    new_geom : list
        List of points for each wall.
    height : int
        Height of the input room image.
    width : int
        Width of the input room image.
    walls : np.ndarray
        List of indices which are labelled as walls.

    Returns
    -------
    np.ndarray
        Mask of points labelled as wall and which are also part of the geometry.
    """
    # new wall indices
    geom_mask = []

    # Create mask of walls from contours by setting the walls to 1 and everything else to 0
    for contour in new_geom:
        single_contour_mask = []
        for i in range(height):
            row = []
            for j in range(width):
                # For each point, check if it is within (or on boundary of) the current contour, and set to 1 if it is
                t = cv2.pointPolygonTest(contour, tuple([j, i]), False)
                if t in [1, 0]:
                    row.append(1)
                else:
                    row.append(0)
            single_contour_mask.append(row)
        single_contour_mask = np.array(single_contour_mask)
        # Add current contour to the overall mask of the geometry
        geom_mask.append(single_contour_mask)

    if len(geom_mask) > 0:
        geom_mask = np.amax(geom_mask, axis=0)

    # Create mask of walls from segmentation map by setting the walls to 1 and everything else to 0
    wall_mask = np.zeros([height, width])
    wall_mask[walls[0], walls[1]] = 1

    if len(geom_mask) > 0:
        # Combine masks by only taking the points where both masks overlap
        final_mask = np.zeros([height, width])

        for i in range(height):
            for j in range(width):
                if wall_mask[i, j] + geom_mask[i, j] == 2:
                    final_mask[i, j] = 1
    else:
        final_mask = wall_mask

    return final_mask


def combine_wallpaper_and_input(
    input_cv2: np.ndarray, final_mask: np.ndarray, result_1: np.ndarray, result_2: np.ndarray, walls: np.ndarray
) -> "tuple[np.ndarray, np.ndarray]":
    """Combine transformed wallpaper image with input room image using masks.

    Parameters
    ----------
    input_cv2 : np.ndarray
        Input room image.
    final_mask : np.ndarray
        Mask of points labelled as wall and which are also part of the geometry.
    result_1 : np.ndarray
        Transformed wallpaper image.
    result_2 : np.ndarray
        Transformed wallpaper image (simple).
    walls : np.ndarray
        List of indices which are labelled as walls.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Final pipeline output images with wallpaper 'on' the walls.
    """
    # Create final image by setting mask indices to the wallpaper result produced earlier
    final_output = input_cv2.copy()
    final_output_2 = input_cv2.copy()

    mask_other_x, mask_other_y = np.where(final_mask == 1)
    final_output[mask_other_x, mask_other_y] = result_1[mask_other_x, mask_other_y]

    final_output_2[walls[0], walls[1]] = result_2[walls[0], walls[1]]

    return (final_output, final_output_2)
