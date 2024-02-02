import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

def get_transformed_wallpaper(new_geom, height, width, size):
    wallpaper = cv2.imread('images/inputs/wallpaper/check-even.jpg')
    wall_list_1, wall_list_2 = [], []
    input_corners = np.float32([[0,0], [0, height], [width, height], [width,0]])
    repeat_div = 80

    if len(new_geom) > 0:
        # Find and perform transform for each wall
        for wall in range(len(new_geom)):
            wall_coords = np.float32(new_geom[wall])

            xs = []
            # find 2 smallest x
            for i in wall_coords:
                xs.append(i[0])
            xs = np.array(xs)
            largest_xs = [0,1,2,3]
            smallest_xs = np.argpartition(xs, 2)[:2]
            [largest_xs.remove(smallest_xs[i]) for i in range(2)]

            # of those, find biggest y and smallest y
            left_points = wall_coords[smallest_xs]
            tl = left_points[np.argmax(left_points[:,1])]
            bl = left_points[np.argmin(left_points[:,1])]

            largest_xs = np.array(largest_xs)
            right_points = wall_coords[largest_xs]
            tr = right_points[np.argmax(right_points[:,1])]
            br = right_points[np.argmin(right_points[:,1])]

            # x_coords = set()
            # for i in wall_coords:
            #     x_coords.add(i[0])
            # x_coords = list(x_coords)
            # if len(x_coords) >= 2:
            #     # x = abs(x_coords[0] - x_coords[1])
            #     x = abs(max(x_coords) - min(x_coords))
            # else:
            #     x = 0

            ys = []
            for i in wall_coords:
                ys.append(i[1])
            smallest_inds = np.argpartition(ys, 2)[:2]
            smallest_ys = wall_coords[smallest_inds]
            hyp = math.sqrt((smallest_ys[0][0]-smallest_ys[1][0])**2 + (smallest_ys[0][1]-smallest_ys[1][1])**2)

            y_coords = set()
            for i in wall_coords:
                y_coords.add(i[1])
            y_coords = list(y_coords)
            if len(y_coords) >= 2:
                y = abs(max(y_coords) - min(y_coords))
            else:
                y = 0

            wall_coords = np.array([bl, tl, tr, br])
            # wall_coords = np.array([tl, br, tr, bl])

            # Tile the wallpaper
            temp_tiled = np.tile(wallpaper,(max(math.ceil(y/repeat_div), 1), max(math.ceil(hyp/repeat_div), 1), 1))
            y_ratio = y / repeat_div
            x_ratio = hyp / repeat_div
            y_percentage = y_ratio / math.ceil(y/repeat_div)
            x_percentage = x_ratio / math.ceil(hyp/repeat_div)
            tiled = temp_tiled[:round(y_percentage * temp_tiled.shape[0]), :round(x_percentage * temp_tiled.shape[1])]
            corners = np.float32([[0, 0], [0, tiled.shape[0]], [tiled.shape[1], tiled.shape[0]], [tiled.shape[1], 0]])

            # Get and apply perspective transform
            matrix = cv2.getPerspectiveTransform(corners, wall_coords)
            temp_result = cv2.warpPerspective(tiled, matrix, size)
            # plt.clf()
            # plt.imshow(temp_result)
            # plt.show()
            wall_list_1.append(temp_result)

    # Tile the wallpaper
    temp_tiled = np.tile(wallpaper, (math.ceil(height/repeat_div), math.ceil(width/repeat_div), 1))
    y_ratio = height / repeat_div
    x_ratio = width / repeat_div
    y_percentage = y_ratio / math.ceil(height/repeat_div)
    x_percentage = x_ratio / math.ceil(width/repeat_div)
    tiled = temp_tiled[0:round(y_percentage * temp_tiled.shape[0]), 0:round(x_percentage * temp_tiled.shape[1])]
    corners = np.float32([[0, 0], [0, tiled.shape[0]], [tiled.shape[1], tiled.shape[0]], [tiled.shape[1], 0]])
    # Get and apply perspective transform
    matrix = cv2.getPerspectiveTransform(corners, input_corners)
    temp_result = cv2.warpPerspective(tiled, matrix, size)
    wall_list_2.append(temp_result)

    result_2 = np.sum(wall_list_2, axis=0)

    if len(wall_list_1) < 1:
        result_1 = result_2
    else:
        result_1 = np.sum(wall_list_1, axis=0)
    
    # plt.clf()
    plt.imshow(result_1)
    plt.savefig('images/outputs/wallpaper.png')
    # plt.show()
    return result_1, result_2

def get_wall_mask(new_geom, height, width, walls):
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
                if wall_mask[i,j] + geom_mask[i,j] == 2:
                    final_mask[i,j] = 1
    else:
        final_mask = wall_mask
    
    return final_mask, wall_mask

def combine_wallpaper_and_input(input_cv2, final_mask, wall_mask, result_1, result_2, walls):
    # Create final image by setting mask indices to the wallpaper result produced earlier
    final_output = input_cv2.copy()
    final_output_2 = input_cv2.copy()

    mask_other_x, mask_other_y = np.where(final_mask == 1)
    final_output[mask_other_x, mask_other_y] = result_1[mask_other_x, mask_other_y]

    final_output_2[walls[0], walls[1]] = result_2[walls[0], walls[1]]

    # plt.imshow(cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB))

    return final_output, final_output_2