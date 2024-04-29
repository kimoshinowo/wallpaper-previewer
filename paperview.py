import numpy as np
import cv2
import PIL.Image as pil
import sys
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import warnings
# Import custom methods
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import semantic_segmentation
import general_methods
import edge_detection
import depth_estimation
import geometry
import transforms

def depth_and_edge_corners(input_pil, input_cv2, walls, height, width, other):
    # Depth estimation
    depth_image = depth_estimation.estimate_depth(input_pil)

    # Edge detection
    edge_map = edge_detection.detect_edges(input_cv2)
    segmented_edges = edge_detection.get_segmented_edges(edge_map, walls)

    # Find corners
    hough_img = edge_detection.hough_transform(segmented_edges, height, 50, 0.03, 0.02)
    vertical_lines = edge_detection.get_vertical_lines(hough_img)
    hough_colours = general_methods.get_labels_string(vertical_lines, vertical_lines.shape[0])
    hough_corners = edge_detection.get_hough_corners(hough_colours)
    mean_depth = depth_estimation.get_mean_depths(depth_image, other)
    matrix = general_methods.get_matrix(mean_depth)
    matrix_corners = depth_estimation.get_harris_corners(matrix)
    harris_colours = general_methods.get_labels_string(matrix_corners, matrix_corners.shape[0])
    _, harris_corners = np.where(harris_colours == "255,0,0")
    corner_inds = geometry.find_corners(hough_corners, harris_corners, width)

    return corner_inds

def corner_detection(labels, height, width, input_cv2, walls, input_pil, other):
    ceiling_x, _ = np.where(labels == "0.47058824,0.47058824,0.3137255,1.0")

    if ceiling_x.size > (0.01 * (height * width)):
        non_ceil_x, non_ceil_y = np.where((labels != "0.47058824,0.47058824,0.47058824,1.0") & (labels != "0.47058824,0.47058824,0.3137255,1.0"))
        non_ceil = np.array([non_ceil_x, non_ceil_y])

        input_copy = np.ones((input_cv2.shape[0], input_cv2.shape[1], 3))
        input_copy = input_copy.astype(float)
        input_copy[non_ceil[0], non_ceil[1]] = np.array([255, 255, 255], dtype=float)
        input_copy[walls[0], walls[1]] = np.array([255, 255, 255], dtype=float)
        input_copy = np.clip(input_copy, 0, 255)

        segmented_input_2 = input_copy.astype(np.uint8)

        for i in range(input_cv2.shape[1]):
            for j in range(input_cv2.shape[0]-1, -1, -1):
                if np.array_equal(segmented_input_2[j][i], [1, 1, 1]):
                    segmented_input_2[:j, i] = np.array([1, 1, 1], dtype=float)
                    break

        edge_map = edge_detection.detect_edges(segmented_input_2)
        edge_map = np.asarray(edge_map.convert("RGB"))

        hough_img = edge_detection.hough_transform(edge_map, input_cv2.shape[0], 10, 0.001, 0.5)
        corners = depth_estimation.get_harris_corners(hough_img)
        labels_red = general_methods.get_labels_string(corners, 40)
        red = general_methods.find_colour_indices(labels_red, "255.0,0.0,0.0")
        
        temp2 = corners.copy()
        temp2[red[0], red[1]] = np.array([0, 0, 0], dtype=float)

        ceiling_colours = general_methods.get_labels_string(temp2, temp2.shape[0])
        _, ceiling_corners = np.where(ceiling_colours == "255.0,0.0,0.0")
        corner_inds = geometry.find_corners(ceiling_corners, ceiling_corners, width)

        if corner_inds.size == 0:
            corner_inds = depth_and_edge_corners(input_pil, input_cv2, walls, height, width, other)
    else:
        corner_inds = depth_and_edge_corners(input_pil, input_cv2, walls, height, width, other)
    
    return corner_inds

def pipeline(filename, wallpaper_filename, corners = None):
    input_pil = general_methods.import_and_resize(filename)
    input_img = general_methods.import_mx_image("images/outputs/intermediate-outputs/resized-input.png")
    input_cv2 = general_methods.import_cv2_image("images/outputs/intermediate-outputs/resized-input.png")
    height = input_cv2.shape[0]
    width = input_cv2.shape[1]
    size = (width, height)

    # Segmentation
    seg_model = semantic_segmentation.get_pretrained_model()
    mmask = semantic_segmentation.get_segementation(input_img, seg_model)
    labels = general_methods.get_labels_string(mmask, mmask.shape[0])
    walls = general_methods.find_colour_indices(labels, "0.47058824,0.47058824,0.47058824,1.0")
    other = general_methods.find_not_colour_indices(labels, "0.47058824,0.47058824,0.47058824,1.0")
    segmented_input = semantic_segmentation.remove_inds(width, height, other)

    if corners is None:
        corner_inds = corner_detection(labels, height, width, input_cv2, walls, input_pil, other)
    else:
        corner_inds = np.array([int(corner) for corner in corners.split()])

    only_walls = geometry.create_wall_corner_map(segmented_input, other, walls, corner_inds)

    # Find room geometry
    contours = geometry.find_contours(only_walls)
    corner_adj_geom = geometry.find_walls(contours, corner_inds)
    new_geom = geometry.find_quadrilaterals(corner_adj_geom)
    new_corner_geom = geometry.remove_nested_geometry(new_geom)
    new_corner_geom = geometry.move_edges_to_corners(new_corner_geom, corner_inds, width)

    # Perspective transform
    wallpaper = general_methods.import_cv2_image(wallpaper_filename)
    result_1, result_2 = transforms.get_transformed_wallpaper(new_corner_geom, height, width, size, wallpaper)

    # Create final image
    final_mask, extra_mask = transforms.get_wall_mask(new_corner_geom, height, width, walls)
    final_output_1, final_output_2 = transforms.combine_wallpaper_and_input(input_cv2, final_mask, extra_mask, result_1, result_2, walls)
 
    return final_output_1, final_output_2


room_img_path = input("Please enter the path to your room image: ")
room_img = pil.open(room_img_path)
room_img.show()
correct = input("Is the image shown the correct image? (y/n): ")

if correct != 'y':
    sys.exit()

wallpaper_img_path = input("Please enter the path to your wallpaper sample image: ")
wallpaper_img = pil.open(wallpaper_img_path)
wallpaper_img.show()
correct = input("Is the image shown the correct image? (y/n): ")

if correct != 'y':
    sys.exit()

print("Pipeline is running, please wait...")

output, output_simple = pipeline(room_img_path, wallpaper_img_path)
cv2.imwrite("outputs/output.png", output)
# cv2.imwrite("output-simple.png", output_simple)
output_img = pil.open("outputs/output.png")
output_img.show()

improve = input("Would you like to answer a question to attempt to help improve the output? (y/n): ")

if improve != 'y':
    print("Thank you for using PaperView. Your ouput has been saved.")
else:
    graph_img_fig = mpimg.imread("images/outputs/intermediate-outputs/resized-input.png")
    plt.imshow(graph_img_fig)
    plt.xticks(list(range(0, output.shape[1], 50)), rotation = -45)
    plt.grid(color = 'red')
    plt.savefig("images/outputs/intermediate-outputs/graph_img.png")
    plt.clf()

    graph_img = pil.open("images/outputs/intermediate-outputs/graph_img.png")
    graph_img.show()

    corners_input = input("In the image shown, at approximately what x-values are the corners between the walls located?\n Please give a list of numbers, starting with the smallest, separated by a space. E.g. 350 720\n Enter numbers now: ")

    print("Pipeline is running again, please wait...")

    improved_output, _ = pipeline(room_img_path, wallpaper_img_path, corners_input)
    cv2.imwrite("outputs/improved-output.png", improved_output)
    # cv2.imwrite("output-simple.png", output_simple)
    output_img = pil.open("outputs/improved-output.png")
    output_img.show()

    print("Thank you for using PaperView. Your ouput has been saved.")
