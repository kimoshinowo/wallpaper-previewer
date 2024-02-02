import numpy as np
import PIL.Image as pil
from mxnet import image as mimage
import cv2

def import_and_resize(filename):
    # Open image with PIL and resave to smaller size if needed (to stop CPU error)
    input_img = pil.open(filename)
    input_img.thumbnail((1000, 1000))
    input_img.save(filename)
    return input_img

def import_mx_image(filename):
    input_img = mimage.imread(filename)
    return input_img

def import_cv2_image(filename):
    input_img = cv2.imread(filename)
    return input_img

def get_labels_string(mmask):
    labels = []
    for i in range(mmask.shape[0]):
        row = []
        for j in range(mmask.shape[1]):
            row.append(",".join(mmask[i, j].astype(str)))
        labels.append(row)

    labels = np.array(labels)

    return labels

def find_wall_indices(labels):
    # Save the indices of every pixel that is part of the walls
    walls_x, walls_y = np.where(labels == "0.47058824,0.47058824,0.47058824,1.0")
    walls = np.array([walls_x, walls_y])
    return walls

def find_non_wall_indices(labels):
    # Save the indices of every pixel that isn't part of the walls
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

def get_matrix(line):
    y_round = np.round(line.copy(), 0).astype(int)
    y_round = y_round - np.nanmin(line)
    y_round = np.where(y_round < 0, 0, y_round)
    y_round = np.round(y_round, 0).astype(int)
    x = range(len(y_round))
    matrix = np.zeros((np.amax(y_round)+1, len(x), 3))

    for i in x:
        matrix[y_round[i], i] = [255, 255, 255]

    matrix = matrix.astype(np.uint8)
    pil.fromarray(matrix).save("images/outputs/matrix.png")

    return matrix
