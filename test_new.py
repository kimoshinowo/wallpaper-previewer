import general_methods
import edge_detection
import numpy as np
import cv2
from matplotlib import pyplot as plt

def test_new(labels, input_cv2):
    other_x, other_y = np.where((labels != "0.47058824,0.47058824,0.47058824,1.0") & (labels != "0.47058824,0.47058824,0.3137255,1.0"))
    other = np.array([other_x, other_y])

    walls_x, walls_y = np.where(labels == "0.47058824,0.47058824,0.47058824,1.0")
    walls = np.array([walls_x, walls_y])

    input_copy = np.ones((input_cv2.shape[0], input_cv2.shape[1], 3))
    input_copy = input_copy.astype(float)
    input_copy[other[0], other[1]] = np.array([255, 255, 255], dtype=float)
    input_copy[walls[0], walls[1]] = np.array([255, 255, 255], dtype=float)
    input_copy = np.clip(input_copy, 0, 255)

    segmented_input = input_copy.astype(np.uint8)

    edge_map = edge_detection.detect_edges(segmented_input)
    edge_map = np.asarray(edge_map.convert("RGB"))

    # hough_img = edge_detection.hough_transform(edge_map, input_cv2.shape[0], input_cv2.shape[1])
    test = np.asarray(edge_map)
    hough_img = np.empty((test.shape[0], test.shape[1], 3))
    image = cv2.cvtColor(edge_map, cv2.COLOR_BGR2GRAY)

    # Apply HoughLinesP method to directly obtain line end points
    lines_list = []
    lines = cv2.HoughLinesP(
        image,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 180,  # Angle resolution in radians
        threshold=50,  # Min number of votes for valid line
        minLineLength=(0.03 * input_cv2.shape[0]),  # Min allowed length of line
        maxLineGap=(0.05 * input_cv2.shape[0]),  # Max allowed gap between line for joining them
    )

    # Iterate over points
    for points in lines:
        # Extracted points nested in the list
        x1, y1, x2, y2 = points[0]
        # Draw the lines joing the points on the original image
        cv2.line(hough_img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        # Maintain a simples lookup list for points
        lines_list.append([(x1, y1), (x2, y2)])

    cv2.imwrite("images/outputs/test-hough.png", hough_img)

    operatedImage = np.float32(hough_img)
    operatedImage = cv2.cvtColor(operatedImage, cv2.COLOR_BGR2GRAY)

    dest = cv2.cornerHarris(operatedImage, 30, 5, 0.07)
    dest = cv2.dilate(dest, None)  # Results are marked through the dilated corners

    # Reverting back to the original image, with optimal threshold value
    temp = hough_img.copy()
    temp[dest > 0.01 * dest.max()] = [255, 0, 0]

    labels = []
    for i in range(40):
        row = []
        for j in range(temp.shape[1]):
            row.append(",".join(temp[i, j].astype(str)))
        labels.append(row)

    labels = np.array(labels)

    red_x, red_y = np.where(labels == "255.0,0.0,0.0")
    red = np.array([red_x, red_y])

    temp2 = temp.copy()
    temp2[red[0], red[1]] = np.array([0, 0, 0], dtype=float)
    # plt.imshow(temp2)
    # plt.show()
    # plt.clf()

    ceiling_colours = general_methods.get_labels_string(temp2)
    _, ceiling_corners = np.where(ceiling_colours == "255.0,0.0,0.0")

    return ceiling_corners
