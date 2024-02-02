import cv2
import numpy as np
import PIL.Image as pil
from matplotlib import pyplot as plt

def detect_edges(image):
    blur = cv2.blur(image, (3, 3)) # Add blur
    gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY) # Convert image to grayscale
    edges = cv2.Canny(gray,15,50,apertureSize=3) # Use canny method
    edge_map = pil.fromarray(edges)
    edge_map.save('images/outputs/edge-detection-output.png')

    return edge_map

def get_segmented_edges(edge_map, walls):
    # Combine edge map and segmentation map
    edge_map_array = np.asarray(edge_map.convert('RGB'))
    segmented_edges = np.empty( (edge_map_array.shape[0], edge_map_array.shape[1], 3) )
    segmented_edges[:] = np.nan
    segmented_edges[walls[0], walls[1]] = edge_map_array[walls[0], walls[1]]
    segmented_edges = segmented_edges.astype(dtype=np.uint8)

    pil.fromarray(segmented_edges).save('images/outputs/segmented-edges.png')

    return segmented_edges

def hough_transform(image):
    test = np.asarray(image)
    hough_img = np.empty( (test.shape[0], test.shape[1], 3) )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply HoughLinesP method to directly obtain line end points
    lines_list =[]
    lines = cv2.HoughLinesP(
                image, # Input edge image
                1, # Distance resolution in pixels
                np.pi/180, # Angle resolution in radians
                threshold=50, # Min number of votes for valid line
                minLineLength=25, # Min allowed length of line
                maxLineGap=15 # Max allowed gap between line for joining them
                )
    
    # Iterate over points
    for points in lines:
        # Extracted points nested in the list
        x1,y1,x2,y2=points[0]
        # Draw the lines joing the points on the original image
        cv2.line(hough_img,(x1,y1),(x2,y2),(0,0,255),2)
        # Maintain a simples lookup list for points
        lines_list.append([(x1,y1),(x2,y2)])

    cv2.imwrite('images/outputs/hough-output.png', hough_img)

    return hough_img

def get_vertical_lines(hough_img):
    # Get vertical lines only - https://www.youtube.com/watch?v=veoz_46gOkc
    kernel = np.ones((20,1), np.uint8)
    vertical_lines = cv2.erode(hough_img, kernel, iterations=1)
    cv2.imwrite('images/outputs/hough_corners.png', vertical_lines)
    corners = plt.imread('images/outputs/hough_corners.png')[:, :, :3] * 255
    
    return corners

def get_hough_corners(vertical_lines, colours):
    hough_corners = []

    # Find indices where there is colour
    _, hough_corners = np.where(colours == "255.0,0.0,0.0")

    return hough_corners
