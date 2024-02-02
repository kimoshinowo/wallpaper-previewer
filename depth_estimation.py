from transformers import pipeline
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
from transformers import DPTForDepthEstimation, DPTFeatureExtractor
import torch
import cv2
import numpy as np
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
import PIL.Image as pil

def estimate_depth(image):
    # checkpoint = "vinvino02/glpn-nyu"
    # depth_estimator = pipeline("depth-estimation", model=checkpoint)
    # predictions = depth_estimator(image)
    # depth_image = predictions["depth"]
    # plt.imsave('images/outputs/depth-output.png', depth_image)

    processor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
    model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

    # prepare image for the model
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # visualize the prediction
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    
    # model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
    # feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")

    # # prepare image for the model
    # inputs = feature_extractor(images=image, return_tensors="pt")

    # with torch.no_grad():
    #     outputs = model(**inputs)
    #     predicted_depth = outputs.predicted_depth

    # # interpolate to original size
    # prediction = torch.nn.functional.interpolate(
    #     predicted_depth.unsqueeze(1),
    #     size=image.size[::-1],
    #     mode="bicubic"
    # )

    # # visualize the prediction
    # output = prediction.squeeze().cpu().numpy()
    # formatted = (output * 255 / np.max(output)).astype("uint8")
    
    depth = pil.fromarray(formatted)

    plt.imsave('images/outputs/depth-output.png', depth)

    return depth

def get_mean_depths(depth_image, other):
    # Find the mean depth for each column (1px wide) in the wall depth map and plot
    test_depth = np.array(depth_image).copy().astype(float)
    test_depth[other[0], other[1]] = np.nan

    mean_depth = np.nanmean(test_depth, axis=0)
    # Smooth using savgol filter
    try:
        mean_depth = savgol_filter(mean_depth, 51, 3) # window size 51, polynomial order 3
    except:
        mean_depth = np.nanmean(test_depth, axis=0)

    return mean_depth

# https://stackoverflow.com/questions/47519626/using-numpy-scipy-to-identify-slope-changes-in-digital-signals
def savgol_corners(mean_depth):
    window = 55
    savgol = savgol_filter(mean_depth, window_length=window, polyorder=2, deriv=2)
    max_savgol = np.nanmax(np.abs(savgol))
    large = np.where(np.abs(savgol) > max_savgol/2)[0]
    gaps = np.diff(large) > window
    begins = np.insert(large[1:][gaps], 0, large[0])
    ends = np.append(large[:-1][gaps], large[-1])
    changes = ((begins+ends)/2).astype(int)

    return changes

def harris_corners(matrix):
    operatedImage = cv2.cvtColor(matrix, cv2.COLOR_BGR2GRAY) 
    operatedImage = np.float32(operatedImage) 

    dest = cv2.cornerHarris(operatedImage, 30, 5, 0.07)
    dest = cv2.dilate(dest, None) # Results are marked through the dilated corners 
    
    # Reverting back to the original image, with optimal threshold value 
    matrix[dest > 0.01 * dest.max()]=[255, 0, 0]
    
    pil.fromarray(matrix.astype(np.uint8)).save("images/outputs/corners.png")

    return matrix