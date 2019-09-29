#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from utils import Line


# In[ ]:


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    """ Vertices are hard coded :3 """
    y, x = img.shape[:2]
    vertices = np.array([[(50, y), 
                      (450, 310), 
                      (490, 310),
                      (x-50, y)]], dtype="int32")
    
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    #draw_lines(line_img, lines)
    return lines

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)
  
#Some additional functions for smoothing predictions per frame
def smoothen_over_time(lane_lines):
    """
    Smooth the lane line inference over a window of frames and returns the average lines.
    """
    avg_left_lane = np.zeros((len(lane_lines), 4))
    avg_right_lane = np.zeros((len(lane_lines), 4))
    
    for t in range(0, len(lane_lines)):
      avg_left_lane[t] += lane_lines[t][0].get_coords()
      avg_right_lane[t] += lane_lines[t][1].get_coords()
      
    return Line(*np.mean(avg_left_lane, axis=0)), Line(*np.mean(avg_right_lane, axis=0))
  
def compute_lane_from_candidates(lines, img_shape):
    """
      Left lane = positive gradients/slope
      Right Lane = negative gradients/slope
      Below we are separating the left and right lane
    """
    pos_line = [l for l in lines if l.slope > 0]
    neg_line = [l for l in lines if l.slope < 0]

    neg_bias = np.median([l.bias for l in neg_line]).astype(int)
    neg_slope = np.median([l.slope for l in neg_line])
    x1, y1 = 0, neg_bias
    x2, y2 = -np.int32(np.round(neg_bias/neg_slope)), 0
    left_lane = Line(x1, y1, x2, y2)

    right_bias = np.median([l.bias for l in pos_line]).astype(int)
    right_slope = np.median([l.slope for l in pos_line])
    x1, y1 = 0, right_bias
    x2, y2 = np.int32(np.round((img_shape[0] - right_bias)/right_slope)), img_shape[0]
    right_lane = Line(x1, y1, x2, y2)

    return left_lane, right_lane

    
def get_lane_lines(image, solid_lines=True):  
    gray = grayscale(image)
    blur_img = gaussian_blur(gray, 5)

    y, x = image.shape[:2]

    edges = canny(gray, 255 * 1/3.0, 255)

    roi = region_of_interest(edges)

    detected_lines = hough_lines(img=roi, 
                                 rho=2, 
                                 theta=np.pi/180, 
                                 threshold=1, 
                                 min_line_len=15, 
                                 max_line_gap=5)

    detected_lines = [Line(l[0][0], l[0][1], 
                          l[0][2], l[0][3]) for l in detected_lines]

    if solid_lines:
      candidate_lines = []

      for line in detected_lines:
        # consider only lines having slope btwn 30 and 60
        if 0.5 <= np.abs(line.slope) <= 2.0:
          candidate_lines.append(line)
          
      lane_lines = compute_lane_from_candidates(candidate_lines, gray.shape)
    else:
      lane_lines = detected_lines

    return lane_lines

def color_frame_pipeline(frames, solid_lines=True, temporal_smoothing=True):
    is_videoclip = len(frames) > 0

    img_h, img_w = frames[0].shape[:2]

    lane_lines = []

    for t in range(0, len(frames)):
        inferred_lanes = get_lane_lines(frames[t], True)
        lane_lines.append(inferred_lanes)

    if temporal_smoothing and solid_lines:
        lane_lines = smoothen_over_time(lane_lines)
    else:
        lane_lines = lane_lines[0]

    line_img = np.zeros((img_h, img_w, 3)).astype("uint8")

    for lane in lane_lines:
        lane.draw(line_img, color=[0, 255, 0], thickness=10)

    img_masked = region_of_interest(line_img)


    img_color = frames[-1] if is_videoclip else frames[0]
    img_blend = weighted_img(img_masked, img_color, α=0.8, β=1., γ=-0.1)

    return img_blend


