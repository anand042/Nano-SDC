import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
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
    right_lane_slope = []
    left_lane_slope = []
    right_array = np.empty((0,4), dtype=int)
    left_array = np.empty((0,4), dtype=int)

    for line in lines:
        for x1,y1,x2,y2 in line:
          slope = -1*((y2-y1)*1.0)/((x2-x1)*1.0)
#          print x1,y1,x2,y2, slope
          if abs(slope) > math.tan(15*np.pi/180):
            if slope < 0:
              right_lane_slope = np.append(right_lane_slope, slope)
              right_array = np.vstack((right_array,[x1,y1,x2,y2]))
              cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            else:
              left_lane_slope = np.append(left_lane_slope, slope)
              left_array = np.vstack((left_array, [x1, y1, x2, y2]))
              cv2.line(img, (x1, y1), (x2, y2), color, thickness)   
#    print np.mean(right_lane_slope), np.mean(left_lane_slope)            
    avg_color = [0,0,255]
    avg_thickness = 5
    left_top_x = np.amax(left_array, axis=0)[2]
    left_top_y = np.amin(left_array, axis=0)[3]    
    left_bottom_x = int((-1280 + left_top_y)/(np.mean(left_lane_slope)) + left_top_x)

    right_top_x = np.amin(right_array, axis=0)[0]
    right_top_y = np.amin(right_array, axis=0)[1]
    right_bottom_x = int((-1280 + right_top_y)/(np.mean(right_lane_slope)) + right_top_x)

#    print right_top_y, left_top_y, right_top_x
    cv2.line(img, (left_top_x, left_top_y), (left_bottom_x, 1280), avg_color, avg_thickness)
    cv2.line(img, (right_top_x, right_top_y), (right_bottom_x, 1280), avg_color, avg_thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * alpha + img * beta + gamma
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)

#image_counter = 1

#Set parameters for Gaussian Blur
kernel_size = 5

#Define parameters for Canny
low_threshold = 75
high_threshold = 225

#Define parameters for Hough Transform
rho = 2
theta = 3*np.pi/180
threshold = 25
min_line_length = 25
max_line_gap = 10

#output_path = "test_images/test_images_output/"

#Clear all files in output directory

cap = cv2.VideoCapture('test_videos/challenge.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_file = cv2.VideoWriter('test_videos_output/challenge.mp4',fourcc,20.0,(960,540))

while (cap.isOpened()):
  ret, frame = cap.read()
  print frame.shape
  #Check to see if frame is valid
  if ret:
    #Convert image to grayscale
    gray = grayscale(frame)

    #Apply Gaussian blurring
    blur_gray = gaussian_blur(gray, kernel_size)

    #Canny edge detection
    edges = canny(blur_gray, low_threshold, high_threshold)

    #Set vertices for region of interest and apply region mask
    vertices = np.array([[(125,540),(445, 320), (515, 320), (frame.shape[1],frame.shape[0])]], dtype=np.int32)
    masked_image = region_of_interest(edges, vertices)

    #Apply Hough transform
    line_image = hough_lines(masked_image, rho, theta, threshold, min_line_length, max_line_gap)

    #Weight the image
    weighted_image = weighted_img(line_image, frame)
    cv2.imshow('frame', weighted_image)
    out_file.write(weighted_image)

  else:
    print('Error loading frame')
    break

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break


cap.release()
out_file.release()

#cv2.destroyAllWindows()



