import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

os.chdir(os.getcwd())



# Read in the saved camera matrix and distortion coefficients
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Undistort image
def undistort_image(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

# Get binary image
def create_binary(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    #combined thresholds
    combined = np.zeros_like(s_binary)
    combined[(sxbinary == 1) | (s_binary == 1)] = 1
    #return combined
    return combined

# Perspective transform to an undistorted image
#define src
src = np.float32([[585, 460],
                  [203, 720],
                  [1127, 720],
                  [695, 460]])
#define dst
dst = np.float32([[320, 0],
                  [320, 720],
                  [960, 720],
                  [960, 0]])
Minv = cv2.getPerspectiveTransform(dst, src)
def warper(img, src, dst):
    # Compute and apply perpective transform
    # The img should be an undistorted image
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    return warped

def search_around_poly_refit(binary_warped):
    # Declare global variables
    global left_fit_pool, left_fit_avg, right_fit_pool, right_fit_avg
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    margin = 100
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    ### Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### consider the window areas for the similarly named variables ###
    ### in the previous, but change the windows to new search area ###
    left_lane_inds = ((nonzerox >= left_fit_avg[0]*nonzeroy**2+left_fit_avg[1]*nonzeroy+left_fit_avg[2]-margin) & 
    (nonzerox < left_fit_avg[0]*nonzeroy**2+left_fit_avg[1]*nonzeroy+left_fit_avg[2]+margin))
    right_lane_inds = ((nonzerox >= right_fit_avg[0]*nonzeroy**2+right_fit_avg[1]*nonzeroy+right_fit_avg[2]-margin) & 
    (nonzerox < right_fit_avg[0]*nonzeroy**2+right_fit_avg[1]*nonzeroy+right_fit_avg[2]+margin))
    # extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit new polynomials
    left_fit = np.polyfit(x=lefty, y=leftx, deg=2)
    right_fit = np.polyfit(x=righty, y=rightx, deg=2)
    # Check any large deviation and update fitting results
    left_conditions = np.absolute(left_fit-left_fit_avg)/left_fit_avg < 10
    #left_conditions = abs(np.log10(left_fit/left_fit_avg)) < 1.3
    if np.isin(False, left_conditions):
        left_fit = np.copy(left_fit_avg)
    left_fit_pool = update_fit(left_fit_pool, left_fit, holder=7)
    left_fit_avg = np.mean(left_fit_pool, axis=0)
    right_conditions = np.absolute(right_fit-right_fit_avg)/right_fit_avg < 10
    if np.isin(False, right_conditions):
        right_fit = np.copy(right_fit_avg)
    right_fit_pool = update_fit(right_fit_pool, right_fit, holder=7)
    right_fit_avg = np.mean(right_fit_pool, axis=0)
    #Generate x and y values for plotting
    img_shape = binary_warped.shape
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ###  Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    #return
    return left_fitx, right_fitx, ploty, left_fit, right_fit, leftx, rightx
    
def color_lane_region(undist, binary_warped, left_fitx, right_fitx, ploty, Minv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (binary_warped.shape[1], binary_warped.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result

def geometry(left_fitx, right_fitx, ploty, leftx, rightx):
    # Radius
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    ##### Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = (1+(2*left_fit[0]*y_eval+left_fit[1])**2)**(3.0/2)/abs(2*left_fit[0])  
    right_curverad = (1+(2*right_fit[0]*y_eval+right_fit[1])**2)**(3.0/2)/abs(2*right_fit[0])  
    # Meters
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    # Fit
    left_fit_cr = np.polyfit(ym_per_pix*ploty, xm_per_pix*leftx, 2)
    right_fit_cr = np.polyfit(ym_per_pix*ploty, xm_per_pix*rightx, 2)
    y_eval_cr = y_eval*ym_per_pix
    left_curverad_cr = (1+(2*left_fit_cr[0]*y_eval_cr+left_fit_cr[1])**2)**(3.0/2)/abs(2*left_fit_cr[0])
    right_curverad_cr = (1+(2*right_fit_cr[0]*y_eval_cr+right_fit_cr[1])**2)**(3.0/2)/abs(2*right_fit_cr[0])
    # Return
    return left_curverad, right_curverad, left_curverad_cr, right_curverad_cr
    
    

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

# Initiate left lane and right lane
left_lane = Line()
right_lane = Line()
