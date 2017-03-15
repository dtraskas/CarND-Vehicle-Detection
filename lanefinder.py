#
# Lane Detection 
# LaneFinder module tracks the lanes and determines curvature
#
# Dimitrios Traskas
#
#
import numpy as np
import cv2

class LaneFinder:
    
    def initialise(self, Minv):
        self.Minv = Minv

    # Determines the left and right peaks of pixel intensity in an image
    def find_peaks(self, image):
        histogram = np.sum(image[image.shape[0]/2:,:], axis=0)
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        return leftx_base, rightx_base

    # Returns the left and right lanes detected from the specified image
    def sliding_window(self, warped, leftx_base, rightx_base, nwindows=9):
        
        # Set height of windows
        window_height = np.int(warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = warped.shape[0] - (window+1)*window_height
            win_y_high = warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit, leftx, lefty, rightx, righty
    
    # Returns the lane curvature
    def get_curvature(self, image, left_fit, right_fit):
        
        # Define conversions in x and y from pixels space to meters        
        ym_per_pix = 30 / 720 # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700 # meters per pixel in x dimension

        ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
        leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

        # Calculate the new radii of curvature
        y_eval = np.max(ploty)
        left_radius = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_radius = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        offset = ( image.shape[1]/2 - (leftx[-1]+rightx[-1])/2)*xm_per_pix

        return left_radius, right_radius, offset

    # Returns the detected lane using the original undistorted image,
    # thresholded image and detected left and right lanes
    def get_lane(self, undistorted, masked, left_fit, right_fit):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(masked).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        ploty = np.linspace(0, masked.shape[0]-1, masked.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))                
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (masked.shape[1], masked.shape[0])) 
        
        # Combine the result with the original image
        final_image = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)        
        return final_image

    # Adds the stats to the final image
    def add_stats(self, image, left_r, right_r, offset):
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        text = "Left Radius: {:.0f} m".format(left_r)
        cv2.putText(image, text, (20,50), font, 1, (255,255,255), 2)

        text = "Right Radius: {:.0f} m".format(right_r)
        cv2.putText(image, text, (20,80), font, 1, (255,255,255), 2)

        text = "Centre Offset: {:.2f} m".format(offset)
        cv2.putText(image, text, (20,110), font, 1, (255,255,255), 2)

        return image