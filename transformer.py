#
# Lane Detection 
# Transformer module takes an image and applies a perspective transform
#
# Dimitrios Traskas
#
#
import numpy as np
import cv2
import matplotlib.image as mpimg

class Transformer:
    
    def initialise(self, mtx, dist, M, Minv):
        self.mtx = mtx
        self.dist = dist
        self.M = M
        self.Minv = Minv
    
    # Returns the inverse perspective transform matrix
    def get_minv(self):
        return self.Minv

    # Returns an undistorted image using the calibration matrix and coefficients calculated 
    # during an earlier calibration stage
    def undistort(self, image):
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

    # Return a warped image by applying the perspective transform matrix to the specified image
    def warp(self, image):        
        return cv2.warpPerspective(image, self.M, (image.shape[1], image.shape[0]))
    
    # Return an unwarped image by applying the inverse perspective transform matrix to the specified image
    def unwarp(self, image):        
        return cv2.warpPerspective(image, self.Minv, (image.shape[1], image.shape[0]))

    # Calculates gradient magnitude
    def mag_thresh(self, image, sobel_kernel=3, mag_thresh=(0, 255)):
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        # Create a binary image of ones where threshold is met, zeros otherwise
        mag_binary = np.zeros_like(gradmag)
        mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        return mag_binary

    # Calculate gradient direction
    def dir_threshold(self, image, sobel_kernel=3, thresh=(0, np.pi/2)):       

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        
        # Take the absolute value of the gradient direction, 
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        dir_binary = np.zeros_like(absgraddir)
        dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        return dir_binary

    # Calculates directional gradient
    def abs_sobel_thresh(self, image, orient='x', sobel_kernel=3, thresh=(0, 255)):
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)    
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))

        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))        
        
        # Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))    
        # Create a mask of 1's where the scaled gradient magnitude 
        # is > thresh_min and < thresh_max
        grad_binary = np.zeros_like(scaled_sobel)
        grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return grad_binary
    
    # Applies the gradient and color thresholds on the selected channel
    def color_hls_threshold(self, image, hls=2, sobel_kernel=3, thresh_x=(0, 150), thresh_c=(0, 150)):

        channel = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)[:,:,hls]
        sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0)         
        # Absolute x derivative to accentuate lines away from horizontal
        abs_sobelx = np.absolute(sobelx) 
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh_x[0]) & (scaled_sobel <= thresh_x[1])] = 1
        
        # Threshold color channel
        s_binary = np.zeros_like(channel)
        s_binary[(channel >= thresh_c[0]) & (channel <= thresh_c[1])] = 1
                
        # Combine thresholds        
        color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
        return color_binary
    
    # Applies a gradient and color threshold to the specified image
    def color_grad_threshold(self, image, sobel_kernel=3, thresh_x=(0, 150), thresh_c=(0, 150)):
        
        #
        # Threshold x gradient on an RGB image converted to gray
        #
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)         
        # Absolute x derivative to accentuate lines away from horizontal
        abs_sobelx = np.absolute(sobelx) 
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        grad_binary = np.zeros_like(scaled_sobel)
        grad_binary[(scaled_sobel >= thresh_x[0]) & (scaled_sobel <= thresh_x[1])] = 1
        
        #
        # Threshold color channel on the saturation channel
        #
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
        l_channel = hsv[:,:,1]
        s_channel = hsv[:,:,2]
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= thresh_c[0]) & (s_channel <= thresh_c[1])] = 1

        # magnitude
        mag = self.mag_thresh(image, sobel_kernel, (30, 150))

        # RGB colour        
        R = image[:,:,2]
        G = image[:,:,1]
        B = image[:,:,0]

        thresh = (170, 255)
        binary = np.zeros_like(R)
        binary[(R > thresh[0]) & (R <= thresh[1])] = 1
                
        #
        # Combine thresholds
        #
        combined = np.zeros_like(s_binary)
        combined[(s_binary == 1) | (grad_binary == 1) | (mag == 1) | (binary == 1)] = 1
        return combined