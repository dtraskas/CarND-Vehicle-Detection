#
# Vehicle Detection Pipeline
# The image processing pipeline that generates the final video
#
# Dimitrios Traskas
#
#

import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

from vehicle_classifier import VehicleClassifier
from transformer import Transformer 
from lanefinder import LaneFinder

# Definition of global Transformer and global LaneFinder used in the process image function
globalTransformer = Transformer()
globalLaneFinder = LaneFinder()
left_fit_buffer = None
right_fit_buffer = None
heatmap_history = []

'''
    global left_fit_buffer
    global right_fit_buffer

    undistorted = globalTransformer.undistort(image)
    warped = globalTransformer.warp(undistorted)
    masked = globalTransformer.color_grad_threshold(warped, sobel_kernel=9, thresh_x=(20, 100),thresh_c=(120, 255))
    left, right = globalLaneFinder.find_peaks(masked)
    left_fit, right_fit, leftx, lefty, rightx, righty = globalLaneFinder.sliding_window(masked, left, right)
    # take an average of previous frames to smooth the detection
    if left_fit_buffer is None:
        left_fit_buffer = np.array([left_fit])

    if right_fit_buffer is None:
        right_fit_buffer = np.array([right_fit])
        
    left_fit_buffer = np.append(left_fit_buffer, [left_fit], axis=0)[-20:]
    right_fit_buffer = np.append(right_fit_buffer, [right_fit], axis=0)[-20:]

    left_fit = np.mean(left_fit_buffer, axis=0)
    right_fit = np.mean(right_fit_buffer, axis=0)

    final_result = globalLaneFinder.get_lane(undistorted, masked, left_fit, right_fit)
    left_r, right_r, offset = globalLaneFinder.get_curvature(masked, left_fit, right_fit)    
    final_result = globalLaneFinder.add_stats(final_result, left_r, right_r, offset)
'''

# Process just a single image
def process_image(image):
    
    global left_fit_buffer
    global right_fit_buffer
    global heatmap_history

    draw_image = np.copy(image)  
    image = image.astype(np.float32)/255      
    detected_windows,_ = globalClf.search_multiple_scales(image)

    heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
    heatmap = globalClf.add_heatmap(heatmap, detected_windows)

    heatmap_history.append(heatmap)
    heatmap_history = heatmap_history[-10:]
    heatmap = np.vstack(heatmap_history)

    heatmap = globalClf.apply_threshold(heatmap,2)
    labels = label(heatmap)
    final_image = globalClf.draw_labeled_bboxes(draw_image, labels)
    
    undistorted = globalTransformer.undistort(final_image)
    warped = globalTransformer.warp(undistorted)
    masked = globalTransformer.color_grad_threshold(warped, sobel_kernel=9, thresh_x=(20, 100),thresh_c=(120, 255))
    left, right = globalLaneFinder.find_peaks(masked)
    left_fit, right_fit, leftx, lefty, rightx, righty = globalLaneFinder.sliding_window(masked, left, right)
    # take an average of previous frames to smooth the detection
    if left_fit_buffer is None:
        left_fit_buffer = np.array([left_fit])

    if right_fit_buffer is None:
        right_fit_buffer = np.array([right_fit])
        
    left_fit_buffer = np.append(left_fit_buffer, [left_fit], axis=0)[-20:]
    right_fit_buffer = np.append(right_fit_buffer, [right_fit], axis=0)[-20:]

    left_fit = np.mean(left_fit_buffer, axis=0)
    right_fit = np.mean(right_fit_buffer, axis=0)

    final_result = globalLaneFinder.get_lane(undistorted, masked, left_fit, right_fit)
    left_r, right_r, offset = globalLaneFinder.get_curvature(masked, left_fit, right_fit)    
    final_result = globalLaneFinder.add_stats(final_result, left_r, right_r, offset)

    return final_result

# Process the entire video
def process_video(inp_fname, out_fname):

    mtx = np.loadtxt("lane model/mtx.dat")
    dist = np.loadtxt("lane model/dist.dat")
    M = np.loadtxt("lane model/matrix.dat")
    Minv = np.loadtxt("lane model/matrix_inv.dat")

    globalTransformer.initialise(mtx, dist, M, Minv)
    globalLaneFinder.initialise(Minv)

    clip = VideoFileClip(inp_fname)    
    output_clip = clip.fl_image(process_image)
    output_clip.write_videofile(out_fname, audio=False)

if __name__ == '__main__':
    
    cspace='YUV'
    spatial_size=(16,16)
    hist_bins=(32,32)
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL'
    
    # Specify here the processing phase
    phase = 'generate'

    if phase == 'train':
        
        # Pass all the directories with training images
        vehicles = ['training_data/vehicles/GTI_Far/*.png', 'training_data/vehicles/GTI_Left/*.png',
                    'training_data/vehicles/GTI_Middle/*.png', 'training_data/vehicles/GTI_Right/*.png',
                    'training_data/vehicles/KITTI_extracted/*.png'
                   ]
        non_vehicles = ['training_data/non-vehicles/Extras/*.png', 'training_data/non-vehicles/GTI/*.png']

        # Initialise the vehicle classifier and all the parameters for extracting features
        clf = VehicleClassifier(False, cspace, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel)
        clf.train(vehicles, non_vehicles, 0.3)
    
    if phase == 'test':
        # Load a saved vehicle classifier and specify all the parameters for extracting features
        clf = VehicleClassifier(True, cspace, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel)
        # Set a test image and convert it since its a jpeg
        image = mpimg.imread('test_images/test1.jpg')
        image = image.astype(np.float32)/255        
        # Search all the scales and generate a heatmap
        detected_windows, all_windows = clf.search_multiple_scales(image)
        heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
        heatmap = clf.add_heatmap(heatmap, detected_windows)
        heatmap = clf.apply_threshold(heatmap,2)
        final_heatmap = np.clip(heatmap, 0, 255)

        labels = label(final_heatmap)
        final_image = clf.draw_labeled_bboxes(np.copy(image), labels)
    
        plt.imshow(final_image)
        plt.show()

    if phase == 'test_generate':
        globalClf = VehicleClassifier(True, cspace, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel)
        print("Started processing test video...")
        process_video('test_video.mp4', 'test_output.mp4')
        print("Completed test video processing!")

    if phase == 'generate':
        globalClf = VehicleClassifier(True, cspace, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel)         
        print("Started processing video...")
        process_video('project_video.mp4', 'project_output.mp4')
        print("Completed video processing!")