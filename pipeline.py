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

# Process just a single image
def process_image(image):
    draw_image = np.copy(image)  
    image = image.astype(np.float32)/255      
    hot_windows, all_windows = globalClf.search_multiple_scales(image)
    heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
    heatmap = globalClf.add_heatmap(heatmap, hot_windows)
    heatmap = globalClf.apply_threshold(heatmap,1)
    labels = label(heatmap)
    final_image = globalClf.draw_labeled_bboxes(draw_image, labels)
    
    return final_image

# Process the entire video
def process_video(inp_fname, out_fname):        
    clip = VideoFileClip(inp_fname)    
    output_clip = clip.fl_image(process_image)
    output_clip.write_videofile(out_fname, audio=False)

if __name__ == '__main__':
    
    cspace='RGB'
    spatial_size=(16,16)
    hist_bins=(32,32)
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL'
    
    # Specify here the processing phase
    phase = 'test'

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
        hot_windows, all_windows = clf.search_multiple_scales(image)
        heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
        heatmap = clf.add_heatmap(heatmap, hot_windows)
        heatmap = clf.apply_threshold(heatmap,1)
        final_heatmap = np.clip(heatmap, 0, 255)

        labels = label(final_heatmap)
        final_image = clf.draw_labeled_bboxes(np.copy(image), labels)
    
        plt.imshow(final_image)
        plt.show()

    if phase == 'generate':
        globalClf = VehicleClassifier(True, cspace, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel)
        print("Started processing video...")
        process_video('project_video.mp4', 'project_output.mp4')
        print("Completed video processing!")