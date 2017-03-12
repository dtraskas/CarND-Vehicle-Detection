#
# Tests for Vehicle Detection Project
# Used to generate all the images required for the project 
#
# Dimitrios Traskas
#
#

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
from skimage.feature import hog
from vehicle_classifier import VehicleClassifier
from scipy.ndimage.measurements import label

def get_examples(veh_paths, nonveh_paths):
    # Read in vehicle images
    vehicles = []
    for veh_path in veh_paths:
        images = glob.glob(veh_path)            
        for image in images:
            vehicles.append(image)

    # Read in non-vehicle images
    non_vehicles = []
    for nonveh_path in nonveh_paths:
        images = glob.glob(nonveh_path)            
        for image in images:
            non_vehicles.append(image)
    
    veh = []
    for cnt in range(2):
        veh.append(mpimg.imread(np.random.choice(vehicles)))
    
    non_veh = []
    for cnt in range(2):
        non_veh.append(mpimg.imread(np.random.choice(non_vehicles)))

    return veh, non_veh

def gen_one():
    veh, non_veh = get_examples(['training_data/vehicles/GTI_Far/*.png'], ['training_data/non-vehicles/Extras/*.png'])

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(veh[0])
    plt.title('Vehicle')
    plt.subplot(122)
    plt.imshow(non_veh[1])
    plt.title('Non Vehicle')    
    plt.show()
    fig.savefig("output_images/vehicle_not_vehicle.png")

def gen_two():
    veh_images, nonveh_images = get_examples(['training_data/vehicles/GTI_Far/*.png'], ['training_data/non-vehicles/Extras/*.png'])    
    
    fig, axes = plt.subplots(4,7,figsize=(10,10))
    fig.subplots_adjust(hspace=0.3, wspace=0.1)
    for cnt in range(len(veh_images)):
        veh = veh_images[cnt]
        feature_veh = cv2.cvtColor(veh, cv2.COLOR_RGB2YUV)   

        axes[cnt,0].imshow(veh)    
        title = "veh-{0}".format(cnt)
        axes[cnt,0].set_title(title, fontsize = 14)
        axes[cnt,0].set_xticks([])
        axes[cnt,0].set_yticks([]) 

        for channel in range(3):    
            axes[cnt,channel+1].imshow(veh[:,:,channel],cmap='gray')
            title = "ch-{0}".format(channel)
            axes[cnt,channel+1].set_title(title, fontsize = 14)
            axes[cnt,channel+1].set_xticks([])
            axes[cnt,channel+1].set_yticks([]) 

        for channel in range(3):    
            feature_image = cv2.cvtColor(veh, cv2.COLOR_RGB2YCrCb)   
            features, hog_image = hog(feature_image[:,:,channel], orientations=9, 
                                     pixels_per_cell=(8, 8),
                                     cells_per_block=(2, 2), 
                                     transform_sqrt=False, 
                                     visualise=True, feature_vector=True)

            axes[cnt,channel+4].imshow(hog_image,cmap='gray')
            title = "HOG ch-{0}".format(channel)
            axes[cnt,channel+4].set_title(title, fontsize = 14)
            axes[cnt,channel+4].set_xticks([])
            axes[cnt,channel+4].set_yticks([]) 
    
    index = 0
    for cnt in range(2,4):
        nonveh = nonveh_images[index]
        index += 1
        feature_veh = cv2.cvtColor(nonveh, cv2.COLOR_RGB2YUV)   

        axes[cnt,0].imshow(nonveh)    
        title = "nonveh-{0}".format(cnt)
        axes[cnt,0].set_title(title, fontsize = 14)
        axes[cnt,0].set_xticks([])
        axes[cnt,0].set_yticks([]) 

        for channel in range(3):    
            axes[cnt,channel+1].imshow(nonveh[:,:,channel],cmap='gray')
            title = "ch-{0}".format(channel)
            axes[cnt,channel+1].set_title(title, fontsize = 14)
            axes[cnt,channel+1].set_xticks([])
            axes[cnt,channel+1].set_yticks([]) 

        for channel in range(3):    
            feature_image = cv2.cvtColor(nonveh, cv2.COLOR_RGB2YCrCb)   
            features, hog_image = hog(feature_image[:,:,channel], orientations=9, 
                                     pixels_per_cell=(8, 8),
                                     cells_per_block=(2, 2), 
                                     transform_sqrt=False, 
                                     visualise=True, feature_vector=True)

            axes[cnt,channel+4].imshow(hog_image,cmap='gray')
            title = "HOG ch-{0}".format(channel)
            axes[cnt,channel+4].set_title(title, fontsize = 14)
            axes[cnt,channel+4].set_xticks([])
            axes[cnt,channel+4].set_yticks([]) 

    plt.show()
    fig.savefig("output_images/HOG_example.png")


def gen_three():
    cspace='YUV'
    spatial_size=(16,16)
    hist_bins=(32,32)
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL'

    clf = VehicleClassifier(True, cspace, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel)
    image = mpimg.imread('test_images/test1.jpg')
    image = image.astype(np.float32)/255            
    detected_windows, all_windows = clf.search_multiple_scales(image)
    
    phase = 5
    fig = plt.figure()    
    if phase == 1:
        for window in all_windows:
            for box in window:            
                cv2.rectangle(image, box[0], box[1], (250,250,25), 3)
        plt.imshow(image)
        plt.show()
        fig.savefig("output_images/sliding_windows.png")

    if phase == 2:
        for window in detected_windows:            
            cv2.rectangle(image, window[0], window[1], (250,250,25), 3)

        plt.imshow(image)        
        plt.show()
        fig.savefig("output_images/sliding_window.png")

    if phase == 3:
        images = ['test_images/test1.jpg', 'test_images/test2.jpg', 'test_images/test3.jpg', 
                   'test_images/test4.jpg','test_images/test5.jpg', 'test_images/test6.jpg']
        
        heatmaps = []
        for file in images:                   
            image = mpimg.imread(file)
            image = image.astype(np.float32)/255            
            detected_windows, all_windows = clf.search_multiple_scales(image)
        
            heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
            heatmap = clf.add_heatmap(heatmap, detected_windows)
            heatmap = clf.apply_threshold(heatmap,1)            
            heatmaps.append(np.clip(heatmap, 0, 255))
        
        fig, axes = plt.subplots(6,2,figsize=(10,10))
        fig.subplots_adjust(hspace=0.3, wspace=0.1)

        for cnt in range(len(heatmaps)):                
            axes[cnt,0].imshow(mpimg.imread(images[cnt]))
            title = "Test Image {0}".format(cnt+1)
            axes[cnt,0].set_title(title, fontsize = 14)
            axes[cnt,0].set_xticks([])
            axes[cnt,0].set_yticks([])     

            axes[cnt,1].imshow(heatmaps[cnt], cmap='gray')
            title = "Heatmap {0}".format(cnt+1)
            axes[cnt,1].set_title(title, fontsize = 14)
            axes[cnt,1].set_xticks([])
            axes[cnt,1].set_yticks([])     
        
        plt.show()
        fig.savefig("output_images/bboxes_and_heat.png")

    if phase == 4:
        heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
        heatmap = clf.add_heatmap(heatmap, detected_windows)
        heatmap = clf.apply_threshold(heatmap,1)
        final_heatmap = np.clip(heatmap, 0, 255)
        labels = label(final_heatmap)        
        plt.imshow(labels[0], cmap='gray')
        plt.show()
        fig.savefig("output_images/labels.png")

    if phase == 5:
        heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
        heatmap = clf.add_heatmap(heatmap, detected_windows)
        heatmap = clf.apply_threshold(heatmap,1)
        final_heatmap = np.clip(heatmap, 0, 255)
        labels = label(final_heatmap)
        final_image = clf.draw_labeled_bboxes(np.copy(image), labels)
    
        plt.imshow(final_image, cmap='gray')
        plt.show()
        fig.savefig("output_images/output_bboxes.png")
    
if __name__ == '__main__':

    gen_two()
