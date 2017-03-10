#
# Vehicle Classifier
# Classification system using labeled images of vehicles and not vehicles
#
# Dimitrios Traskas
#
#
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.utils import shuffle
import pickle

class VehicleClassifier:

    def __init__(self, loadmodel=True, cspace='RGB', spatial_size=(32,32), hist_bins=(32,32), 
                       orient = 9, pix_per_cell = 8, cell_per_block = 2, hog_channel = 'ALL'):                
        
        self.cspace = cspace
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel

        if loadmodel:
            self.svm = joblib.load('svm_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
        else:
            self.svm = LinearSVC()

    # Computes binned color features  
    def bin_spatial(self, img, size=(32, 32)):
        color1 = cv2.resize(img[:,:,0], size).ravel()
        color2 = cv2.resize(img[:,:,1], size).ravel()
        color3 = cv2.resize(img[:,:,2], size).ravel()
        return np.hstack((color1, color2, color3))
        
    # Computes histogram features
    def color_hist(self, img, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))        
        return hist_features    
    
    # Computes and returns HOG features
    # Call with two outputs if vis==True    
    def get_hog_features(self, image, vis=False, feature_vec=True):
        
        if vis == True:
            features, hog_image = hog(image, orientations=self.orient, 
                                    pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                    cells_per_block=(self.cell_per_block, self.cell_per_block), 
                                    transform_sqrt=False, 
                                    visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        else:      
            features = hog(image, orientations=self.orient, 
                        pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                        cells_per_block=(self.cell_per_block, self.cell_per_block), 
                        transform_sqrt=False, 
                        visualise=vis, feature_vector=feature_vec)
            return features

    # Extracts features from a single image
    def extract_single_features(self, image):        
        if self.cspace != 'RGB':
            if self.cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif self.cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif self.cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif self.cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        else: feature_image = np.copy(image)      
        
        spatial_features = self.bin_spatial(feature_image, size=self.spatial_size)
        hist_features = self.color_hist(feature_image)

        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)               
        if self.hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(self.get_hog_features(feature_image[:,:,channel], vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:            
            hog_features = self.get_hog_features(feature_image[:,:,self.hog_channel], vis=False, feature_vec=True)
                        
        return np.concatenate((spatial_features, hist_features, hog_features))
            
    # Extracts features from a list of images
    def extract_features(self, images):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in images:                        
            # Read using opencv
            image = mpimg.imread(file)
            features.append(self.extract_single_features(image))
            
        return features
        
    # Prepares all the image data by extracting features and normalizing them
    def prepare_data(self, vehicles, non_vehicles):
        
        # Extract all the features from vehicles and non-vehicles
        veh_features = self.extract_features(vehicles)
        notveh_features = self.extract_features(non_vehicles)
        
        # Create an array stack of feature vectors and fit a per-column scaler
        X = np.vstack((veh_features, notveh_features)).astype(np.float64)                        
        X_scaler = StandardScaler().fit(X)
        scaled_X = X_scaler.transform(X)        
        # Save the scaler when video processing
        joblib.dump(X_scaler, 'scaler.pkl')   

        # Define the labels vector
        y = np.hstack((np.ones(len(veh_features)), np.zeros(len(notveh_features))))

        return scaled_X, y

    # Train the SVM classifier using the images provided in the specified paths
    def train(self, veh_paths, nonveh_paths, split_size=0.2):
        
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
        
        # prepare data by extracting all the features
        X, y = self.prepare_data(vehicles, non_vehicles)        

        # Split up data into randomized training and test sets        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=np.random.randint(0, 100))
        X_train, y_train = shuffle(X_train, y_train)
        print('Training classifier...')

        # Check the training time for the SVC
        t=time.time()
        self.svm.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        print('Test Accuracy of SVC = ', round(self.svm.score(X_test, y_test), 4))
        print('Saving model...')
        joblib.dump(self.svm, 'svm_model.pkl')        
        print('Training complete!')        
    
    # Function that takes an image, start and stop positions in both x and y, window size 
    # (x and y dimensions) and overlap fraction (for both x and y)
    def slide_window(self, image, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64), xy_overlap=(0.5, 0.5)):

        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = image.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = image.shape[0]

        # Compute the span of the region to be searched    
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_windows = np.int(xspan/nx_pix_per_step) - 1
        ny_windows = np.int(yspan/ny_pix_per_step) - 1
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                
                window_list.append(((startx, starty), (endx, endy)))
        return window_list

    # Search the windows specified and run the classifier
    def search_windows(self, image, windows):
        # Create an empty list to receive positive detection windows
        on_windows = []
        # Iterate over all windows in the list
        for window in windows:
            # Extract the test window from original image
            test_img = cv2.resize(image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))     
            features = self.extract_single_features(test_img)            
            test_features = self.scaler.transform(np.array(features).reshape(1, -1))
            prediction = self.svm.predict(test_features)
        
            if prediction == 1:
                on_windows.append(window)
                
        return on_windows

    # Searches windows across multiple scales 
    def search_multiple_scales(self, image):        
        hot_windows = []
        all_windows = []
                
        X_start_stop =[[None,None],[None,None],[None,None],[None,None]]
        XY_window = [(240, 240), (180, 180), (120, 120), (70, 70)]                
        Y_start_stop =[[380, 500], [380, 470], [395, 455], [405, 440]]    
        
        for i in range(len(Y_start_stop)):
            windows = self.slide_window(image, x_start_stop=X_start_stop[i], y_start_stop=Y_start_stop[i], 
                                               xy_window=XY_window[i], xy_overlap=(0.75, 0.75))
            
            all_windows += [windows]                    
            hot_windows +=  self.search_windows(image, windows)

        return hot_windows,all_windows

    # Create a heatmap from the specified boxes
    def add_heatmap(self, heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        return heatmap

    # Applies a threshold to the specified heatmap to exclude false positives
    def apply_threshold(self, heatmap, threshold):
        heatmap[heatmap <= threshold] = 0
        return heatmap

    # Function that draws bounding boxes
    def draw_boxes(self, image, bboxes, color=(0, 0, 255), thick=6):
        imcopy = np.copy(image)
        for bbox in bboxes:
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        return imcopy

    # Returns an image with rectangles for all detected vehicles
    def draw_labeled_bboxes(self, image, labels):
        # Iterate through all detected vehicles
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))            
            cv2.rectangle(image, bbox[0], bbox[1], (0,0,255), 6)        
        return image