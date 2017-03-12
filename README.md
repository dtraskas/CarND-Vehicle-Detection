# Vehicle Detection

In order to implement a vehicle detection pipeline from a video stream the following steps have to be followed:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images.
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Normalize the image features and shuffle the data for training and testing.
* Train an SVM classifier
* Use a sliding-window technique and the trained classifier to search for vehicles in images.
* Run a detection pipeline on the video stream and generate a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/vehicle_not_vehicle.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/sliding_window.png
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_output.mp4

##Histogram of Oriented Gradients (HOG)

In order to train a classifier that detects vehicles from a video stream I utilised the datasets provided by Udacity for vehicles and non-vehicles. More specifically I used all the **GTI** and **KITTI** images provided and ensured that with shuffling and splitting of the data that there is no correlation between consecutive images which would result in a poor model. The code that reads the images and splits them into training and test datasets is within `vehicle_classifier.py` and more specifically the `train()` function. Below is an example of vehicle and non-vehicle images:

![alt text][image1]

I explored a number of different color spaces such as `RGB`, `HLS`, `YUV` and `YCrCb` and HOG parameters. In the end I ended up using the `YUV` color space, `orientation=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` which are set at the beginning of my processing pipeline in `pipeliny.py`. For the HOG function I converted the original image to the `YCrCb` color space which was used only for this part of the feature extraction process. The HOG parameters were determined after a lot of experimentation with the entire pipeline and the SVM classifier.

You can see an example output of the HOG function below:

![alt text][image2]

##Training the SVM Classifier

A Support Vector Machine algorithm was used as the classifier for detecting vehicles and distinguishing them from non-vehicles. The training was performed within the `pipeline.py` training phase and using the `vehicle_classifier.py` module and `VehicleClassifier` class. The image data provided was in `png` format so it had to be scaled in the range of 0-255 before being used by the classifier. 

The classifier extracts spatial features and color histograms for all color channels and HOG features from the images provided. For the spatial feature extraction a `spatial_size=(16,16)` was used and for the color histograms I used `hist_bins=32`. All three feature extraction functions were utilised in the training and classification phases. A crucial part of the training process was the splitting of the data using the `train_test_split()` function from `sklearn.model_selection` and the shuffling of the data using `shuffle()` from `sklearn.utils`. All features were concatenated to one single vector with a total number of features of **5568** and then passed to the `StandardScaler` for normalization. The data was split to **70%** training and **30%** test datasets and once trained the classifier had an accuracy score of approximately **99%**. 

##Sliding Window Search

The processing pipeline uses the sliding window search technique in order to scan the lower part of the input image. Three different scales were utilised for this scanning process since vehicles might have different sizes depending on their distance from the mounted camera. The lower part of the image is used for the scanning process since the upper part is mainly the horizon. The multi-scale search function `search_multiple_scales()` is within the `vehicle_classifier.py` class.

The window sizes and an example image can be seen below:

| XY Window  | Y-Start-Stop  | 
|:----------:|:-------------:| 
| [128, 128] | [450, 600]    | 
| [96, 96]   | [400, 500]    |
| [64, 64]   | [400, 500]    |

![alt text][image3]

The function of `search_multiple_scales()` is to detect windows with vehicles. It does that by passing every single scale window to the SVM classifier which returns positive windows that contain a vehicle. The resulting windows are then used to create a heatmap. 

Below is an example image of the pipeline process and window detection:

![alt text][image4]

##Video Implementation

Once windows with vehicles and potentially some false positives are detected a heatmap is generated. Using a low threshold any false positives are removed and with the `scipy.ndimage.measurements.label()` function individual blobs are identified in the heatmap that are essentially vehicles. A bounding box is constructed to cover the area of each detected blob and highlighted in the video. 

Below you can see 6 frame examples and their corresponding heatmaps:

![alt text][image5]

Here is an output of the `scipy.ndimage.measurements.label()` on the integrated heatmap from the test frame:

![alt text][image6]

Here are the resulting bounding boxes drawn onto the same frame:

![alt text][image7]

And the final project output video can be seen here: [video output][video1]

##Discussion

The SVM classifier works well but the entire process of training it and generating all the features could be potentially replaced by a deep neural network. A deep learning approach would allow the classifier to learn all the distinct features of a vehicle in all sorts of different weather and light conditions, with different scaling and rotations. Such an approach would also be potentially faster in real-time processing. 

The advantage of course of an SVM classifier is that it's quite fast to train, doesn't require big volumes of data like a deep neural net would and it's easy to implement. There are some false positives on the final pipeline but with further fine-tuning of the classifier, using more images could be avoided.

Overall I would prefer to use a deep convolutional neural network for any future work on vehicle detection because I believe it would be a lot more accurate and would not have the issues I experienced with SVMs. It would be an approach that could generalise nicely over different vehicles, pedestrians, animals, objects and conditions.