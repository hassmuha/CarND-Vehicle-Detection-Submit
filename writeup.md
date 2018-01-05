
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./result/dataset_example.jpg
[image2]: ./result/feature_example.jpg
[image3]: ./result/Normalization_example.jpg
[image4]: ./result/search_area.jpg
[image5]: ./result/test1_result.jpg
[image6]: ./result/test2_result.jpg
[image7]: ./result/test3_result.jpg
[image8]: ./result/test4_result.jpg
[image9]: ./result/test5_result.jpg
[image10]: ./result/test6_result.jpg
[image11]: ./result/heat_frame0.jpg
[image12]: ./result/heat_frame1.jpg
[image13]: ./result/heat_frame2.jpg
[image14]: ./result/heat_frame3.jpg
[image15]: ./result/heat_frame4.jpg
[image16]: ./result/heat_frame5.jpg
[image17]: ./result/frame15.jpg
[video1]: ./project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 2,3 and 4 code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

First I analyzed the images in different color spaces to extract features and among them HLS color space is selected.
The main motivation behind this is that I can use L-channel for extracting HOG features. From the L-channel the shape of the objects can be identified and HOG based features are ideal choice as they are robust against variations in shape and are also color invariant.
S-channel and H-channel are used to take color components into account. As both channels are not ideal for identifying shape so HOG based features does not work well. I used simple color histogram for these two channels. As cars appear in the scene always bright and clear so S-channel helps in identifying those characteristics. H-channel improves the accuracy by identifying known colors of the cars from the training samples.
HLS image example for car and non-car images are shown below:
![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried initially HOG and color histogram on 32x32 image size with following parameters for HOG: orient = 9; pix_per_cell = 8; cell_per_block = 2.

I also took following parameters for color histogram:
hist_bins=32; hist_range=(0, 256)

This contributes towards lower accuracy and on the video pipeline lot of false positives also start appearing which were hard to be filtered out. Then I changed it to HOG calculation on 64x64 image size and use the same parameters as discussed above. Moreover color histogram bins has also been changes to 64.

Based on the fact we are interested in saturated objects and last bins for H-channel appears to be zero, I discarded histogram bins as shown in the code snippet below:
`hist_features_S = hist_features_S[10:]`
`hist_features_H = hist_features_H[:-8]`

With the updated features the accuracy of the classifier on training set as well as on the on video pipeline improves a lot.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Before applying the classifier normalization is applied as the feature set is the combination of HOG and color histograms sets and without this normalization one or the other feature set start influencing. For Normalization, as discussed in the lecture I used `StandardScaler(...)` function.

You can see the example result for one car image below:
![alt text][image3]

Randomization of whole dataset and division between training and test set is done using the function `train_test_split(...)`. 20% of the samples are selected as test set. The code for these section can be found in section `Classification Preparation`.

I have tried two main classifier, SVM and Decision tree classifier. But SVM always do the best job. I

In order to tune the parameters for different combinations for the classifier I have used GridSearch method with following parameters
`'kernel':['linear', 'rbf']`
`'C':[2, 4, 6, 8]`
`'gamma':[.002,.004, .006, .008]`

Based on the results I choose kernel to be 'rbf' and C value to be 2. Default gamma also works quite well so I kept it like this. The code is show in the the cells starting from the section `Train the SVM classifier`

With these parameters I am able to achieve accuracy of around 99.5% on the test set.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to divide the whole search space in three main region. The Upper Region where the car appears small and normally not too much rectangular in shape. The Second Right Region where car appears bigger and often rectangular in shape. The last region take care of lower right region where car apears bigger in shape and again square shaped window will detect the car.

The detail of regions, window sizes for that region and search area is defined below:

Upper Region :
- Window size: 64x64, Search Area : starting location (x,y) = (500,400) and (width, height) = (780,128)
- Window size: 96x96, Search Area : starting location (x,y) = (500,400) and (width, height) = (780,128)
- Window size: 96x64, Search Area : starting location (x,y) = (500,400) and (width, height) = (780,128)

Right Region :
- Window size: 128x64, Search Area : starting location (x,y) = (1000,380) and (width, height) = (280,150)
- Window size: 128x128, Search Area : starting location (x,y) = (1000,380) and (width, height) = (280,300)
- Window size: 160x160, Search Area : starting location (x,y) = (1000,380) and (width, height) = (280,300)
- Window size: 192x128, Search Area : starting location (x,y) = (1000,380) and (width, height) = (280,250)

Lower left Region:
- Window size: 128x64, Search Area : starting location (x,y) = (500,500) and (width, height) = (550,180)
- Window size: 128x128, Search Area : starting location (x,y) = (500,500) and (width, height) = (550,180)
- Window size: 160x160, Search Area : starting location (x,y) = (500,500) and (width, height) = (550,180)


For reference the complete search space somehow looks like this
![alt text][image4]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

First of all for each window size, window space has been defined and following pipeline has been defined for finding cars in images:
Step 1 : Cropped the image based on the search area defined for one particular window
Step 2 : Conversion the cropped image into HLS space
Step 3 : Scale the whole cropped image to account for variation if the window size is different from 64 x 64
Step 4 : Calculate HOG features on complete L-channel once
Step 5 : Sliding the window on the entire cropped scaled image
  Step 5.1 : Extract HOG for this patch from the one calculated above
  Step 5.2 : Calculate color histogram on S-channel and H-channel patches
  Step 5.3 : Concatenate the features to one feature vector
  Step 5.4 : Scale features
  Step 5.5 : Make a prediction

Code is shown in the function `find_cars(...)` function.

ALl the windows with positive car presence have been combined to define the one big region for one car. But due to presence of false positives not all of the windows should be selected for final result. For filtering this I used the technique presented in the lectures to define heatmap based image and reject the regions where only few detected windows have been overlapped (non hot region). This is also shown below in the results. Final box around the car is drawn based on the size of one connected heated region.

The result on the test images are shown below:
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result][video1]


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I have defined vehicle object to record result from the N previous frames and make decision for the current frame. This helps not only rejecting false positive in each frame but also predict the location of car if in some frames no car is detected (true negative).

After applying threshold based filtering based on heatmap on each frame, the heatmap of N consecutive frames are combined. Then thresholded N-Frames combined heatmap to identify vehicle positions. I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  Based on the assumption that each blob corresponded to a vehicle, bounding boxes have been constructed to cover the area of each blob detected.

Very few false positive results have also been rejected by using the aspect ratio check (height/width). All blobs with aspect ratio more that 2.5 are also rejected.

### Here are first six frames and their corresponding heatmaps:

![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]

### Here the resulting bounding boxes are drawn onto the last frame when N = 15 in the series:
![alt text][image17]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Most of the time is spent in feature extraction phase. I tried different color spaces (YUV, HSL) and feature extraction mechanism to make the classifier color invariant and size invariant as much as possible. Tunning the HOG parameters was also challenging so that it can generalize better. For the selection of the classifier I focused on SVM mainly and challenge was to choose appropriate parameters for that.

In extreme light variations, HOG on L-channel can have influence where shape of the objects cannot be identified clearly, but color histogram on S-channel can serve as counter measure to this problem to some extent. As in the complete feature vector HOG features dominate so this can effect the whole pipeline.

The classification result of 99.5% also seems extremely good but depending upon the dataset, how much the dataset used to generalize all possible combination also has an impact.

In the current video pipeline, no effort has been done to use the pipeline for real time usage.
Currently it seems like too many features has been used and lot of windows slide with less step size. Windows step can be improved by introduction of randomness and appropriate window size can be selected by doing more effort on looking at the dataset and the shape/size of the car appears in the images.

Other techniques in case of detection and tracking is also useful like background subtraction or frame subtraction which gives first hint about the moving objects.

From the classification point of view appending the dataset with more examples will also be useful.
Other classifier e.g deep neural network can also improves the recognition rate and helps to solve the problem of generalization and appropriate feature extraction.
