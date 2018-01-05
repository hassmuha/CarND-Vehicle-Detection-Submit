## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

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
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

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

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
