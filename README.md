# CV - Computer Vision Projects
# #Python #OpenCV #ORB #FLANN #CLASSIFIER

# Project 3: Detecting a car and calculating the center of it.

Using KeyPoint object from OpenCV, Calculate the center, angle, vectors and module with the center of the image and the KeyPoint.

Using ORB (Oriented FAST and Rotated BRIEF) object from OpenCV, we are detecting the best 100 keypoint and they descriptors from each image.

By using FLANN (Fast Approximate Nearest Neighbor Search Library) we get efficiency because perform a quick matching. (Fast matcher), we use to get the best 5 Keypoint and descriptors.

We use Hough Transform technique to get the center of a car, by voting in matrix or array with the size of the (x=image/10, y=image/10), we make one vote with calculate center of the car and keypoints... At the end the maximum value of the matrix will be the coordinates where the center of the car is.

We train and test this object with the purpose of learn and get a better idea of this methods and techniques.

Also, we use a classifier from the OpenCV to detect cars. This is a generic classifier that can be improve in many ways but it helpful to get the job done faster and focus in other concepts.


# Project 4: Detecting a tag or plate in a car ROI and reading the letters/digits.

First, we use a OpenCV classifier for detecting cars. We change to a binary image to detect better letters and numbers with the function cv2.threshold. With the cv2.findContours function we look for areas (letters and digits) between 4 and 6 in a tag, these areas have to be very close and at the same height. We save every one of this areas in an array "plate".

Later, we use a trained object that recognize digits/letters to read all the plates detected in every image.

This projects are not finished completely but they help to learn more about different techniques, methods, filters and get a better idea of what is computer vision, OpenCV and Python.
