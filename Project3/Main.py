import cv2
import glob
import os
import scipy
import numpy as np

key_points_math_array = []


class KeyPoint:
    """
    Properties of cv2.KeyPoint:
    pt - coordinates of the keypoint
    size - diameter of the meaningful keypoint neighborhood
    angle - computed orientation of the keypoint. range [0,360) degrees.
    response - the response by which the most strong keypoints have been selected. Can be used for further sorting or subsampling
    octave - octave (pyramid layer) from which the keypoint has been extracted
    """
    def __init__(self, angle, dist, size, position):
        self.angle = angle
        self.distance = dist  # (module, vector, angle, center)
        self.size = size
        self.position = position


def calc_center(kp, gray):
    x_key_point, y_key_point = kp.pt[:2]
    height, weight = gray.shape[:2]
    x_center = weight / 2
    y_center = height / 2
    center_img = (x_center, y_center)

    xVector = x_center - x_key_point
    yVector = y_center - y_key_point
    vector = (xVector, yVector)

    module = scipy.sqrt(scipy.power((x_center - x_key_point), 2) + scipy.power((y_center - y_key_point), 2))

    if (y_center - y_key_point) == 0:
        angle = 0
    else:
        angle = scipy.arctan((x_center - x_key_point) / (y_center - y_key_point))

    distance_center = (module, vector, angle, center_img)
    return distance_center


def training_part1():
    os.chdir("./training")
    orb = cv2.ORB(nfeatures=100, nlevels=4, scaleFactor=1.3)
    # FLANN_INDEX_KDTREE = 1 - OK, FLANN_INDEX_LSH    = 6 - error.
    flann = cv2.FlannBasedMatcher(dict(algorithm=6), searchParams=dict())

    for img in glob.glob("*.jpg"):
        gray = cv2.imread(img, 0)
        key_points, descriptors = orb.detectAndCompute(gray, None)
        flann.add([np.uint8(descriptors)])
        flann.train()

        for kp in key_points:
            distance_center = calc_center(kp, gray)
            key_point = KeyPoint(kp.angle, distance_center, kp.size, kp.pt)
            key_points_math_array.append(key_point)

    return flann


def processing_part1(flann):
    os.chdir("../testing")
    orb = cv2.ORB(nfeatures=100, nlevels=4, scaleFactor=1.3)

    for img in glob.glob("*.jpg"):
        gray = cv2.imread(img, 0)
        key_points, query_descriptors = orb.detectAndCompute(gray, None)
        matches = flann.knnMatch(np.uint8(query_descriptors), k=5)
        vote_array = np.zeros((gray.shape[0]/10, gray.shape[1]/10), dtype=np.uint8)

        for match in matches:
            for index in match:
                scale = key_points_math_array[index.trainIdx].size / key_points[index.queryIdx].size
                xVector, yVector = key_points_math_array[index.trainIdx].distance[1]
                """
                vectorx = xVector * scale
                vectory = yVector * scale
                """
                # The formula given from the teacher to calculate the vectorX and vectorY is wrong.
                vectorx = (key_points[index.queryIdx].size * key_points_math_array[index.trainIdx].distance[0] *
                           scipy.cos(key_points_math_array[index.trainIdx].distance[2] + key_points[index.queryIdx].angle) -
                           key_points_math_array[index.trainIdx].distance[2]) / key_points_math_array[index.trainIdx].size

                vectory = (key_points[index.queryIdx].size * key_points_math_array[index.trainIdx].distance[0] *
                           scipy.sin(key_points_math_array[index.trainIdx].distance[2] + key_points[index.queryIdx].angle) -
                           key_points_math_array[index.trainIdx].distance[2]) / key_points_math_array[index.trainIdx].size

                kpx = int(scipy.divide(key_points[index.queryIdx].pt[0] + vectorx, 10))
                kpy = int(scipy.divide(key_points[index.queryIdx].pt[1] + vectory, 10))

                if (kpx > 0 and kpy > 0) and (kpx < vote_array.shape[1] and kpy < vote_array.shape[0]):
                    vote_array[kpy, kpx] += 1

        vote_array = cv2.resize(vote_array, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
        max_index = np.unravel_index(vote_array.argmax(), vote_array.shape)
        position = (max_index[0], max_index[1])
        image = cv2.imread(img)
        cv2.circle(image, position, image.shape[0]/33, (0, 0, 255), thickness=2)
        cv2.imshow("Car Center", image)
        cv2.waitKey(0)


def processing_part3(flann, gray_roi, gray_img):
    orb = cv2.ORB(nfeatures=100, nlevels=4, scaleFactor=1.3)
    key_points, query_descriptors = orb.detectAndCompute(gray_roi, None)
    matches = flann.knnMatch(np.uint8(query_descriptors), k=6)
    vote_array = np.zeros((gray_img.shape[0]/10, gray_img.shape[1]/10), dtype=np.uint8)

    for match in matches:
        for index in match:
            vectorx = (key_points[index.queryIdx].size * key_points_math_array[index.trainIdx].distance[0] *
                           scipy.cos(key_points_math_array[index.trainIdx].distance[2] + key_points[index.queryIdx].distance[2] -
                           key_points_math_array[index.trainIdx].distance[2])) / key_points_math_array[index.trainIdx].size

            vectory = (key_points[index.queryIdx].size * key_points_math_array[index.trainIdx].distance[0] *
                           scipy.sin(key_points_math_array[index.trainIdx].distance[2] + key_points[index.imgIdx].distance[2] -
                           key_points_math_array[index.trainIdx].distance[2])) / key_points_math_array[index.trainIdx].size

            kpx = int(scipy.divide(key_points[index.imgIdx].pt[0] + vectorx, 10))
            kpy = int(scipy.divide(key_points[index.imgIdx].pt[1] + vectory, 10))

            if (kpx > 0 and kpy > 0) and (kpx < vote_array.shape[1] and kpy < vote_array.shape[0]):
                vote_array[kpy, kpx] += 1

    vote_array = cv2.resize(vote_array, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
    max_index = np.unravel_index(vote_array.argmax(), vote_array.shape)
    position = (max_index[0], max_index[1])
    cv2.circle(gray_img, position, gray_img.shape[1]/33, (0, 0, 255), thickness=2)
    return gray_img


def car_detection_image_part2():
    os.chdir("./testing")
    car_detector = cv2.CascadeClassifier("../haar/coches.xml")
    roi_array = []

    for img in glob.glob("*.jpg"):
        gray = cv2.imread(img, 0)
        frontal_cars = car_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(120, 120),
                                                     flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

        for (x, y, w, h) in frontal_cars:
            roi = gray[y:y + h, x:x + w]
            roi_array.append(roi)

    for roi in roi_array:
        cv2.imshow("Car Detection Image", roi)
        cv2.waitKey(0)
    return np.asarray(roi_array, dtype=np.uint32)


def car_detection_image_part3(gray):
    car_detector = cv2.CascadeClassifier("../haar/coches.xml")
    roi_array = []

    frontal_cars = car_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(120, 120),
                                                 flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

    for (x, y, w, h) in frontal_cars:
        roi = gray[y:y + h, x:x + w]
        roi_array.append(roi)

    return np.asarray(roi_array, dtype=np.uint32)


if __name__ == '__main__':

    f = training_part1()
    processing_part1(f)

    car_detection_image_part2()

    flann = training_part1()  # esto es lo mismo que la parte 1
    """
        Notes..
        painted_image = cv2.drawKeypoints(gray, key_points, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("KeyPoints Training", painted_image)
        cv2.waitKey(0)
    """

    """
        For binary descriptors like BRIEF\ORB\FREAK you have to use either LSH or Hierarchical clustering index.
        cv2.flann_Index - LshIndexParams
        table_number the number of hash tables to use (between 10 and 30 usually).
        key_size the size of the hash key in bits (between 10 and 20 usually).
        multi_probe_level the number of bits to shift to check for neighboring buckets (0 is regular LSH, 2 is recommended).
    """

    """
       Trains a descriptor matcher (for example, the flann index). In all methods to match, the method train()
       is run every time before matching. Some descriptor matchers (for example, BruteForceMatcher) have an empty
       implementation of this method. Other matchers really train their inner structures (for example,
       FlannBasedMatcher trains flann::Index ).
    """

    """
        for match in matches:
            for index in match:
                print index.queryIdx, index.trainIdx, index.imgIdx, index.distance
        int queryIdx; // query descriptor index "actual image testing"
        int trainIdx; // train descriptor index "anterior image training"
        int imgIdx;   // train image index
        float distance;
    """

    """
        good = []
        pts1 = []
        pts2 = []

            # ratio test as per Lowe's paper
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.8*n.distance:
                    good.append(m)
                    pts2.append(kp2[m.trainIdx].pt)
                    pts1.append(kp1[m.queryIdx].pt)
    """