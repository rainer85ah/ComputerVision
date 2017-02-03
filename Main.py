import cv2
import glob
import os
import numpy as np


def car_tag_detection_image_part1():
    os.chdir("./testing_ocr/testing_ocr")  # 28 images
    car_detector = cv2.CascadeClassifier("C:\Users\Rainer\Documents\PycharmProjects\VisionArtificial\Practica4\haar\coches.xml")
    tags_array = []
    roi = None
    cont = 0

    for img in glob.glob("*.jpg"):  # 28 images
        gray = cv2.imread(img, 0)
        frontal_cars = car_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

        for (x, y, w, h) in frontal_cars:  # detect 27/28 img.. 1/30 fail
            roi = gray[y:y + h, x:x + w]
            retval, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            #  thresh = cv2.medianBlur(thresh, ksize=3)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            """
            # detect 21/27 img.. 1/5 fail
            cv2.drawContours(roi, contours, -1, (0, 0, 255), 4)
            cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow('gray rectangulo', gray)
            cv2.imshow('Roi Countrs', roi)
            cv2.waitKey(0)
            """
            tag = None

            for cnt in contours:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.03*peri, True)

                if 4 <= len(approx) <= 6:
                    x2, y2, w2, h2 = cv2.boundingRect(approx)
                    if h2 >= 5 and w2 >= 10:  # aspecto
                        area = cv2.contourArea(approx)
                        aspect_ratio = float(w2)/h2

                        if (2000 <= area <= 4500) or (2 <= aspect_ratio <= 6):  # detect 20/21.. 1/20 fail
                            tag = roi[y2:y2+h2, x2:x2+w2]
                            tags_array.append(tag)
                            cv2.imshow('TAG', tag)
                            cv2.waitKey(0)

    tags_array = np.asarray(tags_array, dtype=np.uint8)
    cv2.destroyAllWindows()
    return tags_array


def tags_detection():
    tags_array = []
    os.chdir("./testing_ocr/testing_ocr")  # 28 images
    car_detector = cv2.CascadeClassifier("C:\Users\Rainer\Documents\PycharmProjects\VisionArtificial\Practica4\haar\coches.xml")
    tag_detector = cv2.CascadeClassifier("C:\Users\Rainer\Documents\PycharmProjects\VisionArtificial\Practica4\haar\matriculas"
                                         ".xml")

    for img in glob.glob("*.jpg"):
        gray = cv2.imread(img, 0)
        frontal_cars = car_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

        for (x, y, w, h) in frontal_cars:
            roi = gray[y:y + h, x:x + w]
            tags = tag_detector.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=3, flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
            for (x1, y1, w1, h1) in tags:
                tag = roi[y1:y1 + h1, x1:x1 + w1]
                tags_array.append(tag)
                cv2.imshow('TAG', tag)
                cv2.waitKey(0)

    return np.asarray(tags_array, dtype=np.uint8)


if __name__ == '__main__':
    """ PART 1 : CAR & TAG Detection.
    This function return 20 tag from the 28 images, also return some ROI that are not tags.
    """
    tags = car_tag_detection_image_part1()

    """
    This functions use the haar classifier wih the matriculas.xml file. We need it to have the 28 tags from the images.
    Necessary for the rest of the project. But recognize 24 tags, and two are repeat it. (22 not so different from my 20)
    """
    # tags = tags_detection()