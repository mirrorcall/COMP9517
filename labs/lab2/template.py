# Template for lab02 task 3

import cv2
import math
import numpy as np
import sys

class SiftDetector():
    def __init__(self, norm="L2", params=None):
        self.detector=self.get_detector(params)
        self.norm=norm

    def get_detector(self, params):
        if params is None:
            params={}
            params["n_features"]=0
            params["n_octave_layers"]=3
            params["contrast_threshold"]=0.04
            params["edge_threshold"]=10
            params["sigma"]=1.6

        detector = cv2.xfeatures2d.SIFT_create(
                nfeatures=params["n_features"],
                nOctaveLayers=params["n_octave_layers"],
                contrastThreshold=params["contrast_threshold"],
                edgeThreshold=params["edge_threshold"],
                sigma=params["sigma"])

        return detector


# Rotate an image
#
# image: image to rotate
# x:     x-coordinate of point we wish to rotate around
# y:     y-coordinate of point we wish to rotate around
# angle: degrees to rotate image by
#
# Returns a rotated copy of the original image
def rotate(image, x, y, angle):
    scale = 1.0
    M = cv2.getRotationMatrix2D((x,y), -angle, scale)
    row, col = image.shape[:2]

    # sin and cos come from the affine transformation matrix
    # don't bother to refine them all over again
    # based on the ASSUMPTION: SCALE = 1.0 !!!!
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])

    n_col = int((row * sin) + (col * cos))
    n_row = int((row * cos) + (col * sin))

    M[0, 2] += (n_col / 2) - x
    M[1, 2] += (n_row / 2) - y

    return cv2.warpAffine(image, M, (n_col, n_row))


# Get coordinates of center point.
#
# image:  Image that will be rotated
# return: (x, y) coordinates of point at center of image
def get_img_center(image):
    row, col = image.shape[:2]
    return (col // 2, row // 2)


if __name__ == '__main__':
    # Read image with OpenCV and convert to grayscale
    img = cv2.imread('Eiffel_Tower.jpg', 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    params = {
            'n_features': 100,
            'n_octave_layers': 3,
            'contrast_threshold': 0.005,
            'edge_threshold': 10,
            'sigma': 1.6
    }
    sift = SiftDetector(params=params)

    # Store SIFT keypoints of original image in a Numpy array
    kp1, des1 = sift.detector.detectAndCompute(gray, None)
    kimg = cv2.drawKeypoints(gray, kp1, None)
    # cv2.imshow('demo', kimg)
    # wait = True
    # while wait:
        # wait = cv2.waitKey()=='q113' # hit q to exit
    # import sys
    # sys.exit(0)

    # Rotate around point at center of image.
    rot = rotate(gray, *get_img_center(gray), 45)
    # cv2.imshow('demo', rot)
    # wait = True
    # while wait:
        # wait = cv2.waitKey()=='q113' # hit q to exit
    # import sys
    # sys.exit(0)

    # Degrees with which to rotate image
    angle = 0
    degree = 45

    # Number of times we wish to rotate the image
    times = 3 
    
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    for x in range(times):

        # Rotate image
        rot = rotate(gray, *get_img_center(gray), angle)
        
        # Compute SIFT features for rotated image
        kp2, des2 = sift.detector.detectAndCompute(rot, None)
        
        # Apply ratio test
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])

        # cv2.drawMatchesKnn expects list of lists as matches.
        result = cv2.drawMatchesKnn(gray, kp1, rot, kp2, good, None, flags=2)
        cv2.imwrite('task3_'+str(angle)+'.jpg', result)
        cv2.imshow('Matches', result)
        wait = True
        while wait:
            wait = cv2.waitKey()=='q113' # hit q to exit

        angle += degree
