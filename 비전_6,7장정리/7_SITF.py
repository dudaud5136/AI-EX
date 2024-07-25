# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:04:17 2024

@author: jyshin
"""

import cv2
import numpy as np

img0 = cv2.imread('D:/python/Vision/data/Lena.png', cv2.IMREAD_COLOR)
img1 = cv2.imread('D:/python/Vision/data/Lena_rotated.png', cv2.IMREAD_COLOR)
img1 = cv2.resize(img1, None, fx=0.75, fy=0.75)
img1 = np.pad(img1, ((64,)*2, (64,)*2, (0,)*2), 'constant', constant_values=0)
imgs_list = [img0, img1]

# SIFT 알고리즘, 크기 회전에 불변하는 특징점을 추출하는 알고리즘
detector = cv2.SIFT().create(50)


for i in range(len(imgs_list)):
    keypoints, descriptors = detector.detectAndCompute(imgs_list[i], None)
    
    imgs_list[i] = cv2.drawKeypoints(imgs_list[i], keypoints, None, (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
cv2.imshow('SIFT keypoints', np.hstack(imgs_list))
cv2.waitKey()

cv2.destroyAllWindows()