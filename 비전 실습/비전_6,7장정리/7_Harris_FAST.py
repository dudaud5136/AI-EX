# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:21:52 2024

@author: jyshin
"""

import cv2
import numpy as np

img = cv2.imread('D:/python/Vision/data/scenetext01.jpg', cv2.IMREAD_COLOR)

# 픽셀값의 차이가 큰 영상의 특징점(코너) 검출
corners = cv2.cornerHarris(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)
# 코너 팽창, 코너의 검출률 높히기
corners = cv2.dilate(corners, None)

show_img = np.copy(img)
show_img[corners > 0.1*corners.max()] = [0,0,255]

corners = cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
show_img = np.hstack((show_img, cv2.cvtColor(corners, cv2.COLOR_GRAY2BGR)))

cv2.imshow('Harris corner detector', show_img)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()

# 단순한 픽셀 값 비교 방법을 통해 코너 검출, 빠르게 작동
fast = cv2.FastFeatureDetector.create(30, True, cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
kp = fast.detect(img)

show_img = np.copy(img)
for p in cv2.KeyPoint.convert(kp):
    cv2.circle(show_img, tuple(int(i) for i in p), 2, (0, 255, 0), cv2.FILLED)
    
cv2.imshow('FAST corner detector', show_img)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()

# 비최대 억제는 이미지에서 여러 개의 인접한 픽셀이 동일한 특징을 가질 때, 
# 가장 강한 픽셀만을 선택하고 나머지는 억제하는 기법입니다. 이를 통해 특징점의 수를 줄이고, 
# 더욱 정확한 특징점을 검출할 수 있다.
# 비활성화되면 더 많은 코너 검출, 잘못된 검출 초래할 수 있음

fast.setNonmaxSuppression(False)
kp = fast.detect(img)

for p in cv2.KeyPoint.convert(kp):
    cv2.circle(show_img, tuple(int(i) for i in p), 2, (0, 255, 0 ), cv2.FILLED)

cv2.imshow('FAST corner detector', show_img)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()

    