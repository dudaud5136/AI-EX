# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:48:27 2024

@author: jyshin
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img0 = cv2.imread('C:/Users/dudau/Downloads/Vision/Data\Lena.png', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('C:/Users/dudau/Downloads/Vision/Data\Lena_rotated.png', cv2.IMREAD_GRAYSCALE)

# ORB 객체 생성, 100개의 특징점 추출
detector = cv2.ORB().create(100)

# ORB를 사용해 두 이미지의 특징점, 기술자 계산
kps0, fea0 = detector.detectAndCompute(img0, None)
kps1, fea1 = detector.detectAndCompute(img1, None)

# 두 이미지를 매칭
matcher = cv2.BFMatcher().create(cv2.NORM_HAMMING, False)
matches = matcher.match(fea0, fea1)

# RANSAC 알고리즘을 사용해 호모그래피 계산
# 호모그래피 : 두 평면 사이의 투시변환 관계
pts0 = np.float32([kps0[m.queryIdx].pt for m in matches]).reshape(-1,2)
pts1 = np.float32([kps1[m.trainIdx].pt for m in matches]).reshape(-1,2)
H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0)

plt.figure()
plt.subplot(211)
plt.axis('off')
plt.title('all matches')
dbg_img = cv2.drawMatches(img0, kps0, img1, kps1, matches, None)
plt.imshow(dbg_img[:,:,[2,1,0]])
plt.subplot(212)
plt.axis('off')
plt.title('filtered matches')
dbg_img = cv2.drawMatches(img0, kps0, img1, kps1, [m for i, m in enumerate(matches) if mask[i]], None)
plt.imshow(dbg_img[:,:,[2,1,0]])
plt.tight_layout()
plt.show()