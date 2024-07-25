# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 17:33:16 2024

@author: jyshin
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

# 이미지를 읽음
img0 = cv2.imread('C:/Users/dudau/Downloads/Vision/Data/Lena.png')

# 이미지를 회전
M = np.array([[math.cos(np.pi/12), -math.sin(np.pi/12), 0],
              [math.sin(np.pi/12), math.cos(np.pi/12), 0],
              [0,0,1]])
Moff = np.eye(3)
Moff[0,2] = -img0.shape[1]/2
Moff[1,2] = -img0.shape[0]/2
print(np.linalg.inv(Moff)@M@Moff)
img1 = cv2.warpPerspective(img0, np.linalg.inv(Moff)@M@Moff, (img0.shape[1], img0.shape[0]), borderMode=cv2.BORDER_REPLICATE)
# 회전한 이미지를 저장
cv2.imwrite('C:/Users/dudau/Downloads/Vision/Data/Lena_rotated.png', img1)

# 이미지, 회전 이미지를 읽음
img0 = cv2.imread('C:/Users/dudau/Downloads/Vision/Data/Lena.png', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('C:/Users/dudau/Downloads/Vision/Data/Lena_rotated.png', cv2.IMREAD_GRAYSCALE)

# 두 이미지의 ORB 특징점을 검출 
detector = cv2.ORB().create(100)
kps0, fea0 = detector.detectAndCompute(img0, None)
kps1, fea1 = detector.detectAndCompute(img1, None)

# 두 이미지 기술자 매칭( 유사한 점을 찾음 )
matcher = cv2.BFMatcher.create(cv2.NORM_HAMMING, False)
matches01 = matcher.knnMatch(fea0, fea1, k=2)
matches10 = matcher.knnMatch(fea1, fea0, k=2)

# 매칭 필터링( 매칭된 특징점 쌍 중 좋은 매칭을 선택 )
def ratio_test(matches, ratio_thr):
    good_matches = []
    for m in matches:
        ratio = m[0].distance / m[1].distance
        if ratio < ratio_thr:
            good_matches.append(m[0])
    return good_matches

RATIO_THR = 0.7
good_matches01 = ratio_test(matches01, RATIO_THR)
good_matches10 = ratio_test(matches10, RATIO_THR)

good_matches10 = {(m.trainIdx, m.queryIdx) for m in good_matches10}
final_matches = [m for m in good_matches01 if (m.queryIdx, m.trainIdx) in good_matches10]

dbg_img = cv2.drawMatches(img0, kps0, img1, kps1, final_matches, None)
plt.figure()
plt.imshow(dbg_img[:,:,[2,1,0]])
plt.tight_layout()
plt.show()