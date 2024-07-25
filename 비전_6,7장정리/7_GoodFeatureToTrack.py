# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:38:21 2024

@author: jyshin
"""

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('D:/python/Vision/data/Lena.png', cv2.IMREAD_GRAYSCALE)

# shi-tomashi 알고리즘 Harris 를 기본으로 개선을 통해 더 정확한 코너 추출
# grayscale 이미지, 특징점 개수, 최소 품질(0~1), 최소 거리
corners = cv2.goodFeaturesToTrack(img, 100, 0.05, 10)

for c in corners:
    x, y = c[0]
    cv2.circle(img, (int(x), int(y)), 5, 255, -1)

plt.figure(figsize=(10, 10))
plt.imshow(img, cmap='gray')
plt.tight_layout()
plt.show()