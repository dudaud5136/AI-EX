# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 21:49:32 2024

@author: jyshin
"""

# 이미지에 특정 변환을 주는 예제

import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('C:/Users/dudau/Downloads/Vision/Data/Lena.png')

# 매핑 행렬 생성
xmap = np.zeros((img.shape[1], img.shape[0]), np.float32)
ymap = np.zeros((img.shape[1], img.shape[0]), np.float32)

# 매핑 행렬 계산
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        xmap[y,x] = x + 30 * math.cos(20 * x / img.shape[0])
        ymap[y,x] = y + 30 * math.sin(20 * y / img.shape[1])
    
remapped_img = cv2.remap(img, xmap, ymap, cv2.INTER_LINEAR, None, cv2.BORDER_REPLICATE)

plt.figure(0)
plt.axis('off')
plt.imshow(remapped_img[:,:,[2,1,0]])
plt.show()