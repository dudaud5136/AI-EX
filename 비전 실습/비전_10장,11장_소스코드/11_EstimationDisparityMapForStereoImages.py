import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 20})

left_img = cv2.imread('../Data/stereo/left.png')
right_img = cv2.imread('../Data/stereo/right.png')

# 블록 매칭 알고리즘을 사용하여 스테레오 깊이 맵을 계산하는 객체를 생성
stereo_bm = cv2.StereoBM_create(32)
# 블록 매칭 알고리즘을 사용하여 스테레오 이미지 쌍에서 깊이 맵을 계산
dispmap_bm = stereo_bm.compute(cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY),
                               cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY))

# 반반 매칭 알고리즘을 사용하여 스테레오 깊이 맵을 계산하는 객체를 생성
stereo_sgbm = cv2.StereoSGBM_create(0, 32)
# 반반 매칭 알고리즘을 사용하여 스테레오 이미지 쌍에서 깊이 맵을 계산
dispmap_sgbm = stereo_sgbm.compute(left_img, right_img)

plt.figure(figsize=(12,10))
plt.subplot(221)
plt.title('left')
plt.imshow(left_img[:,:,[2,1,0]])
plt.subplot(222)
plt.title('right')
plt.imshow(right_img[:,:,[2,1,0]])
plt.subplot(223)
plt.title('BM')
plt.imshow(dispmap_bm, cmap='gray')
plt.subplot(224)
plt.title('SGBM')
plt.imshow(dispmap_sgbm, cmap='gray')
plt.show()