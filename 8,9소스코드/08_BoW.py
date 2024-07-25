# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 17:05:15 2024

@author: jyshin
"""

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('font', size=18)

# 두개의 이미지를 읽는다
img0 = cv2.imread('C:/Users/dudau/Downloads/Vision/Data/people.jpg', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('C:/Users/dudau/Downloads/Vision/Data/face.jpeg', cv2.IMREAD_GRAYSCALE)


# 읽은 이미지에 대해서 ORB 특징점 추출
detector = cv2.ORB().create(500)
_, fea0 = detector.detectAndCompute(img0, None)
_, fea1 = detector.detectAndCompute(img1, None)
descr_type = fea0.dtype

# 클러스터의 수를 50으로 해서 K-means 알고리즘 사용
# K-means 데이터를 K 개의 클러스터로 묶는 알고리즘
bow_trainer = cv2.BOWKMeansTrainer(50)
bow_trainer.add(np.float32(fea0))
bow_trainer.add(np.float32(fea1))
vocab = bow_trainer.cluster().astype(descr_type)

bow_descr = cv2.BOWImgDescriptorExtractor(detector, cv2.BFMatcher(cv2.NORM_HAMMING))
bow_descr.setVocabulary(vocab)

# BoW 계산
img = cv2.imread('C:/Users/dudau/Downloads/Vision/Data/Lena.png', cv2.IMREAD_GRAYSCALE)
kps = detector.detect(img, None)
descr = bow_descr.compute(img, kps)

plt.figure(figsize=(10,8))
plt.title('image BoW descriptor')
plt.bar(np.arange(len(descr[0])), descr[0])
plt.xlabel('vocabulary element')
plt.ylabel('frequency')
plt.tight_layout()
plt.show()