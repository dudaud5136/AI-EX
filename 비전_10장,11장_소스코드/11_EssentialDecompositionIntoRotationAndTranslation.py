import cv2
import numpy as np

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

data = np.load('../Data/stereo/case1/stereo.npy').item()
E = data['E']

# 필수 행렬을 분해하여 R1, R2, T 추출
R1, R2, T = cv2.decomposeEssentialMat(E)

print('Rotation 1:')
print(R1)
print('Rotation 2:')
print(R2)
print('Translation')
print(T)