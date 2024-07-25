import cv2
import numpy as np

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

data = np.load('../data/stereo/case1/stereo.npy').item()

K1, Kr, D1, Dr, left_pts, right_pts, E_from_stereo, F_from_stereo = \
    data['Kl'], data['Kr'], data['Dl'], data['Dr'], data['left_pts'], data['right_pts'], data['E'], data['F']

# 단일 배열로 재구성
left_pts = np.vstack(left_pts)
right_pts = np.vstack(right_pts)

# 왜곡 제거
left_pts = cv2.undistortPoints(left_pts, K1, D1, P=K1)
right_pts = cv2.undistortPoints(right_pts, Kr, Dr, P=Kr)

#왼쪽 및 오른쪽 포인트 사이의 기본행렬 계산
F, mask = cv2.findFundamentalMat(left_pts, right_pts, cv2.FM_LMEDS)

E = Kr.T @ F @ K1

print('Fundamental matrix:')
print(F)
print('Essential matrix:')
print(E)