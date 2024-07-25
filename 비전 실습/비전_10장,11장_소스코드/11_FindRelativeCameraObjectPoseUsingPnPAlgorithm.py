import cv2
import numpy as np

camera_matrix = np.load('../Data/pinhole_calib/camera_mat.npy')
dist_coefs = np.load('../Data/pinhole_calib/dist_coefs.npy')
img = cv2.imread('../Data/pinhole_calib/img_00.png')

pattern_size = (10, 7)
# img 이미지에서 코너를 찾아 리스트에 저장
# res : 체스보드 패턴이 성공적으로 찾아졌는지를 나타내는 bool 값
# corners : 찾아진 체스보드 코너들의 위치를 나타내는 2차원 포인트 배열
res, corners = cv2.findChessboardCorners(img, pattern_size)
# 코너 세분화 알고리즘(이미지에서 코너를 더욱 정확하게 찾는 과정)의 종료 기준 설정
# 최대 반복횟수가 30이고, 정확도가 1e-3이면 종료
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
# 코너를 더욱 정밀하게 찾는 함수
corners = cv2.cornerSubPix(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 
                           corners, (10, 10), (-1, -1), criteria)

# 체스보드 패턴의 3D 월드 좌표를 저장할 배열 생성
pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
# 체스보드 패턴의 3D 월드 좌표 설정 (각 코너는 x, y, 0 형태의 3D 좌표를 가짐)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)

# 주어진 3D-2D 점 쌍에 대해 카메라의 위치와 방향을 계산
# pattern_points : 3D 공간에서의 점들의 좌표
# camera_matrix : 카메라 내부 파라미터
# dist_coefs : 카메라 왜곡 계수
# cv2.SOLVEPNP_ITERATIVE : Leavenberg-Marquardt 최적화를 하여 #3D-2D 점 쌍에 대해 카메라 위치와 방향을 계산
ret, rvec, tvec = cv2.solvePnP(pattern_points, corners, camera_matrix, dist_coefs, 
                               None, None, False, cv2.SOLVEPNP_ITERATIVE)
# 주어진 3D 점들을 2D 이미지 평면에 투영
img_points, _ = cv2.projectPoints(pattern_points, rvec, tvec, camera_matrix, dist_coefs)

for c in img_points.squeeze():
    cv2.circle(img, tuple(c.astype(int)), 10, (0, 255, 0), 2)
    
cv2.imshow('points', img)
cv2.waitKey()

cv2.destroyAllWindows()