import cv2
import numpy as np

camera_matrix = np.load('C:/Users/dudau/Downloads/Vision/Data/data/pinhole_calib/camera_mat.npy')
dist_coefs = np.load('C:/Users/dudau/Downloads/Vision/Data/data/pinhole_calib/dist_coefs.npy')

img = cv2.imread('C:/Users/dudau/Downloads/Vision/Data/Lena.png')
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

# 코너에 있는 점들을 camera_matrix와 dist_coefs를 사용하여 왜곡을 제거
h_corners = cv2.undistortPoints(corners, camera_matrix, dist_coefs)
# 왜곡이 제거된 점들에 대해 homogeneous 좌표를 생성
h_corners = np.c_[h_corners.squeeze(), np.ones(len(h_corners))]

# homogeneous 좌표를 이미지 평면에 투영
# 3D 점을 2D 이미지 평면에 매핑
img_pts, _ = cv2.projectPoints(h_corners, (0, 0, 0), (0, 0, 0), camera_matrix, None)

# 원본 이미지에 초록색 원을 그려서 원래의 코너 표시
for c in corners:
    cv2.circle(img, tuple(c[0].astype(int)), 10, (0, 255, 0), 2)

# 변형된 점들을 빨간색 원으로 이미지에 표지
for c in img_pts.squeeze().astype(np.float32):
    cv2.circle(img, tuple(c.astype(int)), 5, (0, 0, 255), 2)
    
cv2.imshow('undistorted corners', img)
cv2.waitKey()
cv2.destroyAllWindows()

img_pts, _ = cv2.projectPoints(h_corners, (0, 0, 0), (0, 0, 0), camera_matrix, dist_coefs)

for c in img_pts. squeeze().astype(np.float32):
    cv2.circle(img, tuple(c.astype(int)), 2, (255, 255, 0), 2)
    
cv2.imshow('reprojected corners', img)
cv2.waitKey()
cv2.destroyAllWindows()
