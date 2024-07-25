import cv2
import numpy as np
import os

pattern_size = (8, 6)
samples = []

file_list = os.listdir('C:/Users/dudau/Downloads/Vision/Data/data/fisheyes')

# img로 시작하는 이미지 파일 이름을 리스트로 저장
img_file_list = [file for file in file_list if file.startswith('Fisheye1_')]

# 이미지 파일 이름 리스트 순회
for filename in img_file_list:
    
    frame = cv2.imread(os.path.join('C:/Users/dudau/Downloads/Vision/Data/data/fisheyes', filename))
    # frame 이미지에서 코너를 찾아 리스트에 저장
    # res : 체스보드 패턴이 성공적으로 찾아졌는지를 나타내는 bool 값
    # corners : 찾아진 체스보드 코너들의 위치를 나타내는 2차원 포인트 배열
    res, corners = cv2.findChessboardCorners(frame, pattern_size)
    
    # 원본 이미지인 frame의 복사본을 만들어 img_show 변수에 저장
    img_show = np.copy(frame)
    # 체스보드 패턴의 코너를 img_show 이미지에 드로잉
    cv2.drawChessboardCorners(img_show, pattern_size, corners, res)
    # 현재 보고 있는 이미지의 인덱스를 나타내는 텍스트
    # (0, 40)은 텍스트가 나타날 위치
    # FONT_HERSHEY_SIMPLEX : 크키 = 1.0, 색상 = 녹색, 두께 = 2
    cv2.putText(img_show, 'Samples captured: %d' % len(samples), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0
                , (0, 255, 0), 2)
    cv2.imshow('chessboard', img_show)
    
    wait_time = 0 if res else 30
    k = cv2.waitKey(wait_time)
    
    if k == ord('s') and res:
        samples.append((cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), corners))
    elif k == 27:
        break

cv2.destroyAllWindows()

# 코너 세분화 알고리즘(이미지에서 코너를 더욱 정확하게 찾는 과정)의 종료 기준 설정
# 최대 반복횟수가 30이고, 정확도가 1e-3이면 종료
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

# 코너 세분화를 수행하는 루프 시작
for i in range(len(samples)):
    img, corners = samples[i]
    # 코너를 더욱 정밀하게 찾는 함수
    corners = cv2.cornerSubPix(img, corners, (10, 10), (-1, -1), criteria)

# 체스보드 패턴의 3D 월드 좌표를 저장할 배열 생성
pattern_points = np.zeros((1, np.prod(pattern_size), 3), np.float32)
# 체스보드 패턴의 각 코너에 대응하는 3D 좌표를 생성
pattern_points[0, :, :2] = np.indices(pattern_size).T.reshape(-1, 2)

# 이미지와 코너를 분리
images, corners = zip(*samples)

# 체스보드 패턴의 3D 월드 좌표를 각 코너에 복제
pattern_points = [pattern_points]*len(corners)

print(len(pattern_points), pattern_points[0].shape, pattern_points[0].dtype)
print(len(corners), corners[0].shape, corners[0].dtype)

# 카메라 왜곡 보정
# rms : 보정의 정확도 (값이 작을수록 보정이 더 정확)
# camera_matrix : 카메라의 내부 파라미터
# dist_coefs : 카메라의 렌즈 왜곡 계수
# rvecs, tvecs : 회전 벡터와 이동 벡터
rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.fisheye.calibrate(
    pattern_points, corners, images[0].shape, None, None)

np.save('camera_mat.npy', camera_matrix)
np.save('dist_coefs.npy', dist_coefs)

print(np.load('camera_mat.npy'))
print(np.load('dist_coefs.npy'))

