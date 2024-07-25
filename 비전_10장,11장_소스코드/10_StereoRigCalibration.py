import cv2
import glob
import numpy as np

PATTERN_SIZE = (9, 6)
left_imgs = list(sorted(glob.glob('C:/Users/dudau/Downloads/Vision/Data/data/stereo/case1/left*.png')))
right_imgs = list(sorted(glob.glob('C:/Users/dudau/Downloads/Vision/Data/data/stereo/case1/right*.png')))
assert len(left_imgs) == len(right_imgs)

## 코너 세분화 알고리즘(이미지에서 코너를 더욱 정확하게 찾는 과정)의 종료 기준 설정
# 최대 반복횟수가 30이고, 정확도가 1e-3이면 종료
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
left_pts, right_pts = [], []
img_size = None

for left_img_path, right_img_path in zip(left_imgs, right_imgs):
    left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
    # 이미지 크기가 None인 경우, 왼쪽 이미지의 크기를 img_size로 저장
    if img_size is None:
        img_size = (left_img.shape[1], left_img.shape[0])
        
    # left_img or right_img에서 코너를 찾아 리스트에 저장
    # res : 체스보드 패턴이 성공적으로 찾아졌는지를 나타내는 bool 값
    # corners : 찾아진 체스보드 코너들의 위치를 나타내는 2차원 포인트 배열
    res_left, corners_left = cv2.findChessboardCorners(left_img, PATTERN_SIZE)
    res_right, corners_right = cv2.findChessboardCorners(right_img, PATTERN_SIZE)
    
    # 코너를 더욱 정밀하게 찾는 함수
    corners_left = cv2.cornerSubPix(left_img, corners_left, (10, 10), (-1, -1), criteria)
    corners_right = cv2.cornerSubPix(right_img, corners_right, (10, 10), (-1, -1), criteria)
    
    left_pts.append(corners_left)
    right_pts.append(corners_right)

# 체스보드 패턴의 3D 월드 좌표를 저장할 배열 생성
pattern_points = np.zeros((np.prod(PATTERN_SIZE), 3), np.float32)
# 체스보드 패턴의 3D 월드 좌표 설정 (각 코너는 x, y, 0 형태의 3D 좌표를 가짐)
pattern_points[:, :2] = np.indices(PATTERN_SIZE).T.reshape(-1, 2)
# 체스보드 패턴의 3D 월드 좌표를 각 코너에 복제
pattern_points = [pattern_points] * len(left_imgs)
    
# err : 보정의 정확도
# Kl : 왼쪽 카메라의 내부 파라미터를 나타내는 카메라 매트릭스
# Dl : 왼쪽 카메라의 렌즈 왜곡을 나타내는 왜곡 계수
# Kr : 오른쪽 카메라의 내부 파라미터를 나타내는 카메라 매트릭스
# Dr : 오른쪽 카메라의 렌즈 왜곡을 나타내는 왜곡 계수
# R : 왼쪽 카메라와 오른쪽 카메라 사이의 회전 행렬
# T : 왼쪽 카메라와 오른쪽 카메라 사이의 이동 벡터
# E : 두 카메라 사이의 회전과 이동
# F : 두 카메라의 픽셀 좌표 사이의 기하학적인 관계
err, K1, D1, Kr, Dr, R, T, E, F = cv2.stereoCalibrate(
    pattern_points, left_pts, right_pts, None, None, None, None, img_size, flags=0)

print('Left camera:')
print(K1)
print('Left camera distortion:')
print(D1)
print('Right camera:')
print(Kr)
print('Right camera distortion:')
print(Dr)
print('Rotation matrix:')
print(R)
print('Translation:')
print(T)

np.save('stereo.npy', {'Kl' : K1, 'Dl' : D1, 'Kr' : Kr, 'Dr': Dr, 'R':R, 'T':T, 'E':E
                       , 'F': F, 'img_size':img_size, 'left_pts': left_pts, 'right_pts': right_pts})