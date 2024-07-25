import cv2
import numpy as np

# 첫 번째, 두 번째 카메라의 투영 행렬 설정
P1 = np.eye(3, 4, dtype=np.float32)
P2 = np.eye(3, 4, dtype=np.float32)
# 첫 번째 카메라에 대해 x축 방향으로 -1만큼 이동한 위치에 있다고 가정
P2[0, 3] = -1

N = 5
# 3D 점을 저장할 배열 초기화
points3d = np.empty((4, N), np.float32)
# 3D 점들의 좌표를 무작위 값으로 설정
points3d[:3, :] = np.random.randn(3, N)
# 3D 점들의 homogeneous 좌표를 설정
points3d[3, :] = 1

# 각각 첫 번째 카메라에서 본 3D 점들의 2D 이미지 좌표 계산
points1 = P1 @ points3d
# 계산된 2D 이미지 좌표를 homogeneous 좌표로 변환
points1 /= points1[2, :]
# 2D 이미지 좌표에 약간의 노이즈를 추가
points1[:2, :] += np.random.randn(2, N) * 1e-2

points2 = P2 @ points3d
points2 /= points2[2, :]
points2[:2, :] += np.random.randn(2, N) * 1e-2

# 두 개의 이미지에서 본 2D 좌표를 사용하여 원래의 3D 점들을 재구성
points3d_reconstr = cv2.triangulatePoints(P1, P2, points1[:2], points2[:2])
# 재구성된 3D 점들의 좌표를 homogeneous 좌표로 변환
points3d_reconstr /= points3d_reconstr[3, :]

print('Original points')
print(points3d[:3].T)
print('Reconstructed points')
print(points3d_reconstr[:3].T)