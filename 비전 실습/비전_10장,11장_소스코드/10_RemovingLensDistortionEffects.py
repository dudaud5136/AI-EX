import cv2
import numpy as np

camera_matrix = np.load('C:/Users/dudau/Downloads/Vision/Data/data/pinhole_calib/camera_mat.npy')
dist_coefs = np.load('C:/Users/dudau/Downloads/Vision/Data/data/pinhole_calib/dist_coefs.npy')
img = cv2.imread('C:/Users/dudau/Downloads/Vision/Data/data/pinhole_calib/img_00.png')

cv2.imshow('original image', img)

# camera_matrix와 dist_coefs를 사용하여 왜곡 제거
ud_img = cv2.undistort(img, camera_matrix, dist_coefs)
cv2.imshow('undistorted image1', ud_img)

# 최적의 새로운 카메라 행렬(opt_cam_mat)과 왜곡이 제거된 이미지에서 실제 이미지 데이터가 있는 영역(valid_roi) 계산
# camera_matrix : 원본 카메라 행렬
# dist_coefs : 왜곡 계수
# img.shape : 이미지 크기
# 0 : 알파 값 (0이면 왜곡이 제거된 이미지에서 가능한 한 많은 빈 영역 제거, 1이면 빈 영역 유지)
opt_cam_mat, valid_roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, 
                                                       img.shape[:2][::-1], 0)

# 최적의 새로운 카메라 행렬으로 왜곡을 다시 제거
ud_img = cv2.undistort(img, camera_matrix, dist_coefs, None, opt_cam_mat)
cv2.imshow('undistorted image2', ud_img)

cv2.waitKey(0)
cv2.destroyAllWindows()