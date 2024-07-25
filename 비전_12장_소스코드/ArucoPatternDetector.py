import cv2
import cv2.aruco as aruco
import numpy as np

# “board_type”의 DICT_6X6_250는 6x6은 Mark를 구분하기 위한 ID 정보를 표현하는 그리드의 가로세로 사이즈이며, 250은 ID 범위를 250개로 제안한다는 말이다.
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

img = np.full((700, 700), 255, np.uint8)

# drawMarkers => generateImageMarker로 변경 cv version 4.7.0이상
# 마커정보와 크기를 이용해서 마커 생성
img[100:300, 100:300] = aruco.generateImageMarker(aruco_dict, 2, 200)
img[100:300, 400:600] = aruco.generateImageMarker(aruco_dict, 76, 200)
img[400:600, 100:300] = aruco.generateImageMarker(aruco_dict, 42, 200)
img[400:600, 400:600] = aruco.generateImageMarker(aruco_dict, 123, 200)

img = cv2.GaussianBlur(img, (11, 11), 0)

cv2.imshow('Created AruCo markers', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# 마커 검출
corners, ids, _ = aruco.detectMarkers(img, aruco_dict)

img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
aruco.drawDetectedMarkers(img_color, corners, ids)

cv2.imshow('Detected AruCo Markers', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()