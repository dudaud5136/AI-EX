import cv2
import matplotlib.pyplot as plt

image = cv2.imread('C:/Users/dudau/Downloads/Vision/Data/people.jpg')
# HOG는 보행자 검출을 위해 만들어진 특징 디스크립터
hog = cv2.HOGDescriptor()

# SVM분류기 계수를 setSVMDetector() 함수에 등록
# 64 x 128 크기의 윈도우에서 똑바로 서 있는 사람을 검출하는 용도로 훈련된 분류기 계수를 반환
hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())
# 입력 영상img 에서 다양한 크기의 객체 사각형 영역을 검출하고,
# 그 결과를 vector<Rect> 타입의 인자 foundLocations에 저장
locations, weights = hog.detectMultiScale(image)

dgb_image = image.copy()
for loc in locations:
    cv2.rectangle(dgb_image, (loc[0], loc[1]), (loc[0] + loc[2], loc[1] + loc[3]), (0, 255, 0), 2)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title('Original Image')
plt.axis('off')
plt.imshow(image[:, :, [2, 1, 0]])
plt.subplot(122)
plt.title('detections')
plt.axis('off')
plt.imshow(dgb_image[:, :, [2, 1, 0]])
plt.tight_layout()
plt.show()