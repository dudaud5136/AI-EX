import cv2
import numpy as np

def detect_face(video_file, detector, win_title):
    cap = cv2.VideoCapture(video_file)

    while True:
        status_cap, frame = cap.read()
        if not status_cap:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 특정 영상에서 내가 찾고자 하는 객체를 검출할 수 있습니다.
        #  입력 영상에서 멀티 스케일을 구성해서 내가 찾고자 하는 객체를 찾게 해줍니다.
        #  입력 영상 하나만 지정해도 동작합니다. 나머지는 디폴트 값으로 지정되어 있습니다.
        #  나머지 인자를 지정하면 빠르게 연산이 가능
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            text_size, _ = cv2.getTextSize('face', cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(frame, (x, y - text_size[1]), (x + text_size[0], y), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, 'face', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow(win_title, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
# 이 함수는 미리 학습되어 있는 정보를 불러와서 내가 찾고자 하는 객체를 검출하는 기능을 제공합니다.
#  filename으로 저장된 객체의 특징을 담고 있는 파일을 불러올 수 있습니다.
hear_face_cascade = cv2.CascadeClassifier('C:/Users/dudau/Downloads/Vision/Data/haarcascade_frontalface_default.xml')
detect_face('C:/Users/dudau/Downloads/Vision/Data/traffic.mp4', hear_face_cascade, 'Hear cascade face detector')


# 미리 학습된 XML 파일 다운로드 github
# github.com/opencv/opencv/tree/master/data/haarcascades
lbp_face_cascade = cv2.CascadeClassifier()
lbp_face_cascade.load('C:/Users/dudau/Downloads/Vision/Data/data/lbpcascade_frontalface.xml')


detect_face(0, hear_face_cascade, 'LBP cascade face detector')
