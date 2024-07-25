# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:25:05 2024

@author: jyshin
"""

import cv2
import numpy as np

# 비디오 로드 및 특징점 검출 객체(ORB) 생성
def video_keypoints(matcher, cap=cv2.VideoCapture('C:/Users/dudau/Downloads/Vision/Data/traffic.mp4'), detector=cv2.ORB().create(40)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        status_cap, frame = cap.read()
        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        if not status_cap:
            break
        if( cap.get(cv2.CAP_PROP_POS_FRAMES) - 1) % 40 == 0:
            key_frame = np.copy(frame)
            key_points_1, descriptors_1 = detector.detectAndCompute(frame,None)
        else:
            key_points_2, descriptors_2 = detector.detectAndCompute(frame,None)
            matches = matcher.match(descriptors_2, descriptors_1)
            frame = cv2.drawMatches(frame, key_points_2, key_frame, key_points_1, matches, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS | cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    
        cv2.imshow('Keypoints matching', frame)
        if cv2.waitKey(300) == 27:
            break
    
    cv2.destroyAllWindows()
    
# 두 프레임의 기술자 간 매칭    
bf_matcher = cv2.BFMatcher().create(cv2.NORM_HAMMING2, True)
video_keypoints(bf_matcher)

flann_kd_matcher = cv2.FlannBasedMatcher()
video_keypoints(flann_kd_matcher, detector=cv2.xfeatures2d.SURF().create(20000))

FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=20, key_size=15, multi_probe_level=2)
search_params = dict(checks=10)

flann_kd_matcher = cv2.FlannBasedMatcher(index_params, search_params)
video_keypoints(flann_kd_matcher)

FLANN_INDEX_COMPOSITE = 3
index_params = dict(algorithm=FLANN_INDEX_COMPOSITE, trees=16)
search_params = dict(checks=10)

flann_kd_matcher = cv2.FlannBasedMatcher(index_params, search_params)
video_keypoints(flann_kd_matcher, detector=cv2.xfeatures2d.SURF().create(20000))
