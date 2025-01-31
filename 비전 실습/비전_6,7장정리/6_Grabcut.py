# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 20:15:05 2024

@author: jyshin
"""

import cv2
import numpy as np

img = cv2.imread('D:/python/Vision/data/Lena.png', cv2.IMREAD_COLOR)
show_img = np.copy(img)

mouse_pressed = False
y = x = w = h = 0

# 사각형 공간 입력 받는 부분###################################
def mouse_callback(event, _x, _y, flags, param):
    global show_img, x, y, w, h, mouse_pressed
    
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        x, y = _x, _y
        show_img = np.copy(img)
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            show_img = np.copy(img)
            cv2.rectangle(show_img, (x, y), (_x, _y), (0, 255, 0), 3)
    
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False
        w, h = _x - x, _y - y        
############################################################

cv2.namedWindow('image')
cv2.setMouseCallback('image',mouse_callback)

while True:
    cv2.imshow('image', show_img)
    k = cv2.waitKey(1)
    
    if k == ord('a') and not mouse_pressed:
        if w * h > 0:
            break;

cv2.destroyAllWindows()

labels = np.zeros(img.shape[:2], np.uint8)
# 그랩 컷 적용, 사각형을 기준으로 배경, 전경으로 분할
labels, bgdModel, fgdModel = cv2.grabCut(img, labels, (x,y,w,h), None, None, 5, cv2.GC_INIT_WITH_RECT)

show_img = np.copy(img)
# 배경은 어둡게 표현
show_img[(labels == cv2.GC_PR_BGD) | (labels == cv2.GC_BGD)] //= 3

cv2.imshow('image',show_img)
cv2.waitKey()
cv2.destroyAllWindows()

label = cv2.GC_BGD
lbl_clrs = {cv2.GC_BGD: (0, 0, 0), cv2.GC_FGD: (255, 255, 255)}

# 배경 및 전경 추가 입력 받는 부분 ( 검은색은 배경, 흰색은 전경 ) 
def mouse_callback(event, x, y, flags , param):
    global mouse_pressed
    
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed  = True
        cv2.circle(labels, (x, y), 5, label, cv2.FILLED)
        cv2.circle(show_img, (x, y), 5, lbl_clrs[label], cv2.FILLED)
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            cv2.circle(labels, (x, y), 5, label, cv2.FILLED)
            cv2.circle(show_img, (x, y), 5, lbl_clrs[label], cv2.FILLED)
    
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False        
############################################################

cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)

while True:
    cv2.imshow('image', show_img)
    k = cv2.waitKey(1)
    
    if k == ord('a') and not mouse_pressed:
        break
    elif k == ord('l'):
        label = cv2.GC_FGD - label

cv2.destroyAllWindows()

# 그랩 컷 적용
labels, bgdModel, fgdModel = cv2.grabCut(img, labels, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

show_img = np.copy(img)
show_img[(labels == cv2.GC_PR_BGD) | (labels == cv2.GC_BGD)] //= 3

cv2.imshow('image',show_img)
cv2.waitKey()
cv2.destroyAllWindows()