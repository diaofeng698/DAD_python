#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2/17/2022 7:25 PM
# @Author  : FengDiao
# @Email   : diaofeng698@163.com
# @File    : cv_video.py.py
# @Describe:


import sys
import cv2

print(cv2.__version__)

video = cv2.VideoCapture("mix_gesture.avi")
if video.isOpened():
    print(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(video.get(cv2.CAP_PROP_FPS))

else:
    print("Get Camera Error ! ")
    sys.exit()

fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')

out = cv2.VideoWriter("live.mp4", fourcc, video.get(cv2.CAP_PROP_FPS), (640, 480), False)

while True:
    ret, frame = video.read()

    if ret:
        cv2.imshow("video", frame)
        out.write(frame)
    else:
        print("Finish")
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按键盘 'q' 退出
        break
        
        
video.release()
out.release()
cv2.destroyWindow("video")
