import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import time
import os
import random

import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

import copy

# ????????? ??Python??
map_list = ['safe drive', 'text-right', 'phone-talk-right', 'text-left', 
       'phone-talk-left', 'operate-radio', 'drink', 'reach-behind',
       'hair&makeup', 'talk-passenger']

gray_weights_path = 'weights_gray'

gray_model_load = tf.keras.models.load_model(gray_weights_path)

classes = 10  # ???DAD feature???
buffer_30 = {k:[0,0] for k in range(classes)}
print(buffer_30)


cap = cv2.VideoCapture('./video/phone.avi')
# ??????????,?? VideoCapture ??? 0 ? 1 ?????

fps = cap.get(cv2.CAP_PROP_FPS)
print(f'frame per second is {fps}')

num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(f'num frame is {num_frames}')

video_length = round(num_frames / fps)
print(f'video time length is {video_length}s')

frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
print(f'frame height is {frame_height}. width is {frame_width}')

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output_func3.mp4', fourcc, fps, (640, 480), False)
# ?????????????, ????, ???????, ???????(width, height),
# ?????isColor??, ???True, ????????,???????


dummy_img = cv2.imread('dummy_img.jpg')
_ = gray_model_load.predict(dummy_img[np.newaxis, ...])
#??????? warm up the model

# ????? C++?numpy????? ??opencv ? ???? ????,

buffer = []
result_list = []

time_2 = 2
time_10 = 4  #10
time_30 = 12  #30
#??????

frame_2s = time_2 * fps
frame_10s = time_10 * fps
frame_30s = time_30 * fps
#????fps???????? ???frame??

conf_thres = 0.5
state_previous = 255

count = 0

while cap.isOpened():
    ret, img = cap.read()
    if ret == False:
        print('read video failed or complete, break')
        break

    frame_now = cap.get(cv2.CAP_PROP_POS_FRAMES)
    #??????

    time_start = time.time()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = img.shape
    h = size[0]
    w = size[1]
    frame = copy.deepcopy(img)
    
    for i in range(h):
      for j in range(w):
          frame[i,w-1-j] = img[i,j]
      
    
    gray_img = cv2.resize(frame, (224, 224))
    gray_img = np.repeat(gray_img[..., np.newaxis], 3, -1)
 

    # ????? C++?numpy????? ??opencv ????
    
    gray_img_infer = gray_model_load.predict(gray_img[np.newaxis, ...])
    gray_img_pred = np.argmax(gray_img_infer)
    # ??????? np.argmax
    conf = gray_img_infer[0][gray_img_pred]
    # ?????? ??numpy np.squeeze

    time_end = time.time()
    time_cost = round(time_end - time_start, 3)

    state_now = gray_img_pred
    output_text = f'current frame is {frame_now}, infer result is {map_list[gray_img_pred]}, infer time is {time_cost}'
    print(output_text)    

    if (frame_now % frame_30s) != 0:
      if state_now == state_previous:
        if conf >= conf_thres:
          buffer.append((state_now, conf))
        img_text = ' P:' + map_list[gray_img_pred] + str(round(conf, 2))
      else:
        length = len(buffer)
        if length >= frame_2s:  
          print('buffer:', buffer) 
          buffer_30[state_previous][0] += length
          buffer_30[state_previous][1] += sum([item[1] for item in buffer])
          img_text = f'{map_list[state_previous]} last more than {time_2}s'
          print(img_text + ' ' + output_text)
        else:
          img_text = ' P:' + map_list[gray_img_pred] + str(round(conf, 2))
        buffer = []
        if conf >= conf_thres:
          buffer.append((state_now, conf))
      state_previous = state_now
    else:
      for category, value in buffer_30.items():
        if value[0] >= frame_10s:
          result_list.append((category, value[0], value[1]))
      if len(result_list) > 0:
        sorted_result = sorted(result_list, key=lambda item:item[1], reverse=True)
        #??????????alert, ????????,?????
        print('result:', sorted_result)
        img_text = f'alert in {time_30}s:{map_list[sorted_result[0][0]]}, conf:{round(sorted_result[0][2] / sorted_result[0][1], 2)} > {time_10}s'
        print(img_text + ' ' + output_text) 
      else:
        img_text = ' P:' + map_list[gray_img_pred] + str(round(conf, 2))
      result_list = []
      buffer_30 = {k:[0,0] for k in range(classes)}
      buffer = []
      if conf >= conf_thres:
        buffer.append((state_now, conf))
      state_previous = state_now


    cv2.putText(frame, img_text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    #out.write(frame)
    cv2.imwrite("./img/%d.jpg"%count, frame)
    count = count + 1
cap.release()


print('pred video is saved')