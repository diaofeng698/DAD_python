import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import cv2
import numpy as np
import time
import os
import warnings
import cv2
import sys
warnings.filterwarnings('ignore')
classes = ['normal driving',
           'texting - right',
           'talking on the phone - right',
           'texting - left',
           'talking on the phone - left',
           'operating the radio',
           'drinking',
           'reaching behind',
           'hair and makeup',
           'talking to passenger'
           ]

# base_model = MobileNet(
#     include_top=False,
#     input_shape=(
#         224,
#         224,
#         3))
#
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
#
# preds = Dense(10, activation='softmax')(x)
# model = Model(inputs=base_model.input, outputs=preds)
#
# # print(model.summary())
#
# model.load_weights(
#     './weights/mobilenet_sgd_nolayers.hdf5')

# video = cv2.VideoCapture(0)
# if video.isOpened():
#     print(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#     print(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     print(video.get(cv2.CAP_PROP_FPS))
#
# else:
#     print("Get Camera Error ! ")
#     sys.exit()


model = tf.keras.models.load_model('./weights_gray')

# for item in os.listdir('./drive-download-20220215T074848Z-001'):
#     start_time = time.time() * 1000
#     img_name = "./drive-download-20220215T074848Z-001/" + item
#     img = cv2.imread(img_name)
#img = color.rgb2gray(img)
# img = img[50:, 120:-50]

img = cv2.imread("img.png")
print(img.shape)

resized_img = cv2.resize(img, (224, 224))
print(resized_img.shape)

gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
print(gray_img.shape)
input_img = np.repeat(gray_img[..., np.newaxis], 3, -1)
input_img = np.array(input_img).reshape(-1, 224, 224, 3)
print(input_img.shape)
# while True:
#     ret, img = video.read()
#
#     if ret:
#         cv2.imshow("video", img)
#         if cv2.waitKey(1) & 0xFF == ord('q'):  # 按键盘 'q' 退出
#             break
#
# img = cv2.resize(img, (224, 224))
# # img = tf.expand_dims(img, axis=0)
# img = np.array(img).reshape(-1, 224, 224, 3)
# print(img[0][0][0][0])

test_pred = model.predict(input_img)
print(test_pred)
# end_time = time.time() * 1000
# dur_time = end_time - start_time
#
# print(item + "inference time: %s  ms" % dur_time)
gray_img_pred = np.argmax(test_pred)
conf = test_pred[0][gray_img_pred]
print(conf)
print(np.argmax(test_pred, axis=1)[0])
print(classes[np.argmax(test_pred, axis=1)[0]])


# video.release()
# cv2.destroyWindow()
