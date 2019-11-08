import numpy as np
import pandas as pd
import os
from Handwritten_recognition.utils import letters
from sklearn.utils import shuffle
from Handwritten_recognition.utils import *
import tensorflow as tf
import cv2
from Handwritten_recognition.preprocess_data import *
from Handwritten_recognition.model import *

# a = np.load("C:/data/hand_writing/input_np/train/per_column_uint8_h32/input_5.npy")
# a = a.T
# # print(a.shape)
# cv2.imshow("test", a)
# cv2.waitKey(0)


# folder = "D:/DATASET/words/a01/a01-000u/"
# list_img = []
# max_width = 0
# for file in os.listdir(folder):
#     a = cv2.imread(folder+file)
#     a = preprocessing_img(a)
#     h, w,  = a.shape[0], a.shape[1]
#     max_width = max(max_width, w)
#
#     a = cv2.rotate(a, cv2.ROTATE_90_CLOCKWISE)
#     list_img.append(a)
#     cv2.imshow("test", a)
#     cv2.waitKey(0)
#
# img_pad = sequence.pad_sequences(list_img, padding='post', value=0, dtype='uint8')
# print(img_pad[0].shape)
# print(max_width)
#
# for img in img_pad:
#     cv2.imshow("pad", img)
#     cv2.waitKey(0)


model = MyModel()
myModel = model.create_model_cnn_blstm()
myModel.load_model("../model_cnn_blstm/", Adam(lr=0.0001), init_archi=True,
                   file_weights="model_cnnblstm_sparse_h32_final_2.h5")

folder = "D:/DATASET/words/a01/a01-000u/"
for i in os.listdir(folder):
    list_input = []

    img = cv2.imread(folder + i)
    print(img.shape)

    img_input = preprocessing_img(img)
    h, w = img_input.shape[0], img_input.shape[1]
    left = 150 - h
    if left >= 0:
        zero_np = np.zeros((left, w))
        img_input = np.vstack((img_input, zero_np))
    else:
        img_input = cv2.resize(img_input, (32, 150))  # (w,h)

    img_input = normalize_img(img_input)
    img_input = np.expand_dims(img_input, axis=2)
    list_input.append(img_input)
    # dummy data
    list_input.append(np.zeros_like(img_input))

    print(img_input.shape)
    pred = myModel.predict([np.array(list_input), np.array([37, 37])], batch_size=2, max_value=-1)
    print("pred", [letters[int(i)] for i in pred[0]])

    cv2.imshow("npy", img)
    cv2.waitKey(0)
