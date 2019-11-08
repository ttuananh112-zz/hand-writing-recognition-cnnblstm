import cv2
import os
import numpy as np

from Handwritten_recognition.model import *
from Handwritten_recognition.preprocess_data import *

model = MyModel()
myModel = model.create_model_cnn_blstm()
myModel.load_model("../model_cnn_blstm/", Adam(lr=0.0001), init_archi=True,
                   file_weights="model_cnnblstm_sparse_h32_final_final.h5")

folder = "../test_files/t1/"
for i in os.listdir(folder):
    img = cv2.imread(folder + i)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    img_gray = cv2.bitwise_not(img_gray)

    threshold = 200
    img_gray[img_gray >= threshold] = 255
    img_gray[img_gray < threshold] = 0
    img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, np.ones((3,3)).astype(np.uint8))
    cv2.imshow("img", img_gray)

    img_gray = resize_img(img_gray, h_out=32)
    img_gray = cv2.rotate(img_gray, cv2.ROTATE_90_CLOCKWISE)

    h, w = img_gray.shape[0], img_gray.shape[1]
    left = 150 - h
    if left >= 0:
        zero_np = np.zeros((left, w))
        img_input = np.vstack((img_gray, zero_np))
    else:
        img_input = cv2.resize(img_gray, (32, 150))  # (w,h)

    list_input = []
    img_input = normalize_img(img_input)
    img_input = np.expand_dims(img_input, axis=2)
    list_input.append(img_input)
    # dummy data
    list_input.append(np.zeros_like(img_input))

    print(img_input.shape)
    pred = myModel.predict([np.array(list_input), np.array([37, 37])], batch_size=2, max_value=-1)
    print("pred", [letters[int(i)] for i in pred[0]])

    cv2.waitKey(0)
