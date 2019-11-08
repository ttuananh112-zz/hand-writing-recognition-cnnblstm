import os
from builtins import print
from datetime import time
import pandas as pd

import cv2
import numpy as np
from keras.preprocessing import sequence
from Handwritten_recognition.utils import *


def preprocessing_img(img_rgb, threshold_low=50, threshold_high=100):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_gray = resize_img(img_gray, h_out=height_img)
    # rotate img -> width=time_steps, height=features
    img_gray = cv2.rotate(img_gray, cv2.ROTATE_90_CLOCKWISE)
    # equalized_img_gray = cv2.equalizeHist(img_gray)
    img_gray = cv2.bitwise_not(img_gray)
    img_gray[img_gray < threshold_low] = 0
    img_gray[img_gray > threshold_high] = 255
    return img_gray


def normalize_img(sequences):
    return (sequences / 255.0).astype("float16")


def resize_img(img, h_out=32):
    h, w = img.shape[0], img.shape[1]
    scale_percent = h_out / h
    w_out = int(w * scale_percent)

    if w_out == 0:
        w_out = 1

    img_out = cv2.resize(img, (w_out, h_out))
    return img_out


# bug transpose..? maybe at time_steps.append*
def separate_to_time_step(img_gray, step_size=10):
    h, w = img_gray.shape
    time_steps = []

    # in case the image is smaller than step_size -> post-padding 0
    if w < step_size:
        rest_w = step_size - w
        # print("rest_w", rest_w)
        rest_part = np.zeros((h, rest_w))
        # print("SHAPE", normed_img_gray.shape, rest_part.shape)
        time_steps.append(np.hstack((img_gray, rest_part)))
        return np.array(time_steps)

    for i in range(int(w / step_size)):
        time_steps.append(img_gray[:, i * step_size:(i + 1) * step_size])

    # append the last that's not fit enough one step
    if w / step_size != int(w / step_size):
        time_steps.append(img_gray[:, -step_size:])

    return np.array(time_steps).astype(type_np)


def cut_dataset_into_minor_files_npy(folder="C:/data/hand_writing/", isInput=False, isOutput=False):

    if not os.path.exists(folder + "input_np/train/" + folder_input_name):
        os.mkdir(folder + "input_np/train/" + folder_input_name)
    if not os.path.exists(folder + "output_np/train/" + folder_output_name):
        os.mkdir(folder + "output_np/train/" + folder_output_name)
    if not os.path.exists(folder + "input_np/test/" + folder_input_name):
        os.mkdir(folder + "input_np/test/" + folder_input_name)
    if not os.path.exists(folder + "output_np/test/" + folder_output_name):
        os.mkdir(folder + "output_np/test/" + folder_output_name)

    # INPUT
    if isInput:
        print("loading")
        input_pad = np.load(folder + "input_np/input_pad_" + folder_input_name + ".npy")
        print("loaded")
        i = 0
        for sample in input_pad:
            if i % 100 == 0:
                print(i)
            np.save(folder + "input_np/train/" + folder_input_name + "/input_" + str(i), sample)
            i += 1
        print(input_pad.shape)

    # OUTPUT
    if isOutput:
        print("loading")
        output_pad = np.load(folder + "output_np/output_pad_" + folder_output_name + ".npy")
        print("loaded")
        i = 0
        for sample in output_pad:
            if i % 100 == 0:
                print(i)
            np.save(folder + "output_np/train/" + folder_output_name + "/output_" + str(i), sample)
            i += 1
        print(output_pad.shape)


def save_dataset_to_npy_file(folder="C:/data/hand_writing/", inputs_all=None, outputs_all=None):
    # Read from dataset and save to npy file

    if inputs_all is not None:
        inputs_all = sequence.pad_sequences(inputs_all, padding='post', value=padding_input_value, dtype=type_np).astype(
            type_np)

        # if type_time_step == "per_column":
        #     inputs_all = np.squeeze(inputs_all, axis=3)

        print("input:", inputs_all.shape)
        np.save(folder + "input_np/input_pad_" + folder_input_name, inputs_all)
        print("saved input")

    if outputs_all is not None:
        output_pad = sequence.pad_sequences(outputs_all, padding='post', value=padding_output_value, dtype=type_np).astype(
            'int8')

        print("output:", output_pad.shape)
        np.save(folder + "output_np/output_pad_" + folder_output_name, output_pad)
        print("saved output")


def create_csv_and_seperate_dataset(csv_file_name="dataset_h32.csv", isCreateNpy=False, isSeperateNpy=False):
    print("loading")
    inputs_all, max_len_timesteps, list_index = read_dataset(type_time_step=type_time_step)
    outputs_all, max_len_characters = read_label(list_index, type_label=type_label)
    print("loaded")
    # # length of inputs and outputs
    inputs_all_len = np.array([len(x) for x in inputs_all])
    outputs_all_len = np.array([len(x) for x in outputs_all])

    print(inputs_all_len.shape)
    print(outputs_all_len.shape)

    len_in = len(inputs_all_len)
    len_out = len(outputs_all_len)

    print("num input:", len_in)
    print("num output:", len_out)

    input_file_name = ["input_" + str(i) + ".npy" for i in range(len_in)]
    output_file_name = ["output_" + str(i) + ".npy" for i in range(len_out)]

    df = pd.DataFrame({"input": input_file_name,
                       "output": output_file_name,
                       "input_len": inputs_all_len,
                       "output_len": outputs_all_len})

    # purified dataset
    df_np = df.to_numpy()
    check = np.nonzero((inputs_all_len-outputs_all_len > 2) & (inputs_all_len > 5) & (outputs_all_len > 0))
    # print(check)
    df_np = df_np[check]
    df_new = pd.DataFrame({"input": df_np[:, 0],
                           "output": df_np[:, 1],
                           "input_len": df_np[:, 2],
                           "output_len": df_np[:, 3]})

    df_new.to_csv(csv_file_name)
    print("done to csv")

    if isCreateNpy:
        # save_dataset_to_npy_file(outputs_all=outputs_all)
        save_dataset_to_npy_file(inputs_all=inputs_all, outputs_all=outputs_all)
        if isSeperateNpy:
            # cut_dataset_into_minor_files_npy(isOutput=True)
            cut_dataset_into_minor_files_npy(isInput=True, isOutput=True)


# type_time_step = ['window', 'per_column']
# dataset return from this function is not normalized yet
def read_dataset(data_folder="D:/DATASET/words", type_time_step='window'):
    dataset = []
    max_len_timesteps = 0
    len_timesteps_limit = 150
    num_samples_have_timesteps_bigger_than_limit = 0
    index = 0
    list_index = []

    for sub_folder in os.listdir(data_folder):
        path = data_folder + "/" + sub_folder
        for sub_folder2 in os.listdir(path):
            path2 = path + "/" + sub_folder2
            for img_name in os.listdir(path2):
                img_path = path2 + "/" + img_name
                img_rgb = cv2.imread(img_path)
                # try:
                img_gray = preprocessing_img(img_rgb).astype(type_np)

                # if type_time_step == 'per_column':
                #     time_steps = separate_to_time_step(img_gray, step_size=1)
                # else:
                #     time_steps = separate_to_time_step(img_gray)


                # do not take the one that longer than limit
                if img_gray.shape[0] > len_timesteps_limit:
                    num_samples_have_timesteps_bigger_than_limit += 1
                else:
                    dataset.append(img_gray)
                    list_index.append(index)

                # update max_len_timesteps (just for debugging)
                if img_gray.shape[0] > max_len_timesteps:
                    max_len_timesteps = img_gray.shape[0]

                index += 1
                # except:
                #     print("err: ", img_name)

                # ####################
                # if len(dataset) == 100:
                #     return dataset, max_len_timesteps
                # ####################

    print("max_len_timesteps:", max_len_timesteps)
    print("num_samples_have_timesteps_bigger_than_limit:", num_samples_have_timesteps_bigger_than_limit)
    return dataset, max_len_timesteps, list_index


# type_label = ["one_hot", "sparse"]
def read_label(list_index, label_file="D:/DATASET/sentences_ascii_label/words.txt", type_label="one_hot"):
    label = []
    max_len_character = 0
    with open(label_file) as f:
        for line in f.readlines():
            if line[0] != "#":
                components = line.split(" ")
                id, result, gray_level, x, y, w, h, tag = components[:8]
                word = components[8:]
                word = "".join(word)
                word = word.strip().replace(" ", "").replace("\'", "\"").upper()

                series = []
                for c in word:
                    if c == "*":
                        continue

                    if type_label == "sparse":
                        series.append(letters.index(c))
                    # one_hot
                    else:
                        sample = np.zeros(len(letters))
                        sample[letters.index(c)] = 1
                        series.append(sample)
                    # break and only get the first one if c is not a character
                    if not c.isalpha():
                        break

                series = np.array(series).astype(type_np)

                if series.shape[0] > max_len_character:
                    max_len_character = series.shape[0]

                label.append(series)

                #####################
                # if len(label) == 100:
                #     return label, max_len_character
                #####################
    new_label = [i for i in np.array(label)[list_index]]
    return new_label, max_len_character


if __name__ == '__main__':
    # len dataset : (115318,)
    # max_len_timesteps : 103 (with time_step=10)

    # save_dataset_to_npy_file(isOutput=True)
    # print("done to_npy")
    # input: (115318, 514, 16)
    # output: (115318, 17)
    # cut_dataset_into_minor_files_npy(isOutput=True)
    # create_csv_and_seperate_dataset()

    create_csv_and_seperate_dataset("dataset_h32.csv", isCreateNpy=True, isSeperateNpy=True)
    print("hi")
