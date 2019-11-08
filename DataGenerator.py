from keras.utils import Sequence
import numpy as np
from Handwritten_recognition.utils import *
from Handwritten_recognition.preprocess_data import normalize_img
from keras.utils import np_utils


# type_generator=['train','validate']
class DataGenerator(Sequence):
    def __init__(self, indexes, data, iterators, batch_size=batch_size):
        self.batch_size = batch_size
        self.indexes = indexes
        self.index = 0
        # data from csv
        self.data = data
        self.iterators = iterators

    def __getitem__(self, idx):

        index_per_batch = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        list_input_pad = []
        list_output_pad = []
        list_input_len = []
        list_output_len = []

        for i in range(self.batch_size):

            input_np = np.load("C:/data/hand_writing/input_np/train/" + folder_input_name + "/" + self.data.iloc[index_per_batch[i]]['input'])
            input_np = normalize_img(input_np)
            # CNN-LSTM (+ channel dimension)
            input_np = np.expand_dims(input_np, axis=2)
            list_input_pad.append(input_np)

            output_np = np.load("C:/data/hand_writing/output_np/train/" + folder_output_name + "/" + self.data.iloc[index_per_batch[i]]['output'])
            list_output_pad.append(output_np)

            # # LSTM
            # input_len = self.data.iloc[index_per_batch[i]]['input_len']
            # list_input_len.append(input_len)

            # CNN-LSTM
            # 2**2 ( 2 max pooling kernel 2x2 )
            list_input_len.append(width_img // 2**2)

            output_len = self.data.iloc[index_per_batch[i]]['output_len']
            list_output_len.append(output_len)

        return [np.array(list_input_pad), np.array(list_output_pad),
                np.array(list_input_len), np.array(list_output_len)], np.zeros(self.batch_size)

    def __len__(self):
        return self.iterators

    def _shuffle(self):
        np.random.shuffle(self.indexes)

    def on_epoch_end(self):
        self._shuffle()
