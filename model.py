from netrc import netrc

import tensorflow as tf
from keras import Model, Sequential
from keras.layers import ConvLSTM2D, Bidirectional, Input, Activation, Dense, Masking, GaussianNoise, TimeDistributed, \
    Flatten, Reshape, LSTM, Conv2D, MaxPooling2D
from keras.optimizers import Adam

from Handwritten_recognition.utils import *
from Handwritten_recognition.CTCModel import CTCModel


class MyModel():
    def __init__(self):
        self.padding_value = 0
        self.height = height_img
        self.width = width_img

    def create_model_convlstm(self):
        # Define the network architecture
        input_data = Input(name='input', shape=(32, self.width, self.height, 1))

        # 1...
        # flatten_input = TimeDistributed(Flatten())(input_data)
        # masking = TimeDistributed(Masking(mask_value=self.padding_value))(flatten_input)
        # noise = TimeDistributed(GaussianNoise(0.01))(masking)
        # reshaped = TimeDistributed(Reshape(input_data[1:]))(noise)

        # 2...
        # masking = TimeDistributed(Masking(mask_value=self.padding_value))(input_data)
        # masking = TimeDistributed(Reshape(input_data))(masking)
        # noise = GaussianNoise(0.01)(masking)

        blstm = Bidirectional(
            ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same', return_sequences=True, dropout=0.1))(input_data)
        blstm = Bidirectional(
            ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same', return_sequences=True, dropout=0.1))(blstm)
        blstm = Bidirectional(
            ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same', return_sequences=True, dropout=0.1))(blstm)

        flatten = TimeDistributed(Flatten())(blstm)
        dense = TimeDistributed(Dense(len(letters) + 1, name="dense"))(flatten)
        outrnn = Activation('softmax', name='softmax')(dense)

        network = CTCModel([input_data], [outrnn])
        network.compile(Adam(lr=0.0001))
        return network

    def create_model_lstm(self):

        # Define the network architecture
        input_data = Input(name='input', shape=(None, self.height))

        masking = Masking(mask_value=self.padding_value)(input_data)
        noise = GaussianNoise(0.01)(masking)

        blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(noise)
        blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(blstm)
        blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(blstm)

        dense = TimeDistributed(Dense(len(letters) + 1, name="dense"))(blstm)
        outrnn = Activation('softmax', name='softmax')(dense)

        network = CTCModel([input_data], [outrnn])
        # network.compile(Adam(lr=0.0001))
        return network

    def create_model_cnn_blstm(self):
        act = 'relu'
        conv_filters = 16
        kernel_size = (3, 3)
        pool_size = 2
        time_dense_size = 32
        rnn_size = 512

        input_data = Input(name='input', shape=(self.width, self.height, 1))

        # masking = Masking(mask_value=self.padding_value)(input_data)
        # noise = GaussianNoise(0.01)(masking)
        inner = Conv2D(conv_filters, kernel_size, padding='same',
                       activation=act, kernel_initializer='he_normal',
                       name='conv1')(input_data)
        inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
        inner = Conv2D(conv_filters, kernel_size, padding='same',
                       activation=act, kernel_initializer='he_normal',
                       name='conv2')(inner)
        inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

        conv_to_rnn_dims = (self.width // (pool_size ** 2),
                            (self.height // (pool_size ** 2)) * conv_filters)
        inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

        # cuts down input size going into RNN:
        inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

        blstm = Bidirectional(LSTM(rnn_size, return_sequences=True, dropout=0.1))(inner)
        blstm = Bidirectional(LSTM(rnn_size, return_sequences=True, dropout=0.1))(blstm)
        # blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(blstm)

        dense = TimeDistributed(Dense(len(letters) + 1, name="dense"))(blstm)
        outrnn = Activation('softmax', name='softmax')(dense)

        network = CTCModel([input_data], [outrnn])
        # network.compile(Adam(lr=0.0001))
        return network
