import numpy as np
import pylab
import cv2
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Reshape, Lambda, Dropout
from keras.layers.merge import add, concatenate
from keras.layers.recurrent import GRU
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.utils.data_utils import get_file
from sklearn.cross_validation import train_test_split
from keras.preprocessing import image
from keras.callbacks import EarlyStopping, Callback
from keras import backend as K
from random import randint

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import matplotlib.pyplot as plt

## loading training images
images = []
labels = []

chars = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
         "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q",
         "R", "S", "T", "U", "V", "W", "X", "Y", "Z"];


def encodePlate(str):
    num = np.zeros((37, 7))

    #    print(str)
    for i in range(len(str)):
        for j in range(36):
            #            print(i,j)
            if (str[i] == chars[j]):
                num[j, i] = 1;

    if (len(str) == 6):
        num[36, 6] = 1;
    if (len(str) == 5):
        num[36, 6] = 1;
        num[36, 5] = 1;

    print(str, '\n', num)

    return num


## load images
pattern = re.compile('(.+\/)?(\w+)\/([^_]+)_.+jpg')
all_files = glob(os.path.join('train/*jpg'))
all_files = [re.sub(r'\\', r'/', file) for file in all_files]

for entry in all_files:
    r = re.match(pattern, entry)
    if r and os.path.getsize(entry) != 0:
        #        print(entry)
        plateNum = entry.split('_', 1)[0];
        #        print(plateNum)
        plateNum = plateNum.split('/', 2)[2];
        #        print(plateNum)
        labels.append(encodePlate(plateNum))
        images.append(cv2.imread(entry))

images = np.asarray(images)
labels = np.asarray(labels)

print(images.shape)
print(labels.shape)

train_X, valid_X, train_Y, valid_Y = train_test_split(images, labels, test_size=0.1)
trainNum = train_X.shape[0];
validNum = valid_X.shape[0];


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


## create model
width, height, n_len, n_class = 70, 30, 7, 37
rnn_size = 128

##########################
###### BUILD MODEL #######
input_tensor = Input((width, height, 3))
x = input_tensor
for i in range(2):
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

conv_shape = x.get_shape()
x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)

x = Dense(32, activation='relu')(x)

gru_1 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru1')(x)
gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True,
             init='he_normal', name='gru1_b')(x)
gru1_merged = add([gru_1, gru_1b])

gru_2 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru2')(gru1_merged)
gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True,
             init='he_normal', name='gru2_b')(gru1_merged)
x = concatenate([gru_2, gru_2b])

x = Dropout(0.25)(x)
x = Dense(n_class, init='he_normal', activation='softmax')(x)

base_model = Model(input=input_tensor, output=x)
##########################

labels = Input(name='the_labels', shape=[n_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

loss_out = Lambda(ctc_lambda_func, output_shape=(1,),
                  name='ctc')([x, labels, input_length, label_length])

model = Model(input=[input_tensor, labels, input_length, label_length], output=[loss_out])

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')


## CTC cost
def decodePlateVec(y):
    vec = np.zeros((1, n_len), dtype=np.uint8)
    for i in range(7):
        vec[0, i] = np.argmax(y[:, i])
    return vec


def evaluate(model, batch_num=50):
    batch_acc = 0

    for i in range(batch_num):
        ## random select 50 images as its CTC cost, larger batch_num is better
        randIdx = randint(0, train_X.shape[0] - 1);
        X_test = train_X[randIdx, :, :, :].transpose(1, 0, 2).reshape(1, 70, 30, 3)
        y_test = decodePlateVec(train_Y[randIdx, :, :])

        y_pred = base_model.predict(X_test)
        shape = y_pred[:, 2:, :].shape
        ctc_decode = K.ctc_decode(y_pred[:, 2:, :],
                                  input_length=np.ones(shape[0]) * shape[1])[0][0]
        out = K.get_value(ctc_decode)[:, :7]
        if out.shape[1] == 7:
            batch_acc += ((y_test == out).sum(axis=1) == 7).mean()
    return batch_acc / batch_num


class Evaluate(Callback):
    def __init__(self):
        self.accs = []

    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(base_model) * 100
        self.accs.append(acc)
        print()
        print('acc:', acc)


evaluator = Evaluate()

## fit model

trainX = train_X.transpose(0, 2, 1, 3)
validX = valid_X.transpose(0, 2, 1, 3)

trainLabel = np.zeros((trainNum, n_len), dtype=np.uint8)
validLabel = np.zeros((validNum, n_len), dtype=np.uint8)
for i in range(trainNum):
    trainLabel[i] = decodePlateVec(train_Y[i, :, :])
for i in range(validNum):
    validLabel[i] = decodePlateVec(valid_Y[i, :, :])

trainInputL = np.ones(trainNum) * int(conv_shape[1] - 2)
trainLabelL = np.ones(trainNum) * n_len
validInputL = np.ones(validNum) * int(conv_shape[1] - 2)
validLabelL = np.ones(validNum) * n_len

## this should be correct, trainY is one hot results. But it didn't work for processing
# history = model.fit([trainX, trainLabel, trainInputL, trainLabelL], trainY, batch_size=32, epochs=30, callbacks=[evaluator], shuffle=True, verbose=1, validation_data=([validX,validLabel,validInputL,validLabelL], validY))
## trainLabel isn't one hot labels
history = model.fit([trainX, trainLabel, trainInputL, trainLabelL], trainLabel, batch_size=32, epochs=30,
                    callbacks=[evaluator], shuffle=True, verbose=1,
                    validation_data=([validX, validLabel, validInputL, validLabelL], validLabel))

# model.save('plateCTC.h5')
base_model.save('plateCTC_base.h5')
