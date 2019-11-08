from Handwritten_recognition.model import MyModel
from Handwritten_recognition.preprocess_data import read_dataset, read_label, normalize_img
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint
from Handwritten_recognition.utils import *
import pandas as pd
from Handwritten_recognition.DataGenerator import *
import numpy as np
from keras.optimizers import Adam, Adadelta
from keras.models import load_model, model_from_json


if __name__ == '__main__':
    # read csv file
    data = pd.read_csv("dataset_h32.csv")
    dataset_len = data.shape[0]

    model = MyModel()
    myModel = model.create_model_cnn_blstm()
    myModel.load_model(model_folder, Adam(lr=0.0001), init_archi=True, file_weights=file_weights)
    # myModel.compile(Adam(lr=0.0001))

    # json_file = open(model_folder + 'model_train.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # myModel = model_from_json(loaded_model_json)
    # myModel.load_weights(model_folder+new_file_weights)

    # generate data
    indexes = np.arange(dataset_len)
    np.random.shuffle(indexes)
    ###############################
    # train
    indexes_train = indexes[:int(dataset_len*train_percent)]
    indexes_validate = indexes[int(dataset_len*train_percent):int(dataset_len*(train_percent+validate_percent))]
    indexes_test = indexes[-int(dataset_len*test_percent):]

    iterators_train = int(int(dataset_len*train_percent) / batch_size)
    iterators_validate = int(int(dataset_len*validate_percent) / batch_size)
    iterators_test = int(int(dataset_len*test_percent) / batch_size)

    # indexes_train = indexes[:64]
    # iterators_train = 1
    # indexes_validate = indexes[64:128]
    # iterators_validate = 1

    train_gen = DataGenerator(indexes_train, data, iterators_train)
    # validate_gen = DataGenerator(indexes_validate, data, iterators_validate)

    checkpoint = ModelCheckpoint(model_folder + new_file_weights, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    history = myModel.fit_generator(generator=train_gen, steps_per_epoch=iterators_train,
                          # validation_data=validate_gen, validation_steps=iterators_validate,
                          verbose=1, epochs=20, callbacks=callbacks_list)

    # myModel.save_model(model_folder)

    ###############################


