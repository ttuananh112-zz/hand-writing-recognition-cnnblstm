from Handwritten_recognition.model import *
from Handwritten_recognition.preprocess_data import *
from keras.models import Model, model_from_json
import tensorflow as tf
import cv2
from Handwritten_recognition.utils import *

data = pd.read_csv("dataset_h32.csv")
dataset_len = data.shape[0]

model = MyModel()
myModel = model.create_model_cnn_blstm()
# myModel.compile(Adam(lr=0.0001))
myModel.load_model("../model_cnn_blstm/", Adam(lr=0.0001), init_archi=True, file_weights="model_cnnblstm_sparse_h32_final_2.h5")
# json_file = open(model_folder + 'model_pred.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# myModel = model_from_json(loaded_model_json, custom_objects={"tf": tf})
# myModel.load_weights(model_folder+file_weights)

# generate data
indexes = np.arange(dataset_len)
np.random.shuffle(indexes)

##############################
# test predict
index_per_batch = indexes[: 10]
list_input_pad = []
list_output_pad = []
list_input_len = []
list_output_len = []

list_input_pad_raw = []

for i in range(10):
    input_np = np.load("C:/data/hand_writing/input_np/train/" + folder_input_name + "/" + data.iloc[index_per_batch[i]]['input'])
    list_input_pad_raw.append(input_np)

    input_norm_np = normalize_img(input_np)
    input_norm_np = np.expand_dims(input_norm_np, axis=2)
    list_input_pad.append(input_norm_np)

    output_np = np.load("C:/data/hand_writing/output_np/train/" + folder_output_name + "/" + data.iloc[index_per_batch[i]]['output'])
    list_output_pad.append(output_np)

    input_len = data.iloc[index_per_batch[i]]['input_len']
    # width / ( 2 * max pooling kernel 2x2 )
    list_input_len.append(width_img//(2**2))

    output_len = data.iloc[index_per_batch[i]]['output_len']
    list_output_len.append(output_len)

pred = myModel.predict([np.array(list_input_pad), np.array(list_input_len)], batch_size=10, max_value=-1)


for i in range(10):
    print("pred", [letters[int(j)] for j in pred[i]])
    cv2.imshow("input", cv2.rotate(list_input_pad_raw[i], cv2.ROTATE_90_COUNTERCLOCKWISE))
    cv2.waitKey(0)

