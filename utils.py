letters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
           "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
           "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
           "U", "V", "W", "X", "Y", "Z", ".", ",", "(", ")",
           "?", "\"", "-", "#", ":", ";", "!", "&", "+"]

type_time_step = "per_column"
type_label = "sparse"
type_np = "uint8"
padding_input_value = 0
padding_output_value = -1
height_img = 32
width_img = 150  # fix manually
# (108662, 150, 32) h32

folder_input_name = type_time_step + "_" + type_np + "_" + "h" + str(height_img)
folder_output_name = type_label + "_" + type_np

# |----------- dataset -----------|
# |--train--|--validate--|--test--|
train_percent = 1
validate_percent = 0.0
test_percent = 0.0
#
batch_size = 32

model_folder = "../model_cnn_blstm/"
file_weights = "model_cnnblstm_sparse_h32_final_2.h5"
new_file_weights = "model_cnnblstm_sparse_h32_final_final.h5"
