from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Dense, Reshape, Flatten

model = Sequential()
model.add(Conv2D(filters=5, kernel_size=(3,3), padding='same'))
model.add(Reshape(target_shape=(10,50)))
# model.add(Flatten())
model.add(Dense(20))
model.build(input_shape=(None,10,10,1))
model.summary()