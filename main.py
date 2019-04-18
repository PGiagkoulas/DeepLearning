# main.py

import keras
# from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Activation,  Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential
from keras.utils import to_categorical
from keras.datasets import cifar10, cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
y_train = to_categorical(y_train)


model = Sequential()
model.add(Conv2D(96, kernel_size=3, activation='relu', input_shape=(32,32,3), padding='same'))
model.add(Conv2D(96, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D((3,3), strides=2))
model.add(Conv2D(192, kernel_size=3, activation='relu', padding='same'))
model.add(Conv2D(192, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D((3,3), strides=2))
model.add(Conv2D(192, kernel_size=3, activation='relu', padding='same'))
model.add(Conv2D(192, kernel_size=1, activation='relu', padding='same'))
model.add(Conv2D(100, kernel_size=1, activation='relu', padding='same'))
model.add(GlobalAveragePooling2D())
model.add(Activation('softmax'))




model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=100)