# networks.py
from keras.layers import Dense, Activation,  Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential


def all_conv_net(args):
	model = Sequential()
	model.add(Conv2D(96, kernel_size=3, activation='relu', input_shape=(32,32,3), padding='same'))
	model.add(Conv2D(96, kernel_size=3, activation='relu', padding='same'))
	model.add(MaxPooling2D((3,3), strides=2))
	model.add(Conv2D(192, kernel_size=3, activation='relu', padding='same'))
	model.add(Conv2D(192, kernel_size=3, activation='relu', padding='same'))
	model.add(MaxPooling2D((3,3), strides=2))
	model.add(Conv2D(192, kernel_size=3, activation='relu', padding='same'))
	model.add(Conv2D(192, kernel_size=1, activation='relu', padding='same'))
	model.add(Conv2D(args.n_outputs, kernel_size=1, activation='relu', padding='same'))
	model.add(GlobalAveragePooling2D())
	model.add(Activation('softmax'))
	return model




MODELS = {
	'all_conv': all_conv_net 
}
