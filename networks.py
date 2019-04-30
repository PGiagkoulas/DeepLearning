# networks.py
from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D, Dropout
from keras.models import Sequential


def all_conv_net(args):
	model = Sequential()
	model.add(Dropout(0.2))
	model.add(Conv2D(96, kernel_size=3, activation='relu', input_shape=(32,32,3), padding='same'))
	model.add(Conv2D(96, kernel_size=3, activation='relu', padding='same'))
	model.add(MaxPooling2D((3,3), strides=2))
	model.add(Dropout(0.5))
	model.add(Conv2D(192, kernel_size=3, activation='relu', padding='same'))
	model.add(Conv2D(192, kernel_size=3, activation='relu', padding='same'))
	model.add(MaxPooling2D((3,3), strides=2))
	model.add(Dropout(0.5))
	model.add(Conv2D(192, kernel_size=3, activation='relu', padding='same'))
	model.add(Conv2D(192, kernel_size=1, activation='relu', padding='same'))
	model.add(Conv2D(args.n_outputs, kernel_size=1, activation='relu', padding='same'))
	model.add(GlobalAveragePooling2D())
	#model.add(Dropout(0.5))
	model.add(Activation('softmax'))
	return model


def simple_conv_net(args):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(args.n_outputs))
        model.add(Activation('softmax'))
        return model

def pre_vgg16_net(args):
        model = VGG16(weights="imagenet", include_top=False)
        # freeze conv layers
        for layer in model.layers:
                layer.trainable = False
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(args.n_outputs))
        model.add(Activation('softmax'))
        return model

def alex_net():
        model = Sequential()
        # 1st Conv + ReLU + MaxPool
        model.add(Conv2D(93, (11,11), strides=(4,4), padding='valid', input_shape=(32, 32, 3), name='conv1'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        # 2nd Conv + ReLU + MaxPool
        model.add(Conv2D(256, (5,5), strides=(1,1), padding='valid', name='conv2'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        # 3rd Conv + ReLU
        model.add(Conv2D(384, (3,3), strides=(1,1), padding='valid', name='conv3'))
        model.add(Activation('relu'))
        # 4th Conv + ReLU
        model.add(Conv2D(384, (3,3), strides=(1,1), padding='valid', name='conv4'))
        model.add(Activation('relu'))
        # 5th Conv + ReLU + MaxPool
        model.add(Conv2D(256, (3,3), strides=(1,1), padding='valid', name='conv5'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        # Flatten
        model.add(Flatten())
        # 1st Fully Connected + ReLU
        model.add(Dense(9216), name='fc6')
        model.add(Activation('relu'))
        # 2nd Fully Connected + ReLU
        model.add(Dense(4096), name='fc7')
        model.add(Activation('relu'))
        # 3rd Fully Connected + ReLU
        model.add(Dense(4096), name='fc8')
        model.add(Activation('relu'))
        # Output
        model.add(Dense(10))
        model.add(Activation('softmax'))

        
        return model
		
MODELS = {
	'all_conv': all_conv_net,
        'simple_conv': simple_conv_net,
        'pre_vgg16': pre_vgg16_net
}
