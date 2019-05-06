# main.py
import os
import argparse
from functools import partial
from keras.optimizers import rmsprop, adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.utils import to_categorical
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator

from networks import *
from utils import load_model, save_model_architecture, all_conv_lr_schedule

parser = argparse.ArgumentParser(description='Run a model.')
parser.add_argument('--model_name', type=str, default='test',
					help='Name of the model. Loads model with same name automatically.')
parser.add_argument('--architecture', type=str, default='vgg16',
					help='Architecture to use. Note: this will be ignored if model_name is a different architecture.')
parser.add_argument('--pretrained', action='store_true',
					help='Use "--pretrained" for a model pretrained on imagenet. VGG/ResNet/DenseNet only currently.')
parser.add_argument('--dataset', type=str, default='cifar100',
					help='Dataset to use. [cifar10/cifar100]')
parser.add_argument('--save_interval', type=int, default=1,
					help='Save every x epochs.')
parser.add_argument('--batch_size', type=int, default=64,
					help='Batch size. Default 64.')
parser.add_argument('--n_epochs', type=int, default=1,
					help='Number of epochs to train for. Default 1.')
parser.add_argument('--optimizer', type=str, default='adam',
					help='Optimizer to use. [adam/rmsprop/sgd]')

args = parser.parse_args()
args.model_path = os.path.join('models', args.model_name)
args.initial_epoch = 0 
args.optimizer_name = args.optimizer

args.input_shape = (32, 32, 3)

OPTIMIZERS = {
	'adam': adam,
	'rmsprop': partial(rmsprop, decay=1e-6),
	'sgd': SGD,
	'all_conv_sgd': partial(SGD, momentum=0.9)
}

args.optimizer = OPTIMIZERS[args.optimizer]

if not os.path.isdir('models'):
	os.mkdir('models')

# --- LOAD DATA ---
if args.dataset == 'cifar100':
	(x_train, y_train), (x_test, y_test) = cifar100.load_data()
	args.n_outputs = 100
elif args.dataset == 'cifar10':
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	args.n_outputs = 10

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# train_idg = ImageDataGenerator()
# test_idg = ImageDataGenerator()
# train_idg.fit(x_train)
# test_idg.fit(x_test)

# --- LOAD MODEL ---
if args.model_name == 'test':
	model = MODELS[args.architecture](args)	
elif os.path.isdir(args.model_path):
	model = load_model(args)
else:
	os.mkdir(args.model_path)
	model = MODELS[args.architecture](args)
	save_model_architecture(model, args)

if __name__ == '__main__':
	
	callbacks = []
	if args.model_name != 'test':
		checkpt = ModelCheckpoint(
			os.path.join(args.model_path,'weights.ep{epoch:03d}.val{val_acc:.3f}.hdf5'), 
			save_weights_only=True, 
			period=args.save_interval)
		callbacks.append(checkpt)

	if args.optimizer_name == 'all_conv_sgd':
		lrs = LearningRateScheduler(all_conv_lr_schedule)
		callbacks.append(lrs)

	model.compile(
		# optimizer=SGD(lr=0.0001, momentum=0.9, decay=)
		optimizer=args.optimizer(lr=0.01),
		loss='categorical_crossentropy', 
		metrics=['accuracy'])
	
	model.fit(
		x_train, 
		y_train, 
		validation_data=(x_test, y_test),
		initial_epoch=args.initial_epoch, 
		epochs=args.n_epochs+args.initial_epoch, 
		batch_size=args.batch_size, 
		callbacks=callbacks)

