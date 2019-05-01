# main.py
import cv2
import os
import argparse
from keras.optimizers import rmsprop
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.datasets import cifar10, cifar100

from networks import *
from utils import load_model, save_model_architecture

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

args = parser.parse_args()
args.model_path = os.path.join('models', args.model_name)
args.initial_epoch = 0 

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
	
	checkpt = ModelCheckpoint(
		os.path.join(args.model_path,'weights.ep{epoch:03d}.val{val_acc:.3f}.hdf5'), 
		save_weights_only=True, 
		period=args.save_interval)

	model.compile(
				#optimizer='adam',
		optimizer=rmsprop(lr=0.0001, decay=1e-6),
		loss='categorical_crossentropy', 
		metrics=['accuracy'])
	

	if args.model_name == 'test':
		model.fit(
			x_train, 
			y_train, 
			validation_data=(x_test, y_test),
			initial_epoch=args.initial_epoch, 
			epochs=args.n_epochs+args.initial_epoch, 
			batch_size=args.batch_size)
	else:
		model.fit(
			x_train, 
			y_train, 
			validation_data=(x_test, y_test),
			initial_epoch=args.initial_epoch, 
			epochs=args.n_epochs+args.initial_epoch, 
			batch_size=args.batch_size, 
			callbacks=[checkpt])
	# model.evaluate(x_test, y_test,
	# 	batch_size=args.batch_size)
