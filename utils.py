# utils.py
import os
from keras.models import model_from_yaml 

def load_model(args):
	# Load model architecture first
	model = load_model_architecture(args)
	print(model.layers)
	# Next load weights, load from latest epoch
	models = os.listdir(args.model_path)
	# print(models[-1])
	model.load_weights(os.path.join(args.model_path, models[-1]))
	args.initial_epoch = int(models[-1].split('.')[1][2:])

	print("Loaded weights from: {}/{}".format(args.model_name, models[-1]))
	return model

def save_model_architecture(model, args):
	# write model architecture to yaml file
	if not os.path.exists(os.path.join(args.model_path, 'model.yaml')):
		model_yaml = model.to_yaml()
		with open(os.path.join(args.model_path, 'model.yaml'), 'w') as yaml_file:
			yaml_file.write(model_yaml)

def load_model_architecture(args):
	# load model architecture from yaml file
	yaml_file = open(os.path.join(args.model_path, 'model.yaml'), 'r')
	model_yaml = yaml_file.read()
	yaml_file.close()
	return model_from_yaml(model_yaml)


def all_conv_lr_schedule(epoch_num):
	lr = 0.01
	if epoch_num < 200:
		return lr
	elif epoch_num < 250:
		return 0.1 * lr
	elif epoch_num < 300:
		return 0.01 * lr
	else:
		return 0.001 * lr