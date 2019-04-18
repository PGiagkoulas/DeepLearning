# main.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AllConvNet(nn.Module):
	def __init__(self, num_classes):
		super(AllConvNet, self).__init__()
		self.num_classes = num_classes
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 96, 3, padding=1), nn.ReLU(),
			nn.Conv2d(96, 96, 3, padding=1), nn.ReLU()
			)
		self.max_pool = nn.MaxPool2d((3, 3), 2) 
		self.conv2 = nn.Sequential(
			nn.Conv2d(96, 192, 3, padding=1), nn.ReLU(),
			nn.Conv2d(192, 192, 3, padding=1), nn.ReLU()
			)
		self.conv3 = nn.Sequential(
			nn.Conv2d(192, 192, 3, padding=1), nn.ReLU(),
			nn.Conv2d(192, 192, 1), nn.ReLU(),
			nn.Conv2d(192, num_classes, 1), nn.ReLU()
			)
		# softmax included in xe loss

	def forward(self, x):
		x = self.conv1(x)
		x = self.max_pool(x)
		x = self.conv2(x)
		x = self.max_pool(x)
		x = self.conv3(x)
		x = F.avg_pool2d(x, kernel_size=x.size()[2:])
		return x.squeeze()


class SimpleConvNet(nn.Module):
	def __init__(self, num_classes):
		super(SimpleConvNet, self).__init__()
		self.num_classes = num_classes
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 96, 3, padding=1), nn.ReLU(),
			nn.Conv2d(96, 96, 3, padding=1), nn.ReLU()
			)
		self.max_pool = nn.MaxPool2d((2,2))

		self.conv2 = nn.Sequential(
			nn.Conv2d(96, 96, 3, padding=1), nn.ReLU(),
			nn.Conv2d(96, 96, 3, padding=1), nn.ReLU()
			)

		self.max_pool2 = nn.MaxPool2d((2,2))

		self.fc = nn.Linear(96*8*8, 10)

	def forward(self, x):
		x = self.conv1(x)
		x = self.max_pool(x)
		x = self.conv2(x)
		x = self.max_pool2(x)

		x = x.view(-1, 96*8*8)
		x = self.fc(x)
		return x




if __name__ == '__main__':
	
	model = AllConvNet(10).to(device)


	transform = transforms.Compose(
		[transforms.ToTensor(),
		 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
											download=True, transform=transform)
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=32,
											  shuffle=True, num_workers=2)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False,
										   download=True, transform=transform)

	test_loader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)


	optimizer = optim.Adam(model.parameters())
	loss_fn = nn.CrossEntropyLoss()

	trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
	evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(), 'loss': Loss(loss_fn)}, device=device)

	@trainer.on(Events.EPOCH_COMPLETED)
	def log_training_results(trainer):
	    evaluator.run(train_loader)
	    metrics = evaluator.state.metrics
	    print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
	          .format(trainer.state.epoch, metrics['accuracy'], metrics['loss']))

	@trainer.on(Events.EPOCH_COMPLETED)
	def log_validation_results(trainer):
	    evaluator.run(test_loader)
	    metrics = evaluator.state.metrics
	    print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
	          .format(trainer.state.epoch, metrics['accuracy'], metrics['loss']))


	trainer.run(train_loader, max_epochs=10)
