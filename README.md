# DeepLearning
Comparative report of deep learning architectures on the [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

By [Panagiotis Giagkoulas](https://github.com/PGiagkoulas) and [Lukas Edman](https://github.com/Leukas).

## Requirements
You need to have Anaconda or Miniconda installed:

[Miniconda](https://conda.io/en/latest/miniconda.html)

[Anaconda](https://www.anaconda.com/distribution/)

## Environment
To install the environment run:
```bat
conda env create -f environment.yml
```
After installing, you may need to activate the environment, using either
```bat
source activate keras_env
```
or 
```bat
activate keras_env
```
## Running the code
To run an experiment, run:
```bat
python main.py
```
There are a number of flags available to modify the experiment:
```
--model_name               Name of the model. Loads model with same name automatically. Flag must be used for the model to be saved.
--architecture             Architecture to use. [list of architectures below] 
--pretrained               If provided, will load a model pretrained on ImageNet and only the last layer is trained. Valid for VGG/ResNet/DenseNet.
--dataset                  Dataset to use. [cifar10/cifar100]
--save_interval            Save every x epochs.
--batch_size               Batch size for training and testing.
--n_epochs                 Number of epochs to train for.
--optimizer                Optimizer to use. [adam/rmsprop/sgd]
--lr                       Learning rate.
--export                   Export the training statistics to a file.
```
Supported architectures:
```
all_conv                   Regular All-CNN-C from Springenberg et al.
all_bn_conv                All-CNN-C with batch normalization
model_c                    Model C from Springenberg et al.
simple_conv                Simple CNN from Keras
simple_bn_conv             Simple CNN with batch norm
lenet5                     LeNet5
lenet5_do                  LeNet5 with dropout
lenet5_bn                  LeNet5 with batch norm
vgg16                      VGG16
resnet50                   ResNet50
densenet                   DenseNet
```
An example run of Simple CNN with batch normalization on CIFAR-10 using Adam looks like:
```bat
python main.py --architecture simple_bn_conv --dataset cifar10 --save_interval 1 --batch_size 64 --n_epochs 100 --optimizer adam --lr 1e-3 
```



<!-- ## Reproducing results
* CIFAR-10:
  * Simple CNN with Adam optimizer:
    * Using Dropout:
    ```bat
    python main.py --architecture simple_conv --dataset cifar10 --n_epochs 100 --optimizer adam --lr 1e-3 
    ```
    * Using Batch Normalization:
    ```bat
    python main.py --architecture simple_bn_conv --dataset cifar10 --n_epochs 100 --optimizer adam --lr 1e-3 
    ```
  * Simple CNN with SGD with momentum optimizer:
    * Using Dropout:
    ```bat
    python main.py --architecture simple_conv --dataset cifar10 --n_epochs 100 --optimizer sgd --lr 1e-4
    ```
    * Using Batch Normalization:
    ```bat
      python main.py --architecture simple_bn_conv --dataset cifar10 --n_epochs 100 --optimizer sgd --lr 1e-4
      ```
  * Lenet5 with Adam optimizer
    * No regularization:
    ```bat
    python main.py --architecture lenet5 --dataset cifar10 --n_epochs 100 --optimizer adam --lr 1e-4
    ```
    * Using Dropout:
    ```bat
    python main.py --architecture simple_conv --dataset cifar10 --n_epochs 100 --optimizer adam --lr 1e-4
    ```
    * Using Batch Normalization:
    ```bat
    python main.py --architecture simple_bn_conv --dataset cifar10 --n_epochs 100 --optimizer adam --lr 1e-4
    ```
  * Lenet5 with SGD with momentum optimizer:
    * No regularization:
    ```bat
    python main.py --architecture lenet5 --dataset cifar10 --n_epochs 100 --optimizer sgd --lr 1e-4
    ```
    * Using Dropout:
    ```bat
    python main.py --architecture simple_conv --dataset cifar10 --n_epochs 100 --optimizer sgd --lr 1e-3
    ```
    * Using Batch Normalization:
    ```bat
    python main.py --architecture simple_bn_conv --dataset cifar10 --n_epochs 100 --optimizer sgd --lr 1e-3
    ```
* CIFAR-100:
Run the same commands with --dataset cifar100
 -->
