# DeepLearning
Comparative report of deep learning architectures on the [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

## You need to have anaconda or miniconda installed
[Miniconda](https://conda.io/en/latest/miniconda.html)

[Anaconda](https://www.anaconda.com/distribution/)

## Environment
To install the environment run:
```bat
conda env create -f environment.yml
```

## Reproducing results
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
