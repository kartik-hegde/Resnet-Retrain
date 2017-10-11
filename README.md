# Resnet-Retrain
This repository helps to train ResNet and dump the trained model - which can then be modified and reloaded to the network for retraining.


The original ResNet model is taken from the official Tensorflow ResNet model. 

Usage:

1. Run the prepare.sh by `sh prepare.sh`
2. Train the model : `python cifar10_main.py`
3. Run till the accuracy is good enough.
4. Dump the weights using `python cifar10_main.py --dump=True`
5. This should create a `weights_cifar10.npy` file, which contains the data.
6. You can use [this](https://github.com/karthik-hegde/weight_sharing) to edit the data. (ex: Quantize)
7. If you would like to re-run the training with the changed data(replace the original .npy file), 
   simply run `python cifar10_main.py --retrain=True`
