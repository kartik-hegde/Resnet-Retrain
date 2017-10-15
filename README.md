# Resnet-Retrain
This repository helps to train ResNet and dump the trained model - which can then be modified and reloaded to the network for retraining.

This can be used for purposes like Weight Quantization etc. It converts the TensorFlow `ckpt` checkpoint file to a readable numpy format(`.npy`). This allows us to perform any operations on the trained model/visualize the trained model. This repository allows you to re-load the same data and re-train. You can once again dump the trained model for your requirements.


The original ResNet model is taken from the official Tensorflow ResNet model. 

Usage:

1. Run the prepare.sh by `sh prepare.sh`
2. Train the model : `python cifar10_main.py`
3. Run till the accuracy is good enough.
4. Check the latest check point file by `ls data/model`. For ex: `model.ckpt-8000`
4. Dump the weights using `python cifar10_main.py --dump=True --ckpt_file=<Latest Checkpoint filename>`
5. This should create a `weights_cifar10.npy` file, which contains the data.
6. You can use [this](https://github.com/karthik-hegde/weight_sharing) to edit the data. (ex: Quantize)
7. If you would like to re-run the training with the changed data(replace the original .npy file), 
   simply run `python cifar10_main.py --retrain=True --ckpt_file=<Latest Checkpoint filename>`

Note that, once you re-train, your check-point files are generated in `data/model_retrain` folder. If you would like to do one more iteration of retraining, pass the approprite `--model_dir` path.
