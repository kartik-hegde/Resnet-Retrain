# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import resnet_model
import vgg_preprocessing

PWD = os.getcwd()

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir', type=str, default=PWD + '/imagenet_data/train',
    help='The directory where the ImageNet input data is stored.')

parser.add_argument(
    '--model_dir', type=str, default=PWD+'/imagenet_data/model',
    help='The directory where the model will be stored.')

parser.add_argument(
    '--resnet_size', type=int, default=34, choices=[18, 34, 50, 101, 152, 200],
    help='The size of the ResNet model to use.')

parser.add_argument(
    '--train_steps', type=int, default=6400000,
    help='The number of steps to use for training.')

parser.add_argument(
    '--steps_per_eval', type=int, default=40000,
    help='The number of training steps to run between evaluations.')

parser.add_argument(
    '--batch_size', type=int, default=32,
    help='Batch size for training and evaluation.')

parser.add_argument(
    '--map_threads', type=int, default=5,
    help='The number of threads for dataset.map.')

parser.add_argument(
    '--first_cycle_steps', type=int, default=None,
    help='The number of steps to run before the first evaluation. Useful if '
    'you have stopped partway through a training cycle.')

parser.add_argument('--retrain', type=bool, default=False,
    help='Is this a retraining run?')

parser.add_argument('--dump', type=bool, default=False,
    help='Do you want to dump the data?')

parser.add_argument('--ckpt_file', type=str, default='model.ckpt',
    help='Last saved model')

FLAGS = parser.parse_args()

# Scale the learning rate linearly with the batch size. When the batch size is
# 256, the learning rate should be 0.1.
_INITIAL_LEARNING_RATE = 0.1 * FLAGS.batch_size / 256

_NUM_CHANNELS = 3
_LABEL_CLASSES = 1001

_MOMENTUM = 0.9
_WEIGHT_DECAY = 1e-4

_NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

image_preprocessing_fn = vgg_preprocessing.preprocess_image
network = resnet_model.resnet_v2(
    resnet_size=FLAGS.resnet_size, num_classes=_LABEL_CLASSES)

batches_per_epoch = _NUM_IMAGES['train'] / FLAGS.batch_size

def get_weights(dump):
    """
    From the stored model, get all the trainable variables,
    and return a dictionary of all the corresponding variables
    """
    ## Get all the variable names
    var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    #Define a dictionary whose keys are scope names and values are variables
    model = dict()

    ##Define a dictionary to store the addresses of each variable in the 'model' dictionary
    # This is useful for reloading
    addr_table = dict()

    # GO through every variable and group all the variables with same scope
    for i,item in enumerate(var):
        temp_var = str(var[i]).split("'")[1].split("/")

        #The first part is the scope
        scope = temp_var[0]

        if(len(temp_var)==1):
                path=temp_var[0]
        elif(len(temp_var) < 3):
                path = temp_var[1]
        else:
                path  = temp_var[1]+'/'+temp_var[2]
        # Add them to the dictionary
        if scope in model:
                model[scope].append(path)
                count = count + 1
                addr_table[path] = count
        else:
                model[scope] = [path]
                count = 0
                addr_table[path] = count

    ##Save the address Table if you are dumping weights
    if dump:
        np.save("addr_table.npy", addr_table)

    # Return the model dictionary
    return model



def filenames(is_training):
  """Return filenames for dataset."""
  if is_training:
    return [
        os.path.join(FLAGS.data_dir, 'train-%05d-of-01024' % i)
        for i in range(0, 1024)]
  else:
    return [
        os.path.join(FLAGS.data_dir, 'validation-%05d-of-00128' % i)
        for i in range(0, 128)]


def dataset_parser(value, is_training):
  """Parse an Imagenet record from value."""
  keys_to_features = {
      'image/encoded':
          tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format':
          tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'image/class/label':
          tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
      'image/class/text':
          tf.FixedLenFeature([], dtype=tf.string, default_value=''),
      'image/object/bbox/xmin':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymin':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/xmax':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymax':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/class/label':
          tf.VarLenFeature(dtype=tf.int64),
  }

  parsed = tf.parse_single_example(value, keys_to_features)

  image = tf.image.decode_image(
      tf.reshape(parsed['image/encoded'], shape=[]),
      _NUM_CHANNELS)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  image = image_preprocessing_fn(
      image=image,
      output_height=network.default_image_size,
      output_width=network.default_image_size,
      is_training=is_training)

  label = tf.cast(
      tf.reshape(parsed['image/class/label'], shape=[]),
      dtype=tf.int32)

  return image, tf.one_hot(label, _LABEL_CLASSES)


def input_fn(is_training):
  """Input function which provides a single batch for train or eval."""
  files = filenames(is_training)

  
  dataset = tf.contrib.data.Dataset.from_tensor_slices(filenames(is_training))
  if is_training:
    dataset = dataset.shuffle(buffer_size=1024)
  dataset = dataset.flat_map(tf.contrib.data.TFRecordDataset)

  if is_training:
    dataset = dataset.repeat()

  dataset = dataset.map(lambda value: dataset_parser(value, is_training),
                        num_threads=FLAGS.map_threads,
                        output_buffer_size=FLAGS.batch_size)

  if is_training:
    buffer_size = 1250 + 2 * FLAGS.batch_size
    dataset = dataset.shuffle(buffer_size=buffer_size)

  iterator = dataset.batch(FLAGS.batch_size).make_one_shot_iterator()
  images, labels = iterator.get_next()
  #images = tf.reshape(images, [i]
  return images, labels

def save(saver, sess, logdir):
    #Saves the checkpoint file
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    saver.save(sess, checkpoint_path, write_meta_graph=False)


def resnet_model_fn(features, labels, mode):

  global load_done
  """Our model_fn for ResNet to be used with our Estimator."""
  tf.summary.image('images', features, max_outputs=6)
  logits = network(
      inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))


 ## All the required modifications are here.

 ## Since this routine uses tf.estimators to implement ResNet, if we would like to
 ## dump or reload the data, only method of communication is Checkpoints.
 ## Therefore, each commands are seperately called.



 ## Load the data##

 ## Executed only if the retrain flag is set
 ## Also, load happens only once in the retrain cycle
  RETRAIN = FLAGS.retrain
  if((load_done == 0) and RETRAIN):
        load_done = 1
        print("Loading pretrained weights")

        ## Load the modified/pre-trained weight values
        data = np.load("weights_imagenet.npy").item()
        addr = np.load("addr_table.npy").item()

        ## Path to the most recent Check-point file
        model_path = model_dir_saved +'/'+FLAGS.ckpt_file

        with tf.Session() as sess:

                ## All variables should be initialized
                sess.run(tf.global_variables_initializer())

                ## Define the Saver instance vbased on the check-point file
                saver = tf.train.Saver(tf.global_variables())
                saver.restore(sess, model_path)

                ## Get all the variable names in the required format
                model = get_weights(dump=False)
                get_sessions = model.keys()

                ## Go through every scope
                for i in get_sessions:
                  if 'global_step' not in i:
                    with tf.variable_scope(i, reuse=True):
                            ## Go through every variable in the scope, with Reuse
                            for val in model[i]:
                                print('Loading weight variable:    ' + val + '   in Scope:    ' + i)

                                ## Assign the loaded value to the weights
                                var = tf.get_variable(val.split(":")[0], trainable=False)
                                sess.run(var.assign(data[i][addr[val]]))

                ## Save the model in the Retrain directory
                saver.save(sess, FLAGS.model_dir + '/model.ckpt')
        print("data successfully loaded")

 ##Dump the Data##

 ## Set DUMP to False if you don't want to dump
 ## After the dump, the program ends
  DUMP = FLAGS.dump
  if ((mode == tf.estimator.ModeKeys.TRAIN) and (DUMP == True)):
        with tf.Session() as sess:
            print("Dumping weights now")
            weights_imagenet=dict()

            ## All variables should be initialized
            sess.run(tf.global_variables_initializer())

            ## Define the Saver instance vbased on the check-point file
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, FLAGS.model_dir +'/'+FLAGS.ckpt_file)

            ## Get all the variables
            model = get_weights(dump=True)
            get_sessions = model.keys()

            ## Go through every scope
            for i in get_sessions:
                if 'global_step' not in i:
                    with tf.variable_scope(i, reuse=True):
                            ## Go through every variable in the scope, with Reuse
                            layer_data=[]
                            for val in model[i]:
                                print('Dumping weight variable:    ' + val + '   in Scope:    ' + i)
                                ## Append each variable value to a file
                                layer_data.append(sess.run(tf.get_variable(val.split(":")[0])))
                            weights_imagenet[i]=layer_data

            ## Save them to a npy file
            np.save("weights_imagenet.npy",weights_imagenet)
            ## Exit the program
            sys.exit("Dump Finished, Exiting ...")

  predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  # Add weight decay to the loss. We perform weight decay on all trainable
  # variables, which includes batch norm beta and gamma variables.
  loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()

    # Multiply the learning rate by 0.1 at 30, 60, 120, and 150 epochs.
    boundaries = [
        int(batches_per_epoch * epoch) for epoch in [30, 60, 120, 150]]
    values = [
        _INITIAL_LEARNING_RATE * decay for decay in [1, 0.1, 0.01, 1e-3, 1e-4]]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32), boundaries, values)

    # Create a tensor named learning_rate for logging purposes.
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)
  else:
    train_op = None

  accuracy = tf.metrics.accuracy(
      tf.argmax(labels, axis=1), predictions['classes'])
  metrics = {'accuracy': accuracy}

  # Create a tensor named train_accuracy for logging purposes.
  tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', accuracy[1])

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  global load_done
  load_done = 0

  resnet_classifier = tf.estimator.Estimator(
      model_fn=resnet_model_fn, model_dir=FLAGS.model_dir)

  for _ in range(FLAGS.train_steps // FLAGS.steps_per_eval):
    tensors_to_log = {
        'learning_rate': 'learning_rate',
        'cross_entropy': 'cross_entropy',
        'train_accuracy': 'train_accuracy'
    }

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    print('Starting a training cycle.')
    resnet_classifier.train(
        input_fn=lambda: input_fn(True),
        steps=FLAGS.first_cycle_steps or FLAGS.steps_per_eval,
        hooks=[logging_hook])
    FLAGS.first_cycle_steps = None
    RETRAIN = FLAGS.retrain
    if((load_done==0) and RETRAIN):
        load_done=1
        reload_data(sess)


    print('Starting to evaluate.')
    eval_results = resnet_classifier.evaluate(input_fn=lambda: input_fn(False))
    print(eval_results)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
