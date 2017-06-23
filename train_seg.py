import os
import numpy as np
import tensorflow as tf
import seg_branch as osvos
from both_labels_dataset import Dataset
from datetime import datetime
import math
import sys
import random
slim = tf.contrib.slim

if __name__ == '__main__':
  seqname = 'pb_seg'
  gpu_id = sys.argv[1]
  sys.stderr.write('seqname now is: ' + seqname + ' gpu id is: ' + gpu_id + '\n')

root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(root_folder))
os.chdir(root_folder)
num_classes = 19

parent_path = os.path.join('models', 'OSVOS_parent', 'OSVOS_parent.ckpt-50000')

def start(num_classes, train_path, valid_path, parent_path, logs_path, seq_name):

  max_training_iters = 50000

  # Define BLDataset
  dataset = Dataset(train_path, None, './', store_memory=False, data_aug=False)
  valid_dataset = Dataset(valid_path, None, './', store_memory=False, data_aug=False)

  # Train the network
  #learning_rate = 2e-4 
  decay_steps = 10000
  save_step = max_training_iters
  side_supervision = 3
  display_step = 10
  batch_size = 45 
  starttime = datetime.now()
  with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
      global_step = tf.Variable(0, name='global_step', trainable=False)
      learning_rate = tf.train.exponential_decay(1e-8, global_step, 
                                    decay_steps=decay_steps,
                                    decay_rate=0.9)
                                    #staircase=True)
      osvos.train_finetune(dataset, valid_dataset, num_classes, parent_path, side_supervision, learning_rate, logs_path, max_training_iters, save_step, display_step, global_step, iter_mean_grad=1, ckpt_name=seq_name, batch_size = batch_size)
  endtime = datetime.now()
  print 'Over {0}: Escape time '.format(seq_name)
  print (endtime - starttime)

training_path = 'train_list_'
testing_path = 'valid_list_'
logs_root = 'logs/pb'
start (num_classes, training_path, testing_path, parent_path, logs_root, seqname)
