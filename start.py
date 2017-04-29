"""
Sergi Caelles (scaelles@vision.ee.ethz.ch)

This file is part of the OSVOS paper presented in:
    Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
    One-Shot Video Object Segmentation
    CVPR 2017
Please consider citing the paper if you use this code.
"""
import os
import numpy as np
import tensorflow as tf
import osvos
from dataset import Dataset


def start(train_image_path, train_label_path, test_images_path, result_path, parent_path, logs_path, seq_name):

  gpu_id = 0
  train_model = True
  max_training_iters = 500

  # Define Dataset
  test_frames = sorted(os.listdir(test_images_path))
  test_imgs = [os.path.join(test_images_path, frame) for frame in test_frames]
  if train_model:
    train_imgs = [train_image_path+'\t'+train_label_path]
    dataset = Dataset(train_imgs, test_imgs, './', data_aug=True)
  else:
    dataset = Dataset(None, test_imgs, './')

  # Train the network
  if train_model:
    # More training parameters
    learning_rate = 1e-8
    save_step = max_training_iters
    side_supervision = 3
    display_step = 10
    with tf.Graph().as_default():
      with tf.device('/gpu:' + str(gpu_id)):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        osvos.train_finetune(dataset, parent_path, side_supervision, learning_rate, logs_path, max_training_iters, save_step, display_step, global_step, iter_mean_grad=1, ckpt_name=seq_name)
  import datetime
  starttime = datetime.datetime.now()

  # Test the network
  with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
      checkpoint_path = os.path.join(logs_path, seq_name+'.ckpt-'+str(max_training_iters))
      osvos.test(dataset, checkpoint_path, result_path)
  endtime = datetime.datetime.now()
  print 'Over {0}: Escape time '.format(seq_name)
  print (endtime - starttime)
