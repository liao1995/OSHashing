import sys
from PIL import Image
import tensorflow as tf
import os
slim = tf.contrib.slim
import vgg
import scipy.io as sio
import numpy as np

data_path = 'database/data_list'
paths = list()
labels = list()
with open(data_path, 'r') as f:
  for line in f:
    paths.append(line.split('\t')[0])
    labels.append(int(line.split('\t')[1]))
print 'loaded {0} images'.format(len(paths))
paths = np.array(paths)
labels = np.array(labels)

ckpt_path = 'models/vgg_bbt_seg.ckpt-200000'
# parameters
n_classes = 15
# placeholder
x = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.uint8, [None, n_classes])
# model
pred, end_points = vgg.vgg_16(x, n_classes)
# test
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuray = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.device('/gpu:0'):
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True
  with tf.Session(config=config) as sess:
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)
    for i in range(len(paths)):
      img = Image.open(paths[i])
      batch_x = np.array(img.resize(tuple([224,224])), dtype=np.uint8);      
      batch_x = np.expand_dims(batch_x, 0)
      batch_y = slim.one_hot_encoding(labels[i], n_classes).eval(session=sess)
      batch_y = np.expand_dims(batch_y, 0)
      ep = sess.run(end_points, feed_dict={x: batch_x, y: batch_y})
      cnn_features = np.squeeze(ep['vgg_16/fc7'])
      dest_path = paths[i].replace('SegResults', 'SegCNNFeatures').replace('.jpg', '.mat')
      dest_dirname = os.path.dirname(dest_path)
      if not os.path.exists(dest_dirname): os.makedirs(dest_dirname)
      d = {}
      d['cnn_features'] = cnn_features
      sio.savemat(dest_path, d) 
      sys.stderr.write( 'written ' +  dest_path + '\n')
