import sys
import tensorflow as tf
import os
slim = tf.contrib.slim
import vgg
from single_label_dataset import Dataset

test_path = 'database/test_list'
dataset = Dataset(test_path, None, None,  '.', True)
ckpt_path = 'models/vgg_bbt.ckpt-200000'
# hyperparameters
batch_size = 32
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
    #saver = tf.train.import_meta_graph(ckpt_path + '.meta')
    saver.restore(sess, ckpt_path)
    step = 1
    while step * batch_size < dataset.get_train_size():
      batch_x, batch_y = dataset.next_batch(batch_size, 'train')
      batch_y = slim.one_hot_encoding(batch_y, n_classes).eval(session=sess)
      acc = sess.run(accuray, feed_dict={x: batch_x, y: batch_y})
#      print end_points['vgg_16/fc7']
      sys.stderr.write( 'Iter ' + str(step*batch_size) +  '  accuracy = {:.5f}'.format(acc) + '\n')
      step += 1
