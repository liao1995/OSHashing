import tensorflow as tf
slim = tf.contrib.slim
from datetime import datetime
import os
import numpy as np
import osvos
from single_label_dataset import Dataset
import sys

def my_test(dataset, num_classes, checkpoint_file, result_path, config=None):
    if config is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        config.allow_soft_placement = True
    tf.logging.set_verbosity(tf.logging.INFO)

    # Input data
    batch_size = 1
    input_image = tf.placeholder(tf.float32, [batch_size, None, None, 3])
    image_label = tf.placeholder(tf.float32, [batch_size, num_classes])

    # Create the cnn
    with slim.arg_scope(osvos.osvos_arg_scope()):
        net, fc, end_points = osvos.osvos(input_image, num_classes)
        print fc.shape
        image_label = tf.placeholder(tf.float32, [batch_size, num_classes])
        correct_pred = tf.equal(tf.argmax(fc, 1), tf.argmax(image_label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Create a saver to load the network
    saver = tf.train.Saver([v for v in tf.global_variables() if '-up' not in v.name and '-cr' not in v.name])

    step = 1
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_file)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        for frame in range(0, dataset.get_train_size()):
            img, label = dataset.next_batch(batch_size, 'train')
            image = osvos.preprocess_img(img[0])
            label = slim.one_hot_encoding(label[0], num_classes).eval(session=sess)
            label = np.expand_dims(label, 0)
            fc_r = sess.run(fc, feed_dict={input_image: image})
            #acc = sess.run(accuracy, feed_dict={input_image: image, image_label: label})
            print >> sys.stderr, "{} Iter {}: accuracy = {:.4f}".format(datetime.now(), step, acc)
            step += 1 


#train_path = 'database/total_bbt_list'
train_path = 'database/valid_list'
gpu_id = 0
num_classes = 15
checkpoint_file = 'test_dataset/logs_large/large_dataset_14957.ckpt-50000'
result_path = 'test_dataset/logs_large'

dataset = Dataset(train_path, None, None, './')
start = datetime.now()

with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        my_test(dataset, num_classes, checkpoint_file, result_path)
end = datetime.now()
print 'Escape time ' + str(end-start)

