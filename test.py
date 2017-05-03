import tensorflow as tf
slim = tf.contrib.slim
import os
import sys
from start import start
root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(root_folder))

os.chdir(root_folder)
num_classes = 15

parent_path = os.path.join('models', 'OSVOS_parent', 'OSVOS_parent.ckpt-50000')
#train_img = 'test_dataset/7720.jpg'
#train_label = 'test_dataset/7720.png'
train_path = 'database/hashing_train_list'
test_img = 'test_dataset/test_imgs'
result_root = 'test_dataset/results'
logs_root = 'test_dataset/logs'
start (num_classes, train_path, test_img, result_root, parent_path, logs_root, 'small_dataset_330')
