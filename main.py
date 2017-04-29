import tensorflow as tf
slim = tf.contrib.slim
import os
import sys
from start import start
root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(root_folder))

os.chdir(root_folder)

parent_path = os.path.join('models', 'OSVOS_parent', 'OSVOS_parent.ckpt-50000')
train_root = 'Images'
anno_root = 'Annotations'
result_root = 'Results'
logs_root = 'models'

S_E = 'The.Big.Bang.Theory.S01E01.720p.BluRay.x264-SiNNERS.mkv_0_33024'

for name in os.listdir(os.path.join(train_root, S_E)):
  if not os.path.isdir(os.path.join(train_root, S_E, name)): continue
  result_name_path = os.path.join(result_root, S_E, name)
  if not os.path.isdir(result_name_path): os.makedirs(result_name_path)
  else: 
    print ('skip existed ', result_name_path)
    continue
  logs_name_path = os.path.join(logs_root, S_E, name)
  if not os.path.isdir(logs_name_path): os.makedirs(logs_name_path)
  for seq in os.listdir(os.path.join(train_root, S_E, name)):
    if not os.path.isdir(os.path.join(train_root, S_E, name, seq)): continue
    result_path = os.path.join(result_name_path, seq)
    if not os.path.isdir(result_path): os.mkdir(result_path)
    logs_path = os.path.join(logs_name_path, seq)
    if not os.path.isdir(logs_path): os.mkdir(logs_path) 
    seqname = name + '_' + seq
    # complete training pair path
    train_label_path = os.path.join(anno_root, S_E, name, seq)
    files = os.listdir(train_label_path)
    if len(files) != 1: raise Exception('annotation number error in ', train_label_path)
    filename = files[0][:-4] # no suffix
    test_images_path = os.path.join(train_root, S_E, name, seq)
    train_image_path = os.path.join(test_images_path, filename+'.jpg')
    train_label_path = os.path.join(anno_root, S_E, name, seq, filename+'.png')
    start (train_image_path, train_label_path, test_images_path, result_path, parent_path, logs_path, seqname)
