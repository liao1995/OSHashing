import os
import numpy as np
import tensorflow as tf
import net 
from both_labels_dataset import Dataset
from datetime import datetime
import math
import sys
import random
slim = tf.contrib.slim

if __name__ == '__main__':
  seqname = 'bbt_aggr' + sys.argv[1]
  gpu_id = sys.argv[2]
  sys.stderr.write('seqname now is: ' + seqname + ' gpu id is: ' + gpu_id + '\n')

root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(root_folder))
os.chdir(root_folder)
num_classes = 14

image_train = False

#parent_path = os.path.join('models', 'OSVOS_parent', 'OSVOS_parent.ckpt-50000')
#parent_path = os.path.join('logs', 'bbt', 'bbt_val4.ckpt-6000')
parent_path = os.path.join('logs', 'bbt', 'bbt_val_3kft3.ckpt-3000')

def start(num_classes, train_path, valid_path, parent_path, logs_path, seq_name):

  max_training_iters = 3000

  # Define BLDataset
  dataset = Dataset(train_path, None, './', store_memory=False, data_aug=False)
  valid_dataset = Dataset(valid_path, None, './', store_memory=False, data_aug=False)

  # Train the network
  #learning_rate = 2e-4 
  decay_steps = 500
  save_step = max_training_iters
  side_supervision = 3
  display_step = 10
  starttime = datetime.now()
  with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
      global_step = tf.Variable(0, name='global_step', trainable=False)
      learning_rate = tf.train.exponential_decay(2e-4, global_step, 
                                    decay_steps=decay_steps,
                                    decay_rate=0.9)
                                    #staircase=True)
      net.train_finetune(dataset, valid_dataset, num_classes, parent_path, side_supervision, learning_rate, logs_path, max_training_iters, save_step, display_step, global_step, iter_mean_grad=1, ckpt_name=seq_name)
  endtime = datetime.now()
  print 'Over {0}: Escape time '.format(seq_name)
  print (endtime - starttime)


#def get_bbt_label_dict():
#    label_dict = dict()
#    names = ['Althea', 
#             'Dmitri',
#             'Dr. Eric Gablehauser', 
#             'Howard Wolowitz',
#             'Kurt',
#             'Lalita Gupta',
#             'Leonard Hofstadter', 
#             'Leslie Winkle', 
#             'Mary Cooper',
#             'Missy Cooper',
#             'Penny',
#             'Raj Koothrappali',
#             'Sheldon Cooper',
#             'Toby Loobenfeld']
#    for i in range(len(names)):
#      label_dict[names[i]] = i 
#    return names, label_dict
#
#names_, label_dict = get_bbt_label_dict()
#video_list = [None] * num_classes
#
#train_rate = 0.7
#valid_rate = 0.1
#
#root = 'database/Images'
#S_E = os.listdir(root)
#for s_e in S_E:
#    S_E_folder = os.path.join(root, s_e)
#    names = os.listdir(S_E_folder)
#    for name in names:
#        if name == 'Unknown': continue
#        name_folder = os.path.join(S_E_folder, name)
#        seqs = os.listdir(name_folder)
#        for seq in seqs:
#            seq_folder = os.path.join(name_folder, seq)
#            if video_list[label_dict[name]] == None: video_list[label_dict[name]] = set()
#            video_list[label_dict[name]].add(seq_folder + '\t' + str(label_dict[name]))
#print ('total data list details: \n')
#s = 0
#for i in range(len(video_list)):
#    l = len(video_list[i])
#    s += l
#    print (names_[i] + ': ' + str(l))
#print ('\ttotal: ' + str(s))
#
#print ('\n\ntraining data list details: \n')
#training_list = list()
#valid_list = list()
#testing_list = list()
#for i in range(len(video_list)):
#    sl = int(math.ceil(len(video_list[i]) * train_rate))
#    vl = int(math.ceil(len(video_list[i]) * valid_rate))
#    training_part = random.sample(video_list[i], sl)
#    valid_part = random.sample(video_list[i] - set(training_part), vl)
#    testing_part = video_list[i] - set(training_part) - set(valid_part)
#    print (names_[i]+'train '+str(sl)+' valid '+str(vl)+' test: ' +str(len(testing_part)))
#    training_list.extend(training_part)
#    valid_list.extend(valid_part)
#    testing_list.extend(list(testing_part))
#print ('\ttotal training: '+str(len(training_list))+
#        ' total valid: '+str(len(valid_list))+' total testing: '+str(len(testing_list)))
#
## construct training path and testing path
#training_path = list()
#valid_path = list()
#testing_path = list()
#for p in training_list:
#    seq_path = p.split('\t')[0]
#    label = p.split('\t')[1]
#    imgs = os.listdir(seq_path)
#    if image_train:
#        for img in imgs:
#            img_path = os.path.join(seq_path, img)
#            training_path.append(img_path+'\t'+
#                    img_path.replace('Images', 'OSFaceResults').replace('.jpg','.png')+'\t'+label)
#    else: training_path.append(seq_path+'\t'+seq_path.replace('Images','OSFaceResults')+'\t'+label)
##print ('training size: ' + str(len(training_path)))
#for p in valid_list:
#    seq_path = p.split('\t')[0]
#    label = p.split('\t')[1]
#    imgs = os.listdir(seq_path)
#    if image_train:
#        for img in imgs:
#            img_path = os.path.join(seq_path, img)
#            valid_path.append(img_path+'\t'+
#                    img_path.replace('Images', 'OSFaceResults').replace('.jpg','.png')+'\t'+label)
#    else: valid_path.append(seq_path+'\t'+seq_path.replace('Images','OSFaceResults')+'\t'+label)
#for p in testing_list:
#    seq_path = p.split('\t')[0]
#    label = p.split('\t')[1]
#    imgs = os.listdir(seq_path)
#    if image_train:
#        for img in imgs:
#            img_path = os.path.join(seq_path, img)
#            testing_path.append(img_path+'\t'+
#                    img_path.replace('Images', 'OSFaceResults').replace('.jpg','.png')+'\t'+label)    
#    else: testing_path.append(seq_path+'\t'+seq_path.replace('Images','OSFaceResults')+'\t'+label)
##print ('testing size:  ' + str(len(testing_path)))
#
#print ('write to file...')
#with open('train_list', 'w') as f:
#    for p in training_path:
#        f.write(p + '\n')
#with open('valid_list', 'w') as f:
#    for p in valid_path:
#        f.write(p + '\n')
#with open('test_list', 'w') as f:
#    for p in testing_path:
#        f.write(p + '\n')
#print ('write done.')

training_path = 'train_list'
testing_path = 'valid_list'
logs_root = 'logs/bbt_video'
start (num_classes, training_path, testing_path, parent_path, logs_root, seqname)
