import os
import sys
import numpy as np

database_root = 'database'
S_E = 'Annotations/The.Big.Bang.Theory.S01E01.720p.BluRay.x264-SiNNERS.mkv_0_33024'
train_list_file = os.path.join(database_root, 'hashing_train_list')
valid_list_file = os.path.join(database_root, 'hashing_valid_list')
#test_list_file = os.path.join(database_root, 'test_list')
#train_ratio = 0.7
#valid_ratio = 0.3


def get_label_dict():
  label_dict = dict()
  names = ['Althea', 
           'Dmitri',
           'Dr. Eric Gablehauser', 
           'Howard Wolowitz',
           'Kurt',
           'Lalita Gupta',
           'Leonard Hofstadter', 
           'Leslie Winkle', 
           'Mary Cooper',
           'Missy Cooper',
           'Penny',
           'Raj Koothrappali',
           'Sheldon Cooper',
           'Toby Loobenfeld',
           'Unknown']
  for i in range(len(names)):
    label_dict[names[i]] = i
  return names, label_dict


def write_list(a_list, filename):
  with open(filename, 'wb') as f:
    for line in a_list:
      f.write(line[0] + '\t' + line[1] + '\t' + str(line[2]) + '\n')           
  print 'Done written ' + filename + ': {0} files.'.format(len(a_list)) 
data_list = list()


names, label_dict = get_label_dict()
for name in names:
  name_folder = os.path.join(database_root, S_E, name)
  if not os.path.isdir(name_folder):
    sys.stderr.write('can not find directory ' + name_folder)
    sys.exit()
  seqs = os.listdir(name_folder)
  for seq in seqs:
    seq_folder = os.path.join(name_folder, seq)
    if not os.path.isdir(seq_folder): continue
    files = os.listdir(seq_folder)
    for img in files:
      anno_file = os.path.join(seq_folder, img)
      if not os.path.isdir(anno_file) and anno_file.find('.png') != -1:
        img_file = anno_file.replace('Annotations', 'Images').replace('.png', '.jpg')
        data_list.append([img_file, anno_file, label_dict[name]]) 

num_all_samples = len(data_list)
data_list = np.array(data_list)
write_list(data_list, train_list_file)
#np.random.shuffle(data_list)
#num_train_samples = int(num_all_samples * train_ratio)
#num_valid_samples = int(num_all_samples * valid_ratio)
#train_list = data_list[:num_train_samples]
#valid_list = data_list[num_train_samples:]
#test_list = data_list[num_train_samples+num_valid_samples:] 
#write_list(train_list, train_list_file)
#write_list(valid_list, valid_list_file)
#write_list(test_list, test_list_file)

