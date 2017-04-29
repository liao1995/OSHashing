import os
import shutil

root = '/home/liao/data/codes/hash_liao/dataset/BBT/Annotations/The.Big.Bang.Theory.S01E01.720p.BluRay.x264-SiNNERS.mkv_0_33024/'
#seqname = 'Leonard Hofstadter'
#seqname = 'Penny'
#seqname = 'Raj Koothrappali'
seqname = 'Sheldon Cooper'
with open(os.path.join(root, seqname, 'map'), 'r') as f:
  for line in f:
    line = line.rstrip()
    image = line.split('\t')[0] + '.png'
    dirname = line.split('\t')[1]
    src = os.path.join(root, seqname, image)
    dest = os.path.join(root, seqname, dirname, image)
    shutil.move(src, dest)
    print 'move {0} to {1}\n'.format(image, os.path.join(dirname, image))
