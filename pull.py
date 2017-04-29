import os
import shutil

srcroot = '/home/liao/data/codes/hash/cvc_spc/Release_v1.0_bbt/bbt_faceclips/The.Big.Bang.Theory.S01E01.720p.BluRay.x264-SiNNERS.mkv_0_33024/'
#seqname = 'Leonard Hofstadter'
#seqname = 'Penny'
#seqname = 'Raj Koothrappali'
seqname = 'Sheldon Cooper'

destroot = '/home/liao/data/codes/hash_liao/dataset/BBT/Annotations/The.Big.Bang.Theory.S01E01.720p.BluRay.x264-SiNNERS.mkv_0_33024/'

dirs = os.listdir(os.path.join(srcroot, seqname))

file2dir = dict()

for adir in dirs:
  dir_full = os.path.join(srcroot, seqname, adir)
  if os.path.isdir(dir_full):
    files = os.listdir(dir_full)
    files = sorted(files)
    file2dir[files[0][:-4]] = adir
    shutil.copy(os.path.join(dir_full, files[0]), os.path.join(srcroot, seqname))
  
  with open(os.path.join(os.path.join(destroot, seqname), 'map'), 'w') as f:
    for key in file2dir:
      f.write(key + '\t' + file2dir[key] + '\n')
  
