import os
parent='/home/liao/data/codes/hash/cvc_spc/Release_v1.0_bbt/bbt_faceclips/The.Big.Bang.Theory.S01E01.720p.BluRay.x264-SiNNERS.mkv_0_33024'
destroot='/home/liao/data/codes/hash_liao/dataset/BBT/Annotations/The.Big.Bang.Theory.S01E01.720p.BluRay.x264-SiNNERS.mkv_0_33024'
for adir in os.listdir(parent):
  dir_full = os.path.join(parent, adir)
  if not os.path.isdir(adir):
    os.mkdir(os.path.join(destroot, adir))
  for subdir in os.listdir(dir_full):
    subdir_full = os.path.join(destroot, adir, subdir)
    print subdir_full
    if not os.path.isdir(subdir_full):
      os.makedirs(subdir_full)
