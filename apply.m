imgs_root = 'Images';
results_root = 'Results';
seg_root = 'SegResults';
S_E = 'The.Big.Bang.Theory.S01E01.720p.BluRay.x264-SiNNERS.mkv_0_33024';

names = dir(fullfile(imgs_root, S_E));
for i = 1 : length(names)
  if ~names(i).isdir || strcmpi(names(i).name, '.') || strcmpi(names(i).name, '..')
    continue;
  end
  name = names(i).name;
  if ~exist(fullfile(seg_root, S_E, name))
    mkdir(fullfile(seg_root, S_E, name));
  end    
  seqs = dir(fullfile(imgs_root, S_E, name));
  for j = 1 : length(seqs)
    if ~seqs(j).isdir || strcmpi(seqs(j).name, '.') || strcmpi(seqs(j).name, '..')
      continue;
    end
    seq_name = seqs(j).name;
    if ~exist(fullfile(seg_root, S_E, name, seq_name))
      mkdir(fullfile(seg_root, S_E, name, seq_name));
    end
    files = dir(fullfile(imgs_root, S_E, name, seq_name));
    for k = 1 : length(files)
      if files(k).isdir
        continue;
      end
      file_name = files(k).name;
      fullfile(imgs_root, S_E, name, seq_name, file_name)
      img = imread(fullfile(imgs_root, S_E, name, seq_name, file_name));
      pfile_name = sprintf('%spng', file_name(1:find(file_name=='.')));
      seg = imread(fullfile(results_root, S_E, name, seq_name, pfile_name));
      img_r = img(:,:,1);
      img_r(seg==0) = 0;
      img_g = img(:,:,2);
      img_g(seg==0) = 0;
      img_b = img(:,:,3);
      img_b(seg==0) = 0;
      seg_img(:,:,1) = img_r;
      seg_img(:,:,2) = img_g;
      seg_img(:,:,3) = img_b;
      imwrite(seg_img, fullfile(seg_root, S_E, name, seq_name, file_name));
      sprintf('writed %s', fullfile(seg_root, S_E, name, seq_name, file_name))
    end
  end
end

