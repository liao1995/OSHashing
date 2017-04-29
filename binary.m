srcdir = '/home/liao/data/codes/hash_liao/dataset/BBT/Annotations/The.Big.Bang.Theory.S01E01.720p.BluRay.x264-SiNNERS.mkv_0_33024/';
%seqname = 'Althea';
%seqname = 'Kurt';
%seqname='Leonard Hofstadter'
%seqname = 'Penny';
%seqname = 'Raj Koothrappali';
seqname = 'Sheldon Cooper';

files = dir(fullfile(srcdir, seqname, '*.jpg'));
for i = 1:length(files)
  imfile = fullfile(srcdir, seqname, files(i).name)
  im = imread(imfile);
  new_im = rgb2gray(im);
  new_im(new_im<127) = 0;
  new_im(new_im>=127) = 255;
  new_file = files(i).name;
  new_file = strcat(new_file(1:strfind(new_file, '.')), 'png');
  imwrite(new_im, fullfile(srcdir, seqname, new_file));
end
