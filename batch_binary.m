function y = b(f)
    f
    im = imread(f);
    im = rgb2gray(im);
    im(im>127) = 255;
    im(im<=127) = 0;
    imwrite(im, strrep(f, 'jpg', 'png'));
end

f = dir('*.jpg');
for i = 1:length(f)
    b(f(i).name);
end 

