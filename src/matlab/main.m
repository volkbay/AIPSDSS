%% RASPI Initialize
clear 
close all

RP = raspi;
cam = cameraboard(RP,'Resolution','1920x1080','Quality',100);
cam.Sharpness = 100;
cam.MeteringMode = 'matrix';

%% Take Snapshot to RAW Image

%snapshot(cam);
%snapshot(cam);
raw = snapshot(cam);
raw = flip(raw,2);

imagesc(raw);

%% Pre-Processing RAW Image
img_raw = rgb2gray(raw);
threshold = graythresh(img_raw);
img_mask = imbinarize(img_raw,0.9);
img_mask = bwareafilt(img_mask,1);
img_mask = imdilate(img_mask,strel('disk',10));
img_no_light = img_raw;
img_no_light(img_mask) = uint8(255*threshold);
imshow(img_no_light);

% img_hist = histeq(img_no_light);
img_filt = edge(img_no_light,'Canny',0.01,'both','nothinning'); % 'canny'

img_show=rgb2gray(raw);
img_show(find(img_filt))=255;
imshow(img_show);

se_ver = strel('line',5,90);
se_hor = strel('line',5,0);

% img_first = imopen(bwareafilt(img_filt,[50 Inf],8),se_ver);
% img_last_ver = imopen(bwareafilt(img_first,[20 Inf],8),se_ver);

img_first = bwareafilt(imopen(img_filt,se_ver),[10 Inf],8);
img_last_ver = bwareafilt(imopen(img_first,se_ver),[20 Inf],8);

% img_first = imopen(bwareafilt(img_filt,[50 Inf],8),se_hor);
% img_last_hor = imopen(bwareafilt(img_first,[20 Inf],8),se_hor);

img_first = bwareafilt(imopen(img_filt,se_hor),[10 Inf],8);
img_last_hor = bwareafilt(imopen(img_first,se_hor),[20 Inf],8);

imshowpair(img_last_hor,img_last_ver,'montage')

img_last = or(img_last_ver,img_last_hor);
img_last = and(img_last,~imdilate(img_mask,strel('disk',10)));
imshow(img_last);
%% Hough Transform
hough_peaks_max = 10;
hough_threshold = 0.1;
hough_fill_gap = 200;
hough_min_length = 50;

[H,T,R] = hough(img_last);
P  = houghpeaks(H,hough_peaks_max,'threshold',ceil(hough_threshold*max(H(:))));

lines = houghlines(img_last,T,R,P,'FillGap',hough_fill_gap,'MinLength',hough_min_length);

%% Print Result (Lines)

imshow(rgb2gray(raw)), hold on
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

   % Plot beginnings and ends of lines
   plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
   plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
end
hold off

%% Alternative Result Showing (Boxes)
close all

black = zeros(size(raw,1)+20,size(raw,2)+20);
img_padded = cat(3,padarray(raw(:,:,1),[10 10],0,'both'),padarray(raw(:,:,2),[10 10],0,'both'),padarray(raw(:,:,3),[10 10],0,'both')); 

for k = 1:length(lines)
   xy = sort([lines(k).point1; lines(k).point2]);
   black(xy(1,2):xy(2,2)+30,xy(1,1):xy(2,1)+30) = 1;
end

black = (black == 1);
imshow(img_padded), hold on
box = regionprops(black,'BoundingBox');
for k = 1:size(box,1)
   rectangle('position',box(k).BoundingBox,'LineWidth',2,'EdgeColor','green');
end
hold off