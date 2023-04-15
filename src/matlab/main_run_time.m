for n = 1:65
%% Take Snapshot to RAW Image
tic;
%raw = snapshot(cam); % Raspberry image acq
raw = raw4(:,:,:,n); % Sample image acq

% Frame difference
if n > 1
    if sum(sum(abs(rgb2gray(raw) - rgb2gray(raw4(:,:,:,n-1))))) < size(raw,1) * size(raw,2) * 255 / 200
        raw = previous_frame; % Do not update image
    else
        previous_frame = raw; % Update image
    end
else
    previous_frame = raw; % For the very first image
end
% Flip and Rotate
raw = flip(raw,2);
raw = imrotate(raw,0);
pause(0.01);
%% Pre-Processing RAW Image
img_raw = rgb2gray(raw); % Grayscale
threshold = graythresh(img_raw);
img_mask = imbinarize(img_raw,0.9);
img_mask = bwareafilt(img_mask,1); % Find light source
img_mask = imdilate(img_mask,strel('disk',10));
img_no_light = img_raw;
img_no_light(img_mask) = uint8(255*threshold); % Mask light source

% Cropping on light beam on the surface
box_init = regionprops(img_mask,'BoundingBox');
if size(box_init,1) == 0
    roi = [1 size(img_raw,1) 1 size(img_raw,2)];
else
    roi = floor([box_init.BoundingBox(2)+1 box_init.BoundingBox(2)+box_init.BoundingBox(4)-1 0 size(img_raw,2)-1])+1;
end
img_2_filt = img_no_light(roi(1):roi(2),roi(3):roi(4));

% Edge detection by Canny filter
img_filt = edge(img_2_filt,'Canny',0.01,'both','nothinning'); % 'canny'

% Morphological Pre-processing
se_ver = strel('line',5,90);
se_hor = strel('line',5,0);

img_first = bwareafilt(imopen(img_filt,se_ver),[10 Inf],8);
img_last_ver = bwareafilt(imopen(img_first,se_ver),[20 Inf],8); % Vertical details

img_first = bwareafilt(imopen(img_filt,se_hor),[10 Inf],8);
img_last_hor = bwareafilt(imopen(img_first,se_hor),[20 Inf],8); % Horizontal details

img_last = or(img_last_ver,img_last_hor); % Combine details
img_last = and(imdilate(img_last,se_hor),~imdilate(img_mask(roi(1):roi(2),roi(3):roi(4)),strel('disk',10))); % Highlight details and erase light source edges
%% Hough Transform
hough_peaks_max = 10; % Max number of peaks to be detected on the matrix
hough_threshold = 0.2; % Threshold for peak detection
hough_fill_gap = 25; % Gap tolerance on line detection
hough_min_length = 75; % Min length of lines to be detected

[H,T,R] = hough(img_last,'Theta',-10:0.1:10); % Hough transform 
P  = houghpeaks(H,hough_peaks_max,'threshold',ceil(hough_threshold*max(H(:)))); % Peak detection
lines = houghlines(img_last,T,R,P,'FillGap',hough_fill_gap,'MinLength',hough_min_length); % Line extraction

% Result Showing - 1 (Non-overlapping boxes)
% img_2_show = img_raw(roi(1):roi(2),roi(3):roi(4));
% img_padded = cat(3,padarray(img_2_show(:,:),[10 10],0,'both'),padarray(img_2_show(:,:),[10 10],0,'both'),padarray(img_2_show(:,:),[10 10],0,'both')); 
% 
% figure(1)
% imshow(img_padded), hold on
% for k = 1:length(lines)
%    xy = sort([lines(k).point1; lines(k).point2]);
%    rectangle('position',[xy(1) xy(3) abs(xy(1)-xy(2))+20 abs(xy(4)-xy(3))+20],'LineWidth',2,'EdgeColor',[round(rand) round(rand) round(rand)]);
% end
% hold off

% Result Showing - 2 (Overlapping boxes)
img_2_show = img_raw(roi(1):roi(2),roi(3):roi(4)); % Show image
black = zeros(size(raw,1)+20,size(raw,2)+20);
img_padded = cat(3,padarray(img_2_show(:,:),[10 10],0,'both'),padarray(img_2_show(:,:),[10 10],0,'both'),padarray(img_2_show(:,:),[10 10],0,'both')); 

for k = 1:length(lines)
   xy = sort([lines(k).point1; lines(k).point2]);
   black(xy(1,2):xy(2,2)+30,xy(1,1):xy(2,1)+30) = 1;
end
% Draw rectangles
black = (black == 1);
figure(1),imshow(img_padded), hold on
box = regionprops(black,'BoundingBox');
for k = 1:size(box,1)
   rectangle('position',box(k).BoundingBox,'LineWidth',2,'EdgeColor','green');
end
hold off

toc
end