ave_sens = 0;
ave_spec = 0;
ave_acc = 0;
num_samp = 508;
ave_cap = 0;
ave_main = 0;
ave_show = 0;
for n = 5:num_samp
%% Take Snapshot to RAW Image
capture_start = tic;
%raw = snapshot(cam); % Raspberry image acq
raw = imread(['D:\TEZ\FOTO\Dataset2\img\img' num2str(n,'%04d') '.jpg']); %Read an image
mask = imread(['D:\TEZ\FOTO\Dataset2\mask\mask' num2str(n,'%04d') '.jpg']); %Read an image
capture_time = toc(capture_start);
disp(['Capture =' num2str(capture_time)])
%% Pre-Processing RAW Image
main_start = tic;
img_raw = rgb2gray(raw); % Grayscale
threshold = graythresh(img_raw);
img_mask = imbinarize(img_raw,215/255);
img_mask = imdilate(img_mask,strel('square',3));
img_mask = bwareafilt(img_mask,1); % Find light source

% Cropping on light beam on the surface
box_init = regionprops(img_mask,'BoundingBox');
if size(box_init,1) == 0
    roi = [1 size(img_raw,1) 1 size(img_raw,2)];
else
    roi = floor([box_init.BoundingBox(2)+1 box_init.BoundingBox(2)+box_init.BoundingBox(4)-1 0 size(img_raw,2)-1])+1;
end
img_2_filt = img_raw(roi(1):roi(2),roi(3):roi(4));
img_2_filt = adapthisteq(img_2_filt,'NumTiles',[8 8],'ClipLimit',0.01);
%img_blur = imgaussfilt(img_2_filt,'FilterSize',[7 3]);
% Edge detection by Canny filter
img_filt = edge(img_2_filt,'Canny',[1/255 35/255]); % 'canny'
% Morphological Pre-processing
se1 = strel('line',3,90);
se2 = strel([ 1     1
              1     1
              1     1]);
img_open = imopen(img_filt,se1);
img_close = imclose(img_open,se2);
img_last = img_close;

%% Hough Transform
hough_peaks_max = 50; % Max number of peaks to be detected on the matrix
hough_threshold = 0.2; % Threshold for peak detection
hough_fill_gap = 200; % Gap tolerance on line detection
hough_min_length = 1; % Min length of lines to be detected

[H,T,R] = hough(img_close,'Theta',-20:20); % Hough transform 
P  = houghpeaks(H,hough_peaks_max,'threshold',size(img_close,1)/4); % Peak detection
if ~isempty(P)
    lines = houghlines(img_last,T,R,P,'FillGap',hough_fill_gap,'MinLength',hough_min_length); % Line extraction
else
    lines = [];
end
main_time = toc(main_start);
disp(main_time)
show_start = tic;
black = zeros(size(raw,1),size(raw,2));

for k = 1:length(lines)
   xy = sort([lines(k).point1; lines(k).point2]);
   black(roi(1)+xy(1,2):roi(1)+xy(2,2),xy(1,1):xy(2,1)) = 1;
end
% Draw rectangles
black = (black == 1);

figure(1),imshow(img_raw), hold on
box = regionprops(black,'BoundingBox');
for k = 1:size(box,1)
   rectangle('position',box(k).BoundingBox,'LineWidth',2,'EdgeColor','yellow');
end
hold off
fig = gca;
file = getframe(fig);
imwrite(file.cdata,['D:\TEZ\Tez Result 2\result' num2str(n,'%04d') '.jpg'])
imwrite(black,['D:\TEZ\Tez Result 2\mask' num2str(n,'%04d') '.jpg'])

show_time = toc(show_start);
disp(['Show =' num2str(show_time)])

TP = nnz(and(black,mask));
TN = nnz(and(~black,~mask));
FP = nnz(and(black,~mask));
FN = nnz(and(~black,mask));
if TP == 0 && FN==0
    Sens =0;
else
    Sens = TP / (TP+FN)*100;
end
Spec = TN / (TN+FP)*100;
Acc = (TP+TN)/(TP+TN+FN+FP)*100;

ave_cap = ave_cap + capture_time/num_samp;
ave_main = ave_main + main_time/num_samp;
ave_show  = ave_show + show_time/num_samp;
ave_sens = ave_sens + Sens/num_samp;
ave_spec = ave_spec + Spec/num_samp;
ave_acc = ave_acc + Acc/num_samp;
disp(n)
end