img_hist = histeq(rgb2gray(raw2(:,:,:,30)));
img_filt = edge(img_hist,'sobel',[0.01]); % 'canny'

img_show=img_hist;
img_show(find(img_filt))=255;
imshow(img_show);

% imshow(bwareafilt(img_filt,[100 Inf],8));

se = strel('line',2,90);
img_first = imopen(bwareafilt(img_filt,[50 Inf],8),se);
img_last = imopen(bwareafilt(img_first,[20 Inf],8),se);

imshowpair(img_first,img_last,'montage')
%% Hough Transform
close all
[H,T,R] = hough(img_last);
P  = houghpeaks(H,200,'threshold',ceil(0.25*max(H(:))));

lines = houghlines(img_last,T,R,P,'FillGap',60,'MinLength',40);
figure, imshow(img_hist), hold on
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

   % Plot beginnings and ends of lines
   plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
   plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
end