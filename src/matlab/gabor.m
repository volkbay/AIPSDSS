%% Uploading input
image = im2double(raw(:,:,:,30));
 
%figure(1);
%imshow(image);     
 
G=rgb2gray(image);
Grey=histeq(G);
[m,n]=size(Grey);
row_photo=m;
colomn_photo=n;
del=10;
 
a=zeros(row_photo,colomn_photo);
b=zeros(row_photo,colomn_photo);
c=zeros(row_photo,colomn_photo);
d=zeros(row_photo,colomn_photo);
e=zeros(row_photo,colomn_photo);
f=zeros(row_photo,colomn_photo);
 
% Gabor filtering
 
gb=zeros(11,11);
 
gamma=0.3; %aspect ratio
psi=100; %phase
theta=0; %orientation
bw=1/7; %bandwidth or effective width
lambda=0.365; % wavelength
pi=180;
 
for p=1:6
     
for x=1:11
   for y=1:11
        x_theta=x*cosd(theta)/del+y*sind(theta)/del;
        y_theta=-x*sind(theta)/del+y*cosd(theta)/del;
        gb(x,y)= exp(-0.5*(x_theta.^2*bw^2+y_theta.^2*bw^2))*cosd(2*pi*x_theta/lambda+psi);
   end
end
 
for i=1:row_photo
    for j=1:colomn_photo
        for m=1:11;
            for n=1:11;
                ii = i+(m-10);
                jj = j+(n-10);
                if( (ii > 0) && (ii < row_photo+1) && (jj > 0) && (jj < colomn_photo+1))
                    if ( theta == 0 )
                    a(i,j) = a(i,j) + Grey(ii,jj)* gb(m,n); 
                    end
                    if (theta == 36 )
                    b(i,j) = b(i,j) + Grey(ii,jj)* gb(m,n); 
                    end
                    if (theta == 72 )
                    c(i,j) = c(i,j) + Grey(ii,jj)* gb(m,n); 
                    end
                    if (theta == 108 )
                    d(i,j) = d(i,j) + Grey(ii,jj)* gb(m,n); 
                    end
                    if (theta == 144 )
                    e(i,j) = e(i,j) + Grey(ii,jj)* gb(m,n); 
                    end
                    if (theta == 180 )
                    f(i,j) = f(i,j) + Grey(ii,jj)* gb(m,n); 
                    end
                end
            end
        end
    end
end
 
%changing the orientation
theta = theta + 36 ;
end
% img = sum(abs(gb).^2, 3).^0.5;
% % default superposition method, L2-norm
% img_out = img./max(img(:));
 
% normalize
% imshow(img_out);
 
% %showing results
% figure
% imshow(a);
% figure
% imshow(b);
% figure
% imshow(c);
% figure
% imshow(d);
% figure
% imshow(e);
% figure
% imshow(f);
 
%% Inputs
 
Filter_Length = 5 ; 
kCenterX = (Filter_Length+1) / 2;
kCenterY = (Filter_Length+1) / 2;
Kernel_sigma=1.2;
Kernel_coefficients=zeros(Filter_Length,Filter_Length);
 
a_thres=zeros(row_photo,colomn_photo);
b_thres=zeros(row_photo,colomn_photo);
c_thres=zeros(row_photo,colomn_photo);
d_thres=zeros(row_photo,colomn_photo);
e_thres=zeros(row_photo,colomn_photo);
f_thres=zeros(row_photo,colomn_photo);
 
a_thres1=zeros(row_photo,colomn_photo);
b_thres1=zeros(row_photo,colomn_photo);
c_thres1=zeros(row_photo,colomn_photo);
d_thres1=zeros(row_photo,colomn_photo);
e_thres1=zeros(row_photo,colomn_photo);
f_thres1=zeros(row_photo,colomn_photo);
 
a_thres2=zeros(row_photo,colomn_photo);
b_thres2=zeros(row_photo,colomn_photo);
c_thres2=zeros(row_photo,colomn_photo);
d_thres2=zeros(row_photo,colomn_photo);
e_thres2=zeros(row_photo,colomn_photo);
f_thres2=zeros(row_photo,colomn_photo);
 
a_scale=zeros(row_photo,colomn_photo);
b_scale=zeros(row_photo,colomn_photo);
c_scale=zeros(row_photo,colomn_photo);
d_scale=zeros(row_photo,colomn_photo);
e_scale=zeros(row_photo,colomn_photo);
f_scale=zeros(row_photo,colomn_photo);
h=zeros(row_photo,colomn_photo);
h_scale=zeros(row_photo,colomn_photo);
h_ori=zeros(row_photo,colomn_photo);
x_ori=zeros(row_photo,colomn_photo);
y_ori=zeros(row_photo,colomn_photo);
z_ori=zeros(row_photo,colomn_photo);
 
sum1=0; 
sum2=0; 
sum3=0; 
sum4=0; 
sum5=0; 
sum6 =0;
mean1=0; 
mean2=0; 
mean3=0; 
mean4=0; 
mean5=0; 
mean6 = 0 ;
 
sum01=0; 
sum02=0; 
sum03=0; 
sum04=0; 
sum05=0; 
sum06 =0;
mean01=0; 
mean02=0; 
mean03=0; 
mean04=0; 
mean05=0; 
mean06 = 0 ;
 
sum001=0; 
sum002=0; 
sum003=0; 
sum004=0; 
sum005=0; 
sum006 =0;
mean001=0; 
mean002=0; 
mean003=0; 
mean004=0; 
mean005=0; 
mean006 = 0 ;
 
%calculating the means and standard deviation for thresholding
for i=1:row_photo
    for j=1:colomn_photo
          sum1 = sum1 + a(i,j);
          mean1 = sum1 / (row_photo * colomn_photo);
          sum2 = sum2 + b(i,j);
          mean2 = sum2 / (row_photo * colomn_photo);
          sum3 = sum3 + c(i,j);
          mean3 = sum3 / (row_photo * colomn_photo);
          sum4 = sum4 + d(i,j);
          mean4 = sum4 / (row_photo * colomn_photo);
          sum5 = sum5 + e(i,j);
          mean5 = sum5 / (row_photo * colomn_photo);
          sum6 = sum6 + f(i,j);
          mean6 = sum6 / (row_photo * colomn_photo);
    end
end
 
stddev1 = std2(a);
stddev2 = std2(b);
stddev3 = std2(c);
stddev4 = std2(d);
stddev5 = std2(e);
stddev6 = std2(f);
 
%% Creating two more scales
% %Kernel Coefficients
% for i=1:Filter_Length
%     for j=1:Filter_Length
%         Kernel_coefficients(i,j)=(exp(-((((Filter_Length+1)/2)-i).^2+(((Filter_Length+1)/2)-j).^2)/(2*Kernel_sigma.^2)))/(2*pi*Kernel_sigma.^2);
%     end
% end
% 
% %2D Convolution
% 
% for i=1:row_photo
%     for j=1:colomn_photo
%         for m=1:Filter_Length;
% %             mm = Filter_Length+1-m;
%             for n=1:Filter_Length
% %                 nn = Filter_Length+1-n;  
%                 ii = i+(m-kCenterY);
%                 jj = j+(n-kCenterX);
% 
%                 if( (ii > 0) && (ii < row_photo+1) && (jj > 0) && (jj < colomn_photo+1) )
%                     a1(i,j) = a1(i,j) + a(ii,jj)* Kernel_coefficients(m,n);
%                     b1(i,j) = b1(i,j) + b(ii,jj)* Kernel_coefficients(m,n);
%                     c1(i,j) = c1(i,j) + c(ii,jj)* Kernel_coefficients(m,n);
%                     d1(i,j) = d1(i,j) + d(ii,jj)* Kernel_coefficients(m,n);
%                     e1(i,j) = e1(i,j) + e(ii,jj)* Kernel_coefficients(m,n);
%                     f1(i,j) = f1(i,j) + f(ii,jj)* Kernel_coefficients(m,n);
%                 end
%             end
%         end
%     end
% end
% 
% for i=1:row_photo
%     for j=1:colomn_photo
%         for m=1:Filter_Length;
% %             mm = Filter_Length+1-m;
%             for n=1:Filter_Length
% %                 nn = Filter_Length+1-n;  
%                 ii = i+(m-kCenterY);
%                 jj = j+(n-kCenterX);
% 
%                 if( (ii > 0) && (ii < row_photo+1) && (jj > 0) && (jj < colomn_photo+1) )
%                     a2(i,j) = a2(i,j) + a1(ii,jj)* Kernel_coefficients(m,n);
%                     b2(i,j) = b2(i,j) + b1(ii,jj)* Kernel_coefficients(m,n);
%                     c2(i,j) = c2(i,j) + c1(ii,jj)* Kernel_coefficients(m,n);
%                     d2(i,j) = d2(i,j) + d1(ii,jj)* Kernel_coefficients(m,n);
%                     e2(i,j) = e2(i,j) + e1(ii,jj)* Kernel_coefficients(m,n);
%                     f2(i,j) = f2(i,j) + f1(ii,jj)* Kernel_coefficients(m,n);
%                 end
%             end
%         end
%     end
% end
% C=uint8(a1);
% figure
% imshow(C)
 
%second scale
a1 = imgaussfilt(a);
b1 = imgaussfilt(b);
c1 = imgaussfilt(c);
d1 = imgaussfilt(d);
e1 = imgaussfilt(e);
f1 = imgaussfilt(f);
%third scale
 
a2 = imgaussfilt(a1);
b2 = imgaussfilt(b1);
c2 = imgaussfilt(c1);
d2 = imgaussfilt(d1);
e2 = imgaussfilt(e1);
f2 = imgaussfilt(f1);
%% Calculating the means and standard deviation for thresholding for other
%%scales
 
for i=1:row_photo
    for j=1:colomn_photo
          sum01 = sum01 + a(i,j);
          mean01 = sum01 / (row_photo * colomn_photo);
          sum02 = sum02 + b(i,j);
          mean02 = sum02 / (row_photo * colomn_photo);
          sum03 = sum03 + c(i,j);
          mean03 = sum03 / (row_photo * colomn_photo);
          sum04 = sum04 + d(i,j);
          mean04 = sum04 / (row_photo * colomn_photo);
          sum05 = sum05 + e(i,j);
          mean05 = sum05 / (row_photo * colomn_photo);
          sum06 = sum06 + f(i,j);
          mean06 = sum06 / (row_photo * colomn_photo);
          sum001 = sum001 + a(i,j);
          mean001 = sum001 / (row_photo * colomn_photo);
          sum002 = sum002 + b(i,j);
          mean002 = sum002 / (row_photo * colomn_photo);
          sum003 = sum003 + c(i,j);
          mean003 = sum003 / (row_photo * colomn_photo);
          sum004 = sum004 + d(i,j);
          mean004 = sum004 / (row_photo * colomn_photo);
          sum005 = sum005 + e(i,j);
          mean005 = sum005 / (row_photo * colomn_photo);
          sum006 = sum006 + f(i,j);
          mean006 = sum006 / (row_photo * colomn_photo);
     
    end
end
 
%second scale
stddev01 = std2(a1);
stddev02 = std2(b1);
stddev03 = std2(c1);
stddev04 = std2(d1);
stddev05 = std2(e1);
stddev06 = std2(f1);
 
%third scale
 
stddev001 = std2(a2);
stddev002 = std2(b2);
stddev003 = std2(c2);
stddev004 = std2(d2);
stddev005 = std2(e2);
stddev006 = std2(f2);
 
%% thresholding for feature difference extraction
for i=1:row_photo
    for j=1:colomn_photo
            if ((a(i, j) - mean1) <= (7 * stddev1))
                a_thres(i, j) = 0;
            end
            if ((a(i, j) - mean1) >(7 * stddev1))
                a_thres(i, j) = 255;
            end
            if ((b(i, j) - mean2) <= (7 * stddev2))
                b_thres(i, j) = 0;
            end
            if ((b(i, j) - mean2) >(7 * stddev2))
                b_thres(i, j) = 255;  
            end
            if ((c(i, j) - mean3) <= (7 * stddev3))
                c_thres(i, j) = 0;
            end
            if ((c(i, j) - mean3) >(7 * stddev3))
                c_thres(i, j) = 255;
            end
            if ((d(i, j) - mean4) <= (7 * stddev4))
                d_thres(i, j) = 0;
            end
            if ((d(i, j) - mean4) >(7 * stddev4))
                d_thres(i, j) = 255;
            end
            if ((e(i, j) - mean5) <= (7 * stddev5))
                e_thres(i, j) = 0;
            end
            if ((e(i, j) - mean5) >(7 * stddev5))
                e_thres(i, j) = 255;
            end
            if ((f(i, j) - mean6) <= (7 * stddev6))
                f_thres(i, j) = 0;
            end
            if ((f(i, j) - mean6) >(7 * stddev6))
                f_thres(i, j) = 255;
            end
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if ((a1(i, j) - mean01) <= (7 * stddev01))
                a_thres1(i, j) = 0;
            end
            if ((a1(i, j) - mean01) >(7 * stddev01))
                a_thres1(i, j) = 255;
            end
            if ((b1(i, j) - mean02) <= (7 * stddev02))
                b_thres1(i, j) = 0;
            end
            if ((b1(i, j) - mean02) >(7 * stddev02))
                b_thres1(i, j) = 255; 
            end
            if ((c1(i, j) - mean03) <= (7 * stddev03))
                c_thres1(i, j) = 0;
            end
            if ((c1(i, j) - mean03) >(7 * stddev03))
                c_thres1(i, j) = 255;
            end
            if ((d1(i, j) - mean04) <= (7 * stddev04))
                d_thres1(i, j) = 0;
            end
            if ((d1(i, j) - mean04) >(7 * stddev04))
                d_thres1(i, j) = 255;
            end
            if ((e1(i, j) - mean05) <= (7 * stddev05))
                e_thres1(i, j) = 0;
            end
            if ((e1(i, j) - mean05) >(7 * stddev05))
                e_thres1(i, j) = 255;
            end
            if ((f1(i, j) - mean06) <= (7 * stddev06))
                f_thres1(i, j) = 0;
            end
            if ((f1(i, j) - mean06) >(7 * stddev06))
                f_thres1(i, j) = 255;
            end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if ((a2(i, j) - mean001) <= (7 * stddev001))
                a_thres2(i, j) = 0;
            end
            if ((a(i, j) - mean001) >(7 * stddev001))
                a_thres2(i, j) = 255;
            end
            if ((b2(i, j) - mean002) <= (7 * stddev002))
                b_thres2(i, j) = 0;
            end
            if ((b2(i, j) - mean002) >(7 * stddev002))
                b_thres2(i, j) = 255; 
            end
            if ((c2(i, j) - mean003) <= (7 * stddev003))
                c_thres2(i, j) = 0;
            end
            if ((c2(i, j) - mean003) >(7 * stddev003))
                c_thres2(i, j) = 255;
            end
            if ((d2(i, j) - mean004) <= (7 * stddev004))
                d_thres2(i, j) = 0;
            end
            if ((d2(i, j) - mean004) >(7 * stddev004))
                d_thres2(i, j) = 255;
            end
            if ((e2(i, j) - mean005) <= (7 * stddev005))
                e_thres2(i, j) = 0;
            end
            if ((e2(i, j) - mean005) >(7 * stddev005))
                e_thres2(i, j) = 255;
            end
            if ((f2(i, j) - mean006) <= (7 * stddev006))
                f_thres2(i, j) = 0;
            end
            if ((f2(i, j) - mean006) >(7 * stddev006))
                f_thres2(i, j) = 255;
            end
    end
end
% figure
% imshow(a_thres);
% figure
% imshow(a_thres1);
% figure
% imshow(a_thres2);
 
%%  mean of the images with same orientation and same scale seperately (data fusion)
for i=1:row_photo
    for j=1:colomn_photo
            if (a_thres(i, j) + b_thres(i, j) + c_thres(i, j) + d_thres(i, j) + e_thres(i, j) + f_thres(i, j) >= (255 * 3))
                x_ori(i, j) = 255;
            end
            if (a_thres1(i, j) + b_thres1(i, j) + c_thres1(i, j) + d_thres1(i, j) + e_thres1(i, j) + f_thres1(i, j) >= (255 * 3))
                y_ori(i, j) = 255;
            end
            if (a_thres2(i, j) + b_thres2(i, j) + c_thres2(i, j) + d_thres2(i, j) + e_thres2(i, j) + f_thres2(i, j) >= (255 * 3))
                z_ori(i, j) = 255;
            end
            h_ori(i, j) = (x_ori(i, j) + y_ori(i, j) + z_ori(i, j)) / 3;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
            if (a_thres(i, j) + a_thres1(i, j) + a_thres2(i, j) >= (255 * 2))
                a_scale(i, j) = 255;
            end
            if (b_thres(i, j) + b_thres1(i, j) + b_thres2(i, j) >= (255 * 2))
                b_scale(i, j) = 255;
            end
            if (c_thres(i, j) + c_thres1(i, j) + c_thres2(i, j) >= (255 * 2))
                c_scale(i, j) = 255;
            end
            if (d_thres(i, j) + d_thres1(i, j) + d_thres2(i, j) >= (255 * 2))
                d_scale(i, j) = 255;
            end
            if (e_thres(i, j) + e_thres1(i, j) + e_thres2(i, j) >= (255 * 2))
                e_scale(i, j) = 255;
            end
            if (f_thres(i, j) + f_thres1(i, j) + f_thres2(i, j) >= (255 * 2))
                f_scale(i, j) = 255;
            end
            h_scale(i, j) = (a_scale(i, j) + b_scale(i, j) + c_scale(i, j) + d_scale(i, j) + e_scale(i, j) + f_scale(i, j)) / 6;
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            h(i, j) = (h_ori(i, j) + h_scale(i, j)) / 2;
            if (h(i, j) > 0)
                h(i, j) = 255;
            end
    end
end
 
%figure
%imshow(h_ori);
%figure
%imshow(h_scale);
%figure
%imshow(h);
%% erosion
 
se_ver = strel('line',11,90);
h1 = imdilate(h, se_ver);
h2 = imerode(h1,se_ver);
 
%showing output
figure
imshow(image);
figure    
imshow(h1);
figure
imshow(h2);