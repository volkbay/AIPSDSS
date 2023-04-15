%% MASKING from MATLAB LABELER
height = 480;
width = 640;
path = 'D:\TEZ\FOTO\Dataset2\mask\mask';
image_number = 508;

for i = 1:image_number
    maske = zeros(height,width);
    imwrite(maske,[path num2str(i,'%04d') '.jpg' ]);
end

for i = 1:size(positiveInstances,2)
    [~, name] = fileparts(positiveInstances(i).imageFilename);
    maske = zeros(height,width);
    box = positiveInstances(i).objectBoundingBoxes;
    for n = 1:size(box,1)
       maske(box(n,2):box(n,2)+box(n,4),box(n,1):box(n,1)+box(n,3)) = 1; 
    end
    imwrite(maske,[path name(end-3:end) '.jpg' ]);
end

%% XML MASKING

for m = 1:300
    fid=fopen(['D:\TEZ\FOTO\Dataset4\annot\scratches_' num2str(m) '.xml'],'r');
    A = textscan(fid,'%s');
    rect_list = [];
    line = zeros(1,4);
    for i = 1:size(A{1},1)
        if strncmp('<xmin>',A{1}(i),5)
            num = regexp(A{1}(i),'\d*','Match');
            line(1) = str2double(num{1,1});
            num = regexp(A{1}(i+1),'\d*','Match');
            line(2) = str2double(num{1,1});
            num = regexp(A{1}(i+2),'\d*','Match');
            line(3) = str2double(num{1,1});
            num = regexp(A{1}(i+3),'\d*','Match');
            line(4) = str2double(num{1,1});
            rect_list = [rect_list; line];
        end
    end

    mask_img = zeros(200,200);
    for n = 1:size(rect_list,1)
           mask_img(rect_list(n,2):rect_list(n,4),rect_list(n,1):rect_list(n,3)) = 1; 
    end

    imwrite(mask_img,['D:\TEZ\FOTO\Dataset4\mask\mask' num2str(m, '%04d') '.bmp' ]);
    fclose(fid);
end

%% Rename files
for m = 24:65
    movefile(['D:\TEZ\FOTO\Dataset3\mask\mask' num2str(m,'%d') '.jpg'],['D:\TEZ\FOTO\Dataset3\mask\mask' num2str(m,'%04d') '.jpg'])
end
%% Rename Files -2 
path = 'D:\TEZ\FOTO\Dataset1';
list = dir(path);
m = 1;
for i = 1:size(list,1)
   if list(i).isdir == 0
        movefile([path '\' list(i).name],[path '\dummy\img' num2str(m,'%04d') '.jpg'])
        m = m + 1;
   end
end