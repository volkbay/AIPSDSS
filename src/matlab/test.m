raw = uint8(zeros(1080,1920,3,100));
tic;
for n = 1:100
    img = snapshot(cam);
    %raw(:,:,:,n) = flip(img,2);
    imagesc(flip(img,2));
end
toc
