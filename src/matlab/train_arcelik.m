clear
close all

image = load('raw3.mat')

img = raw(:,:,:,10);
detector = vision.CascadeObjectDetector('denemeTrain.xml');
bbox = step(detector,img);
detectedImg = insertObjectAnnotation(img,'rectangle',bbox,'Çizgi');
figure; imshow(detectedImg);