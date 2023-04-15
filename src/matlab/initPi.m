clear 
close all

RP = raspi;
cam = cameraboard(RP,'Resolution','1920x1080','Quality',100);
cam.Sharpness = 100;
cam.MeteringMode = 'matrix';