import RPi.GPIO as GPIO
import time
import os
from picamera import PiCamera

GPIO.setmode(GPIO.BCM)
led_pin = 4
button_pin = 17
GPIO.setup(led_pin, GPIO.OUT)
GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setwarnings(False)

height = 480
width = 640
fps = 40
with PiCamera(sensor_mode=5,resolution=(width,height),framerate=fps) as camera:
        camera.awb_mode = 'fluorescent'
        camera.brightness = 50
        #camera.color_effects = (128,128)
        camera.contrast = 0
        camera.exposure_compensation = 25
        camera.exposure_mode = 'backlight'
        camera.image_denoise = True
        camera.image_effect = 'denoise'
        camera.meter_mode = 'backlit'
        camera.iso = 100
        camera.drc_strength = 'high'
        camera.saturation = 0
        camera.sharpness = 100
        camera.shutter_speed = 2500
        camera.still_stats = False
        camera.video_denoise = True
        camera.video_stabilization = True
        print(camera.resolution,camera.framerate)
        time.sleep(2)
        
        folder_counter = 1
        while True:
            if GPIO.HIGH == GPIO.input(button_pin):
                try:
                    date_list = time.localtime()
                    path = '/media/usb/Rasp/Images%d_%s-%s-%s_%s-%s-%s' % (folder_counter,date_list[2],date_list[1],date_list[0],date_list[3],date_list[4],date_list[5])
                    folder_counter = folder_counter + 1
                    os.mkdir(path)
                    i = 0
                    start = time.time()
                    for i, filename in enumerate(
                            camera.capture_continuous('%s/image{counter:03d}.bmp' % path,use_video_port=True,burst=False)):
                        print(filename)
                        GPIO.output(led_pin, GPIO.HIGH)
                        if i == 99:
                            break
                        end = time.time()
                        print("Total =",(end-start))
                        start = time.time()
                except:
                    print "create folder failed"
                    time.sleep(0.5)
                    continue
            else:
                GPIO.output(led_pin, GPIO.LOW)
