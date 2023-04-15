import time
import picamera
import picamera.array
import cv2

height = 480
width = 640
fps = 40
with picamera.PiCamera(sensor_mode=5,resolution=(width,height),framerate=fps) as camera:
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
        rawCapture = picamera.array.PiYUVArray(camera, size=(width, height))
        time.sleep(2)
	
	start = 0
        for frame  in camera.capture_continuous(rawCapture,format='yuv',use_video_port=True,burst=False):
                end = time.time()
		print("Total = ",(end-start))
		start = time.time() 
                cv2.imshow("Window", frame.array[:,:,0])
                if cv2.waitKey(1) == 27:
                        break
		show_time = time.time()
		print("Show = ",(show_time - start))
		print(camera.exposure_speed)
                rawCapture.truncate(0)
