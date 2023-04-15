import cv2
import RPi.GPIO as GPIO
import time
import numpy as np
from picamera import PiCamera
from picamera.array import PiYUVArray

###################################################################################################
def main():
    GPIO.setmode(GPIO.BCM)
    led_pin = 4
    GPIO.setup(led_pin, GPIO.OUT)
    GPIO.setwarnings(False)

    height = 480
    width = 640
    fps = 40
    with PiCamera(sensor_mode=5,resolution=(width,height),framerate=fps) as camera:
        camera.awb_mode = 'fluorescent'
        camera.brightness = 50
        #camera.color_effects = (128,128)
        camera.contrast = 0
        camera.exposure_compensation = 0
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
        rawCapture = PiYUVArray(camera, size=(width, height))
        time.sleep(2)
            
	t1=0
	for frame  in camera.capture_continuous(rawCapture,format='yuv',use_video_port=True):
            t4 = time.time()
	    print("Total Time =",t4-t1)
	    t1 = time.time()
	    gray = frame.array[:,:,0]
	    mask = cv2.inRange(gray, 215,255)
            mask = cv2.dilate(mask, None, iterations=1)

            cnts = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[1]
            
            if len(cnts) != 0:
                areas = [cv2.contourArea(c) for c in cnts]
                max_index = np.argmax(areas)
                large=cnts[max_index]
                x,y,w,h = cv2.boundingRect(large)
                cropped = gray[y:y+h,:]
	    cropped = cv2.equalizeHist(cropped)
	    blurred = cv2.GaussianBlur(cropped, (5,3), 0)
            edges = cv2.Canny(blurred,1,35)
	    #edges[:,x:x+w] = 0
	    kernel = np.ones((3,1),np.uint8)
	    kernel2 = np.ones((1,3),np.uint8)
	    imopen = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
	    imclose = cv2.morphologyEx(imopen, cv2.MORPH_CLOSE, kernel2)
	    
            lines = cv2.HoughLines(imclose,1,np.pi/180,h/3)
	    GPIO.output(led_pin, GPIO.LOW)
	    t2 = time.time()
	    print("Calculation Time = ",t2-t1)
	    #cv2.namedWindow("Result2", cv2.WINDOW_AUTOSIZE)
            #cv2.imshow("Result2", cropped)
	    #cv2.waitKey(0)
	    result = cv2.cvtColor(cropped,cv2.COLOR_GRAY2BGR)
	    if lines is not None:
                for line in  lines:
		    for rho,theta in line:
			#print("Teta, Rho",theta*180/np.pi,rho)
			if theta*180/np.pi<20 or theta*180/np.pi>160:
                        	a = np.cos(theta)
                        	b = np.sin(theta)
                        	x0 = a*rho
                        	y0 = b*rho
                        	x1 = int(x0 + 1000*(-b))
                        	y1 = int(y0 + 1000*(a))
                        	x2 = int(x0 - 1000*(-b))
                        	y2 = int(y0 - 1000*(a))
		                cv2.line(result,(x1,y1),(x2,y2),(0,0,255),2)
				GPIO.output(led_pin, GPIO.HIGH)
				print("Found!")
            else:
                GPIO.output(led_pin, GPIO.LOW)
            
            cv2.namedWindow("Result", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Result", np.concatenate((edges,imclose,result[:,:,1]),0))
            #cv2.namedWindow("Result2", cv2.WINDOW_AUTOSIZE)
            #cv2.imshow("Result2", result)
            #cv2.namedWindow("Result3", cv2.WINDOW_AUTOSIZE)
            #cv2.imshow("Result3", imclose)

            if cv2.waitKey(1) == 27:
                break
	    t3 = time.time()
	    print("Show Time = ",t3-t2)
            rawCapture.truncate(0)
            
        # end while

    cv2.destroyAllWindows()                     # remove windows from memory

    return

###################################################################################################
if __name__ == "__main__":
    main()

