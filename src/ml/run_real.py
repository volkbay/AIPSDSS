
import os
from random import randrange

import numpy as np
import tensorflow as tf
from scipy.misc import imread
from scipy.misc import imsave
from my_network_beam import *
import cv2
import time

t1 = time.time()

#Adjustable Param.
feat_depth=256
gauss_size=(7,7)
height_filter=50
soft_filter=5
model_str = 'model_beam_i20000_640x200_b8_fd256_ds2'

#Fixed Param.
org_height = 480
image_width = 640
image_height = 200
gst_str = ('nvcamerasrc ! '
               'video/x-raw(memory:NVMM), '
               'width=(int)2592, height=(int)1944, '
               'format=(string)I420, framerate=(fraction)30/1 ! '
               'nvvidconv ! '
               'video/x-raw, width={}, height={},'
               'format=(string)BGRx ! '
               'videoconvert ! video/x-raw, format=(string)BGR ! appsink').format(image_width, org_height)
camera = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

x = tf.placeholder(tf.float32, [1, image_height, image_width, 1], "Input_Images")
y = tf.placeholder(tf.int32, [1, image_height, image_width], "Input_Labels")

prediction = net_volkan(x, reuse=False, training=False, feat_depth=feat_depth)
pre_cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=prediction)

soft_result = tf.nn.softmax(logits=prediction)

cost = tf.reduce_mean(pre_cost)

training_summary = tf.summary.scalar("training_accuracy", cost)
merged = tf.summary.merge_all()

# Initializing the variables
init = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()

saver = tf.train.Saver()

with tf.Session() \
        as sess:
    saved_model = "./model/"+model_str
    saver.restore(sess,saved_model)
    print("Model restored.")

    writer = tf.summary.FileWriter(".//logs", graph=tf.get_default_graph())
    sess.run(init_l)  # initialize local variables
    step = 1

    batch_x = np.zeros((1, image_height, image_width, 1))
    batch_y = np.zeros((1, image_height, image_width))
    soft_prediction = np.zeros((1, image_height, image_width, 2))
    t2 = time.time()
    print("Prepare Time = ", t2-t1)
    average_loop_time = 0
    while True:
        t3 = time.time()
	
	_, original_image = camera.read()	
	original_image_blur = cv2.GaussianBlur(original_image,gauss_size,0)
    	gray_image = cv2.cvtColor(original_image_blur,cv2.COLOR_BGR2GRAY)
    	mask = cv2.inRange(gray_image, 215, 255)
    	mask = cv2.dilate(mask, None, iterations=1)
    	cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    	cnts = cnts[1]
    	if len(cnts) != 0:
		areas = [cv2.contourArea(c) for c in cnts]
		max_index = np.argmax(areas)
		large = cnts[max_index]
		M = cv2.moments(large)
		center_y = int(M["m01"] / M["m00"])
		if 0<=center_y<200:
			center_y = 200
		elif (org_height-200)<center_y<=org_height:
			center_y = org_height-200
		cropped = gray_image[center_y - 100:center_y + 100, :]
	else:
		center_y = org_height / 2
		cropped = np.zeros((image_height, image_width))
		print('No light source!')
	input_images= np.expand_dims(cropped, 3)
        batch_x[0,:,:,:] = input_images
        batch_y[0,:,:] = input_images[:,:,0]

        loss, numpy_prediction, loss_summary, soft_prediction = sess.run(
            [cost, prediction, merged,
             soft_result], feed_dict={x: batch_x,
                                      y: batch_y})
    	writer.add_summary(loss_summary, step)

    	print("Iter " + str(step))
    	result_image = original_image
	gray_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
	im = soft_prediction[0,:,:,:]
	detect_img = cv2.inRange(np.uint8(im[:, :, 1]*255), soft_filter, 255)
	detect_img = cv2.dilate(detect_img, None, iterations=1)
	cnts = cv2.findContours(detect_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[1]
	if len(cnts) != 0:
     		for index in range(len(cnts)):
         		r1,r2,r3,r4 = cv2.boundingRect(cnts[index])
         		if r4>height_filter:
            			cv2.rectangle(result_image, (r1, r2+center_y-100), (r1+r3, r2+r4+center_y-100), (0, 255, 0), 2)
	cv2.imshow("Result",result_image)
	button = cv2.waitKey(1)
	if button == 27:
		break
	elif button == 32:
		imsave("result/soft_"+model_str+"_"+str(step)+".jpg", np.uint8(im[:, :, 1]*255))
		imsave("result/input_"+model_str+"_"+str(step)+".jpg", gray_image)
		imsave("result/result_"+model_str+"_"+str(step)+".jpg", cv2.cvtColor(result_image,cv2.COLOR_BGR2RGB))
		
        step += 1
        t4 = time.time()
        print("Loop Time = ", t4 - t3)
        average_loop_time = average_loop_time + (t4 - t3)
#print("Total Time = ", t4 - t1)
print("Average Loop Time = ", average_loop_time/step)
