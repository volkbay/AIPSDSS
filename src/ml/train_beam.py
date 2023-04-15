
import os
from random import randrange
import time
import numpy as np
import tensorflow as tf
from scipy.misc import imread
from scipy.misc import imsave
from Volkan.my_network_beam import *
import cv2


t1 = time.time()
#Adjustable Param.
feat_depth=64
training_iters = 5000
batch_size = 8
#fold=1 #0-9 / 10-fold cross validation

#Fixed Param.
learning_rate = 0.001
save_step = training_iters/5
org_height = 480
image_width = 640
image_height = 200
restore_flag = 0
display_step = 2
dataset_no=2

x = tf.placeholder(tf.float32, [batch_size, image_height, image_width, 1], "Input_Images")
y = tf.placeholder(tf.int32, [batch_size, image_height, image_width], "Input_Labels")

img_list = [284, 393, 248, 203, 341, 501, 445, 45, 300, 317, 287, 312, 290, 458, 264, 425, 420, 299, 25, 151, 21, 40, 272, 494, 444, 81, 376, 162, 379, 398, 436, 390, 254, 164, 486, 143, 402, 415, 1, 79, 219, 16, 170, 152, 369, 460, 311, 28, 461, 282, 308, 447, 206, 353, 42, 224, 227, 467, 316, 39, 332, 498, 139, 263, 53, 271, 270, 154, 178, 134, 339, 182, 421, 236, 50, 121, 90, 481, 249, 232, 238, 125, 413, 285, 305, 354, 242, 389, 140, 407, 430, 127, 22, 351, 172, 255, 342, 247, 74, 191, 368, 372, 441, 201, 233, 9, 41, 385, 209, 145, 442, 159, 149, 34, 320, 205, 11, 189, 211, 190, 202, 434, 296, 229, 197, 291, 138, 451, 27, 346, 87, 274, 135, 262, 117, 69, 166, 410, 297, 315, 424, 391, 23, 76, 243, 157, 474, 449, 283, 61, 505, 490, 225, 92, 237, 188, 204, 468, 73, 414, 17, 239, 186, 71, 463, 163, 329, 56, 269, 408, 480, 51, 437, 177, 507, 358, 120, 98, 230, 119, 360, 179, 289, 131, 352, 492, 392, 261, 452, 142, 46, 288, 293, 10, 165, 298, 314, 146, 404, 124, 109, 336, 388, 502, 483, 476, 44, 370, 381, 24, 273, 180, 489, 234, 20, 343, 160, 313, 338, 62, 364, 344, 484, 153, 59, 450, 118, 235, 111, 335, 349, 228, 367, 171, 506, 357, 428, 213, 147, 144, 446, 482, 65, 218, 103, 158, 26, 148, 310, 212, 241, 334, 30, 500, 355, 279, 422, 453, 137, 35, 401, 395, 80, 221, 323, 156, 115, 70, 387, 210, 48, 60, 64, 14, 199, 331, 470, 487, 72, 67, 19, 403, 54, 418, 464, 440, 252, 433, 129, 150, 365, 356, 220, 240, 396, 292, 130, 155, 185, 281, 340, 465, 361, 47, 499, 493, 94, 429, 55, 397, 169, 280, 91, 193, 244, 196, 110, 337, 15, 253, 108, 374, 106, 319, 448, 3, 63, 8, 377, 256, 52, 432, 435, 462, 278, 475, 375, 347, 99, 181, 275, 208, 503, 455, 394, 122, 246, 454, 363, 83, 479, 400, 12, 472, 496, 386, 491, 266, 373, 97, 457, 295, 75, 322, 107, 459, 488, 416, 216, 49, 419, 250, 187, 93, 330, 136, 306, 405, 217, 128, 68, 504, 57, 276, 192, 350, 214, 258, 105, 5, 77, 260, 471, 443, 286, 32, 277, 100, 439, 231, 141, 168, 173, 268, 438, 321, 303, 326, 259, 161, 380, 345, 267, 477, 327, 66, 37, 257, 43, 112, 333, 371, 222, 469, 183, 184, 318, 304, 412, 33, 88, 478, 36, 423, 325, 78, 6, 95, 7, 195, 466, 18, 38, 417, 497, 4, 324, 223, 132, 174, 58, 456, 406, 348, 96, 176, 167, 495, 328, 382, 366, 207, 378, 302, 133, 2, 301, 89, 359, 362, 473, 411, 215, 126, 82, 508, 427, 104, 198, 13, 85, 426, 245, 86, 307, 29, 101, 265, 399, 116, 309, 123, 114, 194, 226, 431, 294, 84, 102, 383]

def my_read_file(number):
    name = str(number)
    img_directory = '//data//Tez//Dataset'+str(dataset_no)+'//img'
    name_to_read = os.path.join(img_directory, 'img' + name + '.bmp')
    original_image = imread(name_to_read,mode='L')
    mask = cv2.inRange(original_image, 215, 255)
    mask = cv2.dilate(mask, None, iterations=1)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1]
    if len(cnts) != 0:
        areas = [cv2.contourArea(c) for c in cnts]
        max_index = np.argmax(areas)
        large = cnts[max_index]
        M = cv2.moments(large)
        cY = int(M["m01"] / M["m00"])
        cropped = original_image[cY-100:cY+100, :]
    else:
        cY = image_height/2
        cropped = np.zeros((image_height, image_width))
        print('No light source!')
    mask_directory = '//data//Tez//Dataset'+str(dataset_no)+'//mask'
    name_to_read = os.path.join(mask_directory, 'mask' + name + '.jpg')
    mask_image = imread(name_to_read)

    new_mask = np.zeros(mask_image.shape)
    indices = np.where(mask_image[:, :] >= 200) # ET
    new_mask[indices[0],indices[1]] = 1
    new_mask = new_mask[cY-100:cY+100, :]
    return np.expand_dims(cropped,3), new_mask


prediction = net_volkan(x, reuse=False, training=True, feat_depth=feat_depth)
pre_cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=prediction)

soft_result = tf.nn.softmax(logits=prediction)

cost = tf.reduce_mean(pre_cost)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

training_summary = tf.summary.scalar("training_accuracy", cost)
merged = tf.summary.merge_all()

saver = tf.train.Saver()

# Initializing the variables
init = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()

for fold in range(0,5):
    test_list = img_list[fold * 50:(fold + 1) * 50]
    train_list = [foo for foo in img_list if foo not in test_list]
    with tf.Session() \
            as sess:
        sess.run(init)

        writer = tf.summary.FileWriter(".//logs", graph=tf.get_default_graph())
        sess.run(init_l)  # initialize local variables
        step = 1

        batch_x = np.zeros((batch_size, image_height, image_width, 1))
        batch_y = np.zeros((batch_size, image_height, image_width))
        soft_prediction = np.zeros((batch_size, image_height, image_width, 2))
        while step * batch_size < training_iters:

            for i in range(batch_size):
                dummy_no = randrange(0,450)
                random_file_num = '{:04d}'.format(train_list[dummy_no])
                input_images, mask = my_read_file(random_file_num)
                batch_x[i,:,:,:] = input_images
                batch_y[i,:,:] = mask

            loss, numpy_prediction, loss_summary, soft_prediction, _ = sess.run(
                [cost, prediction, merged,
                 soft_result, optimizer], feed_dict={x: batch_x,
                                          y: batch_y})
            if step % display_step == 0:
                # below lines added on 20170910
                writer.add_summary(loss_summary, step)

                print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss))


            step += 1

            # if step % save_step == 0:
            #     save_path = saver.save(sess, './data/my_model_step' + str(step))
            #     print("Model saved in file: %s" % save_path)

        print("Total time = ",time.time()-t1)
        for idx, im in enumerate(batch_y):
            for c in range(1):
                imsave("test/" + str(idx) + "_"  + "mask.jpg", im[:, :])

        for idx, im in enumerate(soft_prediction):
            for c in range(1):
                imsave("test/" + str(idx) + "_"  + "result.jpg", np.uint8(im[:, :, 1]*255))

        for idx, im in enumerate(batch_x):
            for c in range(1):
                imsave("test/" + str(idx) + "_"  + "input.jpg", im[:, :, 0])


        save_path = saver.save(sess, './data/model_beam'+'_i'+str(training_iters)+'_'+str(image_width)+'x'+str(image_height)+'_b'+str(batch_size)+'_fd'+str(feat_depth)+'_ds'+str(dataset_no)+'_f'+str(fold))
        print("Model saved in file: %s" % save_path)
        print("Report for training " + save_path)
