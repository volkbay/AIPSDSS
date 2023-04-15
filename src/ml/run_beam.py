
import os
import numpy as np
import tensorflow as tf
from scipy.misc import imread
from scipy.misc import imsave
from Volkan.my_network_beam import *
import cv2
import time

#Adjustable Param.
feat_depth=256
dataset_no=2
dataset_lim1=1
dataset_test=1
dataset_lim2=50
#model_str = 'model_beam_i20000_640x200_b8_fd256_ds2_f'
height_filter=25
soft_filter=5

#Fixed Param.
t1 = time.time()
learning_rate = 0.001
save_step = 1
org_height = 480
image_width = 640
image_height = 200
batch_size = 1
restore_flag = 1
display_step = 1
training_iters = dataset_lim2-dataset_lim1+1
#fold=int(model_str[-1])

x = tf.placeholder(tf.float32, [batch_size, image_height, image_width, 1], "Input_Images")
y = tf.placeholder(tf.int32, [batch_size, image_height, image_width], "Input_Labels")

img_list = [284, 393, 248, 203, 341, 501, 445, 45, 300, 317, 287, 312, 290, 458, 264, 425, 420, 299, 25, 151, 21, 40, 272, 494, 444, 81, 376, 162, 379, 398, 436, 390, 254, 164, 486, 143, 402, 415, 1, 79, 219, 16, 170, 152, 369, 460, 311, 28, 461, 282, 308, 447, 206, 353, 42, 224, 227, 467, 316, 39, 332, 498, 139, 263, 53, 271, 270, 154, 178, 134, 339, 182, 421, 236, 50, 121, 90, 481, 249, 232, 238, 125, 413, 285, 305, 354, 242, 389, 140, 407, 430, 127, 22, 351, 172, 255, 342, 247, 74, 191, 368, 372, 441, 201, 233, 9, 41, 385, 209, 145, 442, 159, 149, 34, 320, 205, 11, 189, 211, 190, 202, 434, 296, 229, 197, 291, 138, 451, 27, 346, 87, 274, 135, 262, 117, 69, 166, 410, 297, 315, 424, 391, 23, 76, 243, 157, 474, 449, 283, 61, 505, 490, 225, 92, 237, 188, 204, 468, 73, 414, 17, 239, 186, 71, 463, 163, 329, 56, 269, 408, 480, 51, 437, 177, 507, 358, 120, 98, 230, 119, 360, 179, 289, 131, 352, 492, 392, 261, 452, 142, 46, 288, 293, 10, 165, 298, 314, 146, 404, 124, 109, 336, 388, 502, 483, 476, 44, 370, 381, 24, 273, 180, 489, 234, 20, 343, 160, 313, 338, 62, 364, 344, 484, 153, 59, 450, 118, 235, 111, 335, 349, 228, 367, 171, 506, 357, 428, 213, 147, 144, 446, 482, 65, 218, 103, 158, 26, 148, 310, 212, 241, 334, 30, 500, 355, 279, 422, 453, 137, 35, 401, 395, 80, 221, 323, 156, 115, 70, 387, 210, 48, 60, 64, 14, 199, 331, 470, 487, 72, 67, 19, 403, 54, 418, 464, 440, 252, 433, 129, 150, 365, 356, 220, 240, 396, 292, 130, 155, 185, 281, 340, 465, 361, 47, 499, 493, 94, 429, 55, 397, 169, 280, 91, 193, 244, 196, 110, 337, 15, 253, 108, 374, 106, 319, 448, 3, 63, 8, 377, 256, 52, 432, 435, 462, 278, 475, 375, 347, 99, 181, 275, 208, 503, 455, 394, 122, 246, 454, 363, 83, 479, 400, 12, 472, 496, 386, 491, 266, 373, 97, 457, 295, 75, 322, 107, 459, 488, 416, 216, 49, 419, 250, 187, 93, 330, 136, 306, 405, 217, 128, 68, 504, 57, 276, 192, 350, 214, 258, 105, 5, 77, 260, 471, 443, 286, 32, 277, 100, 439, 231, 141, 168, 173, 268, 438, 321, 303, 326, 259, 161, 380, 345, 267, 477, 327, 66, 37, 257, 43, 112, 333, 371, 222, 469, 183, 184, 318, 304, 412, 33, 88, 478, 36, 423, 325, 78, 6, 95, 7, 195, 466, 18, 38, 417, 497, 4, 324, 223, 132, 174, 58, 456, 406, 348, 96, 176, 167, 495, 328, 382, 366, 207, 378, 302, 133, 2, 301, 89, 359, 362, 473, 411, 215, 126, 82, 508, 427, 104, 198, 13, 85, 426, 245, 86, 307, 29, 101, 265, 399, 116, 309, 123, 114, 194, 226, 431, 294, 84, 102, 383]

def my_read_file(number):
    name = str(number)
    img_directory = '//data//Tez//Dataset' + str(dataset_no) + '//img'
    name_to_read = os.path.join(img_directory, 'img' + name + '.bmp')
    original_image = imread(name_to_read, mode='L')
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
        cropped = original_image[cY - 100:cY + 100, :]
    else:
        cY = image_height / 2
        cropped = np.zeros((image_height, image_width))
        print('No light source!')
    mask_directory = '//data//Tez//Dataset' + str(dataset_no) + '//mask'
    name_to_read = os.path.join(mask_directory, 'mask' + name + '.jpg')
    mask_image = imread(name_to_read)

    org_mask = np.zeros(mask_image.shape)
    indices = np.where(mask_image[:, :] >= 200)  # ET
    org_mask[indices[0], indices[1]] = 1
    new_mask = org_mask[cY - 100:cY + 100, :]
    return np.expand_dims(cropped, 3), new_mask, original_image, org_mask, cY


prediction = net_volkan(x, reuse=False, training=False, feat_depth=feat_depth)
pre_cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=prediction)

soft_result = tf.nn.softmax(logits=prediction)

cost = tf.reduce_mean(pre_cost)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

training_summary = tf.summary.scalar("training_accuracy", cost)
merged = tf.summary.merge_all()

# Initializing the variables
init = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()

saver = tf.train.Saver()
m_average_loss = 0
m_conf_mat = 0

for fold in range(0,10):
    test_list = img_list[fold * 50:(fold + 1) * 50]
    train_list = [foo for foo in img_list if foo not in test_list]
    model_str = 'model_beam_i5000_640x200_b2_fd'+str(feat_depth)+'_ds2_f'+ str(fold)
    if os.path.isdir("/data/Tez Result/"+model_str)==False:
        os.mkdir("/home/volkan/PycharmProjects/deneme"+model_str)
    with tf.Session() \
            as sess:

        if restore_flag:
            for i in os.listdir("./model/"):
                if model_str in i:
                    filename = os.path.splitext(i)
            saved_model = "./model/"+filename[0]
            saver.restore(sess,saved_model)
            print("Model restored.")
        else:
            print("Model not restored")
            sess.run(init)

        writer = tf.summary.FileWriter(".//logs", graph=tf.get_default_graph())
        sess.run(init_l)  # initialize local variables
        step = 1

        batch_x = np.zeros((batch_size, image_height, image_width, 1))
        batch_y = np.zeros((batch_size, image_height, image_width))
        soft_prediction = np.zeros((batch_size, image_height, image_width, 2))
        t2 = time.time()
        print("Report for running "+model_str)
        print("Prepare Time = ", t2-t1)
        average_loss = 0
        average_loop_time = 0
        conf_mat = np.array([0 ,0 ,0, 0])
        while step < training_iters+1:
            t3 = time.time()
            for i in range(batch_size):
                random_file_num = '{:04d}'.format(test_list[step-1])
                input_images, mask, org_img, org_mask, center_y = my_read_file(random_file_num)
                batch_x[i,:,:,:] = input_images
                batch_y[i,:,:] = mask

            loss, numpy_prediction, loss_summary, soft_prediction = sess.run(
                [cost, prediction, merged,
                 soft_result], feed_dict={x: batch_x,
                                          y: batch_y})
            if step % display_step == 0:
                writer.add_summary(loss_summary, step)

                #print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss))




            if step % save_step == 0:
                org_img=cv2.cvtColor(org_img,cv2.COLOR_GRAY2RGB)
                #imsave("result/" + str(step) + "_" +"input.jpg", org_img)
                #imsave("result/" + str(step) + "_" +"mask.jpg", org_mask)

                for idx, im in enumerate(soft_prediction):
                    imsave("/data/Tez Result/"+model_str+"/soft" +str(random_file_num) +".jpg", np.uint8(im[:, :, 1]*255))
                    result_mask = np.zeros((org_height,image_width))
                    detect_img = cv2.inRange(np.uint8(im[:, :, 1]*255), soft_filter, 255)
                    detect_img = cv2.dilate(detect_img, None, iterations=1)
                    cnts = cv2.findContours(detect_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnts = cnts[1]
                    if len(cnts) != 0:
                         for index in range(len(cnts)):
                             r1,r2,r3,r4 = cv2.boundingRect(cnts[index])
                             if r4>height_filter:
                                 result_mask[r2+center_y-100:r2+r4+center_y-100,r1:r1+r3]=1
                                 cv2.rectangle(org_img, (r1, r2+center_y-100), (r1+r3, r2+r4+center_y-100), (0, 0, 255), 2)
                    imsave("/data/Tez Result/"+model_str+"/result" +str(random_file_num) +".jpg", org_img)


            t4 = time.time()
            average_loss = average_loss + loss
            average_loop_time = average_loop_time + (t4 - t3)
            if (step+dataset_lim1-1)==dataset_test:
                print("Training set loss= ", average_loss/(dataset_test-dataset_lim1+1))
                average_loss_train = average_loss

            #TN,TP,FN,FP
            TP = np.sum(np.logical_and(org_mask,result_mask))
            TN = np.sum(np.logical_and(1-org_mask,1-result_mask))
            FP = np.sum(np.logical_and(1-org_mask,result_mask))
            FN = np.sum(np.logical_and(org_mask,1-result_mask))
            conf_mat = conf_mat + (np.array([TP,TN ,FP ,FN])/(dataset_lim2-dataset_lim1+1))
            step += 1
            #print("Loop Time = ", t4 - t3)
    #print("Total Time = ", t4 - t1)
    print("Test set loss= ", (average_loss-average_loss_train)/(dataset_lim2-dataset_test))
    print("Average Loss = ", average_loss/(dataset_lim2-dataset_lim1+1))
    print("Average Loop Time = ", average_loop_time/(dataset_lim2-dataset_lim1+1))
    print("Average CONF MAT =", conf_mat)
    print("Accuracy =",(conf_mat[0]+conf_mat[1])/np.sum(conf_mat))
    print("Sensitivity =",conf_mat[0]/(conf_mat[0]+conf_mat[3]))
    print("Specitivity =",conf_mat[1]/(conf_mat[1]+conf_mat[2]))
    m_average_loss = m_average_loss + average_loss/10
    m_conf_mat = m_conf_mat + conf_mat/10
print("Main Average Loss = ", m_average_loss/50) # (dataset_lim2-dataset_lim1+1) = 50
print("Main Accuracy =",(m_conf_mat[0]+m_conf_mat[1])/np.sum(m_conf_mat))
print("Main Sensitivity =",m_conf_mat[0]/(m_conf_mat[0]+m_conf_mat[3]))
print("Main Specitivity =",m_conf_mat[1]/(m_conf_mat[1]+m_conf_mat[2]))
