
import os
import numpy as np
import tensorflow as tf
from scipy.misc import imread
from scipy.misc import imsave
from Volkan.my_network_neu import *
import cv2
import time

#Adjustable Param.
feat_depth=256
dataset_no=4
dataset_lim1=1
dataset_test=200
dataset_lim2=300
model_str = 'model_neu_i20000_200x200_b2_fd256_ds4'
height_filter=10
soft_filter=20
if os.path.isdir("/data/Tez Result/"+model_str)==False:
    os.mkdir("/data/Tez Result/"+model_str)

#Fixed Param.
t1 = time.time()
learning_rate = 0.001
save_step = 1
image_width = 200
image_height = 200
batch_size = 1
restore_flag = 1
display_step = 1
training_iters = dataset_lim2-dataset_lim1+1

x = tf.placeholder(tf.float32, [batch_size, image_height, image_width, 1], "Input_Images")
y = tf.placeholder(tf.int32, [batch_size, image_height, image_width], "Input_Labels")

def my_read_file(number):
    name = str(number)
    img_directory = '//data//Tez//Dataset'+str(dataset_no)+'//img'
    name_to_read = os.path.join(img_directory, 'img' + name + '.jpg')
    original_image = imread(name_to_read,mode='L')

    mask_directory = '//data//Tez//Dataset'+str(dataset_no)+'//mask'
    name_to_read = os.path.join(mask_directory, 'mask' + name + '.bmp')
    mask_image = imread(name_to_read)

    new_mask = np.zeros(mask_image.shape)
    indices = np.where(mask_image[:, :] >= 200) # ET
    new_mask[indices[0],indices[1]] = 1
    return np.expand_dims(original_image,3), new_mask


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
    print("Report for running " + model_str)
    print("Prepare Time = ", t2 - t1)
    average_loss = 0
    average_loop_time = 0
    conf_mat = np.array([0, 0, 0, 0])
    while step < training_iters:
        t3 = time.time()
        for i in range(batch_size):
            random_file_num = '{:04d}'.format(step+dataset_lim1-1)
            input_images, mask = my_read_file(random_file_num)
            batch_x[i,:,:,:] = input_images
            batch_y[i,:,:] = mask

        loss, numpy_prediction, loss_summary, soft_prediction = sess.run(
            [cost, prediction, merged,
             soft_result], feed_dict={x: batch_x,
                                      y: batch_y})
        if step % display_step == 0:
            writer.add_summary(loss_summary, step)

        if step % save_step == 0:
            org_img = cv2.cvtColor(input_images, cv2.COLOR_GRAY2RGB)
            # imsave("result/" + str(step) + "_" +"input.jpg", org_img)
            # imsave("result/" + str(step) + "_" +"mask.jpg", mask)

            for idx, im in enumerate(soft_prediction):
                imsave("/data/Tez Result/" + model_str + "/soft" + str(random_file_num) + ".jpg",
                       np.uint8(im[:, :, 1] * 255))
                result_mask = np.zeros((image_height, image_width))
                detect_img = cv2.inRange(np.uint8(im[:, :, 1] * 255), soft_filter, 255)
                cnts = cv2.findContours(detect_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[1]
                if len(cnts) != 0:
                    for index in range(len(cnts)):
                        r1, r2, r3, r4 = cv2.boundingRect(cnts[index])
                        if r4 > height_filter:
                            result_mask[r2 :r2 + r4 , r1:r1 + r3] = 1
                            cv2.rectangle(org_img, (r1, r2), (r1 + r3, r2 + r4), (0, 0, 255), 2)
                imsave("/data/Tez Result/" + model_str + "/result" + str(random_file_num) + ".jpg", org_img)

        t4 = time.time()
        average_loss = average_loss + loss
        average_loop_time = average_loop_time + (t4 - t3)
        if (step + dataset_lim1 - 1) == dataset_test:
            print("Training set loss= ", average_loss / (dataset_test - dataset_lim1 + 1))
            average_loss_train = average_loss

        # TN,TP,FN,FP
        TP = np.sum(np.logical_and(mask, result_mask))
        TN = np.sum(np.logical_and(1 - mask, 1 - result_mask))
        FP = np.sum(np.logical_and(1 - mask, result_mask))
        FN = np.sum(np.logical_and(mask, 1 - result_mask))
        conf_mat = conf_mat + (np.array([TP, TN, FP, FN]) / (dataset_lim2 - dataset_lim1 + 1))
        step += 1
        # print("Loop Time = ", t4 - t3)
    # print("Total Time = ", t4 - t1)
    print("Test set loss= ", (average_loss - average_loss_train) / (dataset_lim2 - dataset_test))
    print("Average Loss = ", average_loss / (dataset_lim2 - dataset_lim1 + 1))
    print("Average Loop Time = ", average_loop_time / (dataset_lim2 - dataset_lim1 + 1))
    print("Average CONF MAT =", conf_mat)
    print("Accuracy =", (conf_mat[0] + conf_mat[1]) / np.sum(conf_mat))
    print("Sensitivity =", conf_mat[0] / (conf_mat[0] + conf_mat[3]))
    print("Specitivity =", conf_mat[1] / (conf_mat[1] + conf_mat[2]))