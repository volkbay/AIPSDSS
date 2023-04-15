
import os
from random import randrange
import time
import numpy as np
import tensorflow as tf
from scipy.misc import imread
from scipy.misc import imsave
from Volkan.my_network_neu import *

t1 = time.time()
#Adjustable Param.
feat_depth=256
training_iters = 20000
batch_size = 2

#Fixed Param.
learning_rate = 0.001
save_step = training_iters/5
image_width = 200
image_height = 200
restore_flag = 0
display_step = 2
dataset_no=4
dataset_lim1=1
dataset_lim2=200

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


prediction = net_volkan(x, reuse=False, training=True, feat_depth=feat_depth)
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
    saved_model = "./data/my_model"
    if os.path.isfile(saved_model+".meta") and restore_flag:
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
    while step * batch_size < training_iters:

        for i in range(batch_size):
            random_file_num = '{:04d}'.format(randrange(dataset_lim1,dataset_lim2))
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

    print("Total time = ", time.time() - t1)
    for idx, im in enumerate(batch_y):
        for c in range(1):
            imsave("test/" + str(idx) + "_"  + "mask.jpg", im[:, :])

    for idx, im in enumerate(soft_prediction):
        for c in range(1):
            imsave("test/" + str(idx) + "_"  + "result.jpg", np.uint8(im[:, :, 1]*255))

    for idx, im in enumerate(batch_x):
        for c in range(1):
            imsave("test/" + str(idx) + "_"  + "input.jpg", im[:, :, 0])


    save_path = saver.save(sess, './data/model_neu'+'_i'+str(training_iters)+'_'+str(image_width)+'x'+str(image_height)+'_b'+str(batch_size)+'_fd'+str(feat_depth)+'_ds'+str(dataset_no))
    print("Model saved in file: %s" % save_path)
    print("Report for training " + save_path)