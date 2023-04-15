from Volkan.opt import *

width = 200
height = 200
x = tf.placeholder("float", [None, height, width, 1])

def net_volkan(x, training=True, reuse=False, feat_depth=256):
    with tf.variable_scope("unet_volkan") as scope:
        if reuse == True:
            scope.reuse_variables()

    #Layer 1
    conv11 = conv2d(x, feat_depth/8, kernel=3, stride=1, name='l1_conv1') # 32 feature channels (filters : 3x3x1x32)
    conv11 = batch_norm(conv11, name='l1_bn1', training=training)
    conv11 = prelu(conv11, name='l1_prelu')

    conv11m = maxpool2d(conv11) # 32 feature channels

    #Layer 2
    conv21 = conv2d(conv11m, feat_depth/4, kernel=3, stride=1, name='l2_conv1') # (filters : 3x3x32x64)
    conv21 = batch_norm(conv21, name='l2_bn1', training=training)
    conv21 = prelu(conv21, name='l2_prelu')

    conv21m = maxpool2d(conv21)

    #Layer 3
    conv31 = conv2d(conv21m, feat_depth / 2, kernel=3, stride=1, name='l3_conv1') # (filters : 3x3x64x128)
    conv31 = batch_norm(conv31, name='l3_bn1', training=training)
    conv31 = prelu(conv31, name='l3_prelu')

    conv31m = maxpool2d(conv31)

    #Conv
    conv41 = conv2d(conv31m, feat_depth, kernel=3, stride=1, name='l4_conv') # (filters : 3x3x128x256)
    conv41 = batch_norm(conv41, name='l4_conv', training=training)
    conv41 = prelu(conv41, name='l4_prelu')

    upsampled1 = tf.image.resize_images(conv41, [height, width])  # 240x135   256 feature channels
    upsampled2 = tf.image.resize_images(conv21m, [height, width]) # 480x270   64 feature channels
    upsampled3 = tf.image.resize_images(conv11m, [height, width]) # 960x540   32 feature channels

    deconvc51 = tf.concat(values=[upsampled1, upsampled2, upsampled3], axis=3) # 1920x1080   352 feature channels

    result = conv2d(deconvc51, 2, kernel=1, stride=1, name='l5_conv') # 1920x1080   2 feature channels

    # dilated_result = dilation_modules(x, 4, name='dilationModule1')
    # dilated_result = conv2d(dilated_result,2, kernel=1, stride=1, name='dilation_1_1')
    # combined_result = tf.concat(values=[result, dilated_result], axis=3)
    # combined_result = conv2d(combined_result, 2, kernel=1, stride=1, name='l6_conv')
    return result