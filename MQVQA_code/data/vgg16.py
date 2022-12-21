import numpy as np
import tensorflow as tf
import time

VGG_MEAN = [103.939, 116.779, 123.68]


class vgg_16():
    def __init__(self, path, image_height, image_width):
        self.model_path = path
        self.image_height = image_height
        self.image_width = image_width

        self.data_dict = np.load(self.model_path, encoding='latin1').item()
        print("vgg16.npy file loaded")

    def build(self, rgb):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build vgg16 model started")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [self.image_height, self.image_width, 1]
        assert green.get_shape().as_list()[1:] == [self.image_height, self.image_width, 1]
        assert blue.get_shape().as_list()[1:] == [self.image_height, self.image_width, 1]
        bgr = tf.concat(axis=3, values=[blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2]])
        assert bgr.get_shape().as_list()[1:] == [self.image_height, self.image_width, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")              # 224*224*64
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")     # 224*224*64
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')           # 112*112*64

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")       # 112*112*128
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")     # 112*112*128
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')           # 56*56*128

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")       # 56*56*256
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")     # 56*56*256
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")     # 56*56*256
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')            # 28*28*256

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")       # 28*28*512
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")     # 28*28*512
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")     # 28*28*512
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')           # 14*14*512

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")       # 14*14*512
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")     # 14*14*512
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")     # 14*14*512
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')           # 7*7*512

        print(("build model finished: %ds" % (time.time() - start_time)))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")


