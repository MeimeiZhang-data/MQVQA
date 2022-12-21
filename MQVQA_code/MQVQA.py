import tensorflow as tf

import os
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v2
import numpy as np
from tqdm import tqdm
import time
from skimage import filters, exposure, io, transform
import matplotlib.pyplot as plt

from data.vgg16 import vgg_16


class mqvqa:
    def __init__(self, config, sess):
        self.config = config
        self.sess = sess

    def build_model_label(self):
        print('\nBuilding Model')
        self.image_input = tf.placeholder(tf.float32, [self.config.batch_size, self.config.image_height,
                                                       self.config.image_width, self.config.image_dim])
        self.ques_input = tf.placeholder(tf.float32, [self.config.batch_size, self.config.max_word_length,
                                                      self.config.word_embedding_dim])
        self.ans = tf.placeholder(tf.int32, [self.config.batch_size, self.config.num_class])
        self.label = tf.placeholder(tf.int32, [self.config.batch_size, ])

        # 文本特征处理模块
        ques_vector = self._question_module(self.ques_input)                                     # 1 2048
        temp1, ques_label = self._question_label(ques_vector)
        # 图像特征提取模块
        image_vector = self._image_extract_vgg_label(self.image_input, ques_label, driven=True)                         # 1 1 1 2048
        # image_vector = tf.zeros_like(image_vector)
        # 注意力融合模块
        weight1, att1 = self._fuse_attention_san(image_vector, ques_vector, name='attn_1', reuse=False)
        weight2, att2 = self._fuse_attention_san(image_vector, att1, name='attn_2', reuse=False)

        # 答案预测模块
        temp2, answers = self._ans_predict(att2)

        # 计算loss
        # loss_la = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.ans, logits=answers)
        # loss_lq = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.label, logits=temp1)
        loss_la = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.ans, logits=temp2)
        loss_lq = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=temp1)
        answer_loss = loss_lq + self.config.alpha * loss_la
        loss = tf.reduce_mean(answer_loss)

        weight = [weight1, weight2]
        return answers, loss, loss_la, loss_lq, weight

    def train(self, data):
        if not os.path.exists(self.config.checkpoint_path):
            os.makedirs(self.config.checkpoint_path)
        if not os.path.exists(self.config.loss_npy_path):
            os.makedirs(self.config.loss_npy_path)

        answers, losses, loss_la, loss_lq, weight = self.build_model_label()
        self.saver = tf.train.Saver()

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.config.checkpoint_path)
        if could_load:
            start_epoch = int(checkpoint_counter / self.config.batch_size)
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            counter = 1
            print(" [!] Load failed...")

        train_op = tf.train.AdamOptimizer(self.config.init_lr).minimize(losses)
        # train_ob = tf.train.AdamOptimizer(self.init_lr).minimize(answer_loss)
        loss_record = []
        loss_rec_a, loss_rec_q = [], []

        self.sess.run(tf.global_variables_initializer())
        all_data = data.read_data_from_csv()
        batches = int(len(all_data) * 0.99 / self.config.batch_size)

        for epoch in tqdm(range(start_epoch, self.config.epoches)):
            start_time = time.time()
            for batch in tqdm(range(batches)):
                # 准备数据
                a_batch_data = all_data[batch * self.config.batch_size: (batch + 1) * self.config.batch_size]
                image_set, question_set, answer_set, label_set = a_batch_data['image_path'].tolist(), a_batch_data[
                    'question'].tolist(), a_batch_data['answer'].tolist(), a_batch_data['ques_label'].tolist()
                read_images, read_questions, read_answers, read_labels = data.read_a_batch_data(image_set, question_set,
                                                                                                answer_set, label_set,
                                                                                                onehot=True)

                _, result_ans, result_los, result_la, result_lq, __ = self.sess.run([train_op, answers, losses, loss_la, loss_lq, weight],
                                                                      feed_dict={
                                                                          self.image_input: read_images,
                                                                          self.ques_input: read_questions,
                                                                          self.ans: read_answers,
                                                                            self.label: read_labels})

                loss_record.append(result_los)
                loss_rec_a.append(result_la)
                loss_rec_q.append(result_lq)

                if batch % 50 == 0:
                    print("the epoch is %d" % epoch, "the batch is %d" % batch, "and the loss is %f" % result_los)
                    print("the loss in answers is %f" % result_la, "the loss in questions is %f" % result_lq)
                    print("true ans is:", np.argmax(read_answers, 1), "predict ans is :", np.argmax(result_ans, 1))
                    # print(result_att.shape)
                    #
                    # io.imshow(lab_img_red[0, :, :, 0])
                    # io.show()
                    # self.save(self.checkpoint_path, counter)
            print("this %d -th epoch cost time: %4.4f" % (epoch, time.time() - start_time))
            np.save(os.path.join(self.config.loss_npy_path, str(epoch) + '.npy'), np.array(loss_record))
            np.save(os.path.join(self.config.loss_npy_path, str(epoch) + '_lossa.npy'), np.array(loss_rec_a))
            np.save(os.path.join(self.config.loss_npy_path, str(epoch) + '_lossq.npy'), np.array(loss_rec_q))
            self.save(self.config.checkpoint_path, epoch)

    def validation(self, data):
        answers, losses, loss_la, loss_lq, weight = self.build_model_label()
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, os.path.join(self.config.checkpoint_path, self.config.pretrain_file))

        start_time = time.time()
        result = []

        ans2id = np.load(self.config.ans_id_path).item()
        id2ans = {k + 1: v for k, v in enumerate(ans2id)}

        true_count, all_count = 0, 0
        # prepare data
        all_data = data.read_data_from_csv()
        star_num = int(len(all_data) * 0.7 / self.config.batch_size)
        num_iters = int(len(all_data) * 0.3 / self.config.batch_size)

        for batch in tqdm(range(num_iters)):
            if (batch + 1) * self.config.batch_size + star_num >= len(all_data):
                batch -= 1
            # 拿出一个batch的数据
            a_batch_data = all_data[
                           batch * self.config.batch_size + star_num: (batch + 1) * self.config.batch_size + star_num]
            image_set, question_set, answer_set, label_set = a_batch_data['image_path'].tolist(), a_batch_data[
                'question'].tolist(), a_batch_data['answer'].tolist(), a_batch_data['ques_label'].tolist()
            read_images, read_questions, read_answers, read_labels = data.read_a_batch_data(image_set, question_set,
                                                                                            answer_set, label_set,
                                                                                            onehot=True)

            result_ans, _, __, ___, result_wei = self.sess.run([answers, losses, loss_la, loss_lq, weight],
                                          feed_dict={self.image_input: read_images,
                                                     self.ques_input: read_questions,
                                                     self.ans: read_answers,
                                                     self.label: read_labels})

            top_ans = np.argmax(result_ans, 1)
            true_ans_num = np.argmax(read_answers, 1)

            print(top_ans, true_ans_num)
            # 找出答案对应的ans
            for i in range(self.config.batch_size):
                top_ans[i] += 1 if top_ans[i] == 0 else 0
                pred_ans = id2ans[top_ans[i]]
                true_ans = id2ans[true_ans_num[i]]
                result.append([pred_ans, true_ans])
                #         # true_ques = id2ques[question[i]]
                #         true_id = ids[i]
                all_count += 1
                if true_ans_num[i] == top_ans[i]:
                    true_count += 1

        print("accuracy is %f" % (true_count / all_count))
        print(true_count, all_count)
        print("time cost is:", round(time.time() - start_time, 2), "s")

        if not os.path.exists(self.config.eval_result):
            os.makedirs(self.config.eval_result)
        np.save(os.path.join(self.config.eval_result, 'eval_result.npy'), np.array(result))

    def test(self, pretrain_file, test_image_path, test_question, data):

        ################################### build model ###############################################################
        print('\nBuilding Test Model')
        self.image = tf.placeholder(tf.float32, [1, self.config.image_height, self.config.image_width, self.config.image_dim])
        self.quest = tf.placeholder(tf.float32, [1, self.config.max_word_length, self.config.word_embedding_dim])

        # 文本特征处理模块
        ques_vector = self._question_module(self.quest)
        _, ques_label = self._question_label(ques_vector)
        # 图像特征处理模块
        image_vector = self._image_extract_vgg_label(self.image, ques_label, driven=True)
        # 注意力融合模块
        weight1, att1 = self._fuse_attention_san(image_vector, ques_vector, name='attn_1', reuse=False)
        weight2, att2 = self._fuse_attention_san(image_vector, att1, name='attn_2', reuse=False)
        # 答案预测模块
        _, answers = self._ans_predict(att2)

        ########################################### build finished ####################################################

        # 加载训练好的模型
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, os.path.join(self.config.checkpoint_path, pretrain_file))

        # 将图像和问题进行编码，输入到网络，然后查看输出即可
        temp1 = transform.resize(io.imread(test_image_path), [self.config.image_height, self.config.image_width,
                                                             self.config.image_dim])
        img = np.expand_dims(temp1, axis=0)

        temp2 = data.ques_used[test_question]
        ques_v = np.expand_dims(temp2, axis=0)

        attn1, attn2, ans_pred= self.sess.run([weight1, weight2, answers], feed_dict={self.image: img, self.quest:ques_v})

        # output and save

        ans2id = np.load(self.config.ans_id_path).item()
        id2ans = {k + 1: v for k, v in enumerate(ans2id)}

        top_ans = np.argmax(ans_pred)
        top_ans += 1 if top_ans == 0 else 0
        pred_ans = id2ans[top_ans]

        print(pred_ans)
        attn1 = transform.resize(np.reshape(attn1, (112, 112)), (224, 224))
        attn2 = transform.resize(np.reshape(attn2, (112, 112)), (224, 224))
        # 保存原图和注意力图
        if not os.path.exists(self.config.test_result):
            os.makedirs(self.config.test_result)
        plt.figure('atte_map_1')
        plt.imshow(attn1, cmap='jet')
        plt.imsave(os.path.join(self.config.test_result, 'attn1.png'), attn1, cmap='jet')
        plt.figure('atte_map_2')
        plt.imshow(attn2, cmap='jet')
        plt.imsave(os.path.join(self.config.test_result, 'attn2.png'), attn2, cmap='jet')
        # plt.figure('atte_map_3')
        # plt.imshow(transform.resize(attn3, (448, 448)), cmap='jet')
        # plt.imsave(os.path.join(self.config.test_result, 'attn2.png'), cmap='jet')
        # plt.figure('atte_map_4')
        # plt.imshow(transform.resize(attn4, (448, 448)), cmap='jet')
        # plt.imsave(os.path.join(self.config.test_result, 'attn2.png'), cmap='jet')
        plt.show()


    # module -----------------------------------------------------------------------------------------------------------
    # 用vgg和label提取
    def _image_extract_vgg_label(self, image, label, reuse=False, driven=True):
        with tf.variable_scope("image_extract", reuse=reuse):
            vgg = vgg_16(self.config.vgg16_weight_path, self.config.image_height, self.config.image_width)
            vgg.build(image)

            x = tf.image.resize_images(vgg.pool5, [vgg.conv2_2.shape[1], vgg.conv2_2.shape[2]])
            x = tf.reshape(x, [-1, x.shape[1] * x.shape[2], x.shape[3]])

            if driven:
                # x1 = tf.layers.flatten(vgg.conv1_2)                   # 224*224*64
                x1 = tf.image.resize_images(vgg.conv1_2, [vgg.conv2_2.shape[1], vgg.conv2_2.shape[2]])
                x1 = tf.reshape(x1, [-1, x1.shape[1] * x1.shape[2], x1.shape[3]])
                # x2 = tf.layers.flatten(vgg.conv3_3)                   # 56*56*256
                x2 = tf.image.resize_images(vgg.conv3_3, [vgg.conv2_2.shape[1], vgg.conv2_2.shape[2]])
                x2 = tf.reshape(x2, [-1, x2.shape[1] * x2.shape[2], x2.shape[3]])

                # x3 = tf.layers.flatten(vgg.conv5_3)                   # 14*14*512
                x3 = tf.image.resize_images(vgg.conv5_3, [vgg.conv2_2.shape[1], vgg.conv2_2.shape[2]])
                x3 = tf.reshape(x3, [-1, x3.shape[1] * x3.shape[2], x3.shape[3]])

                lable0 = tf.reshape(label[:, 0], [self.config.batch_size, 1, 1])
                lable1 = tf.reshape(label[:, 1], [self.config.batch_size, 1, 1])
                lable2 = tf.reshape(label[:, 2], [self.config.batch_size, 1, 1])
                x1 = tf.multiply(tf.tile(lable0, [1, x1.shape[1], x1.shape[2]]), x1)
                x2 = tf.multiply(tf.tile(lable1, [1, x2.shape[1], x2.shape[2]]), x2)
                x3 = tf.multiply(tf.tile(lable2, [1, x3.shape[1], x3.shape[2]]), x3)

                x = tf.concat([x, x1, x2, x3], axis=-1)
                # print(x.shape)

            x = tf.layers.dense(x, self.config.state_size)

        return x

    # 用resnet和label提取
    def _image_extract_resnet_label(self, image, label, driven=False):
        # 图像特征处理模块
        with slim.arg_scope(self.resnet_arg_scope(is_training=True)):
            net, end_points = resnet_v2.resnet_v2_152(image, reuse=tf.AUTO_REUSE)

            if driven:
                # net = tf.layers.flatten(net)                                                          # 1， 1， 2048
                scale_large = end_points['resnet_v2_152/conv1'] * label[:, 0]                           # 112，112，64
                scale_middle = end_points['resnet_v2_152/block1/unit_2/bottleneck_v2'] * label[:, 1]    # 56， 56， 256
                scale_small = end_points['resnet_v2_152/block3/unit_35/bottleneck_v2'] * label[:, 2]    # 14， 14， 1024

                # 拉伸为一维向量处理
                net = tf.layers.flatten(net)
                scale_large = tf.layers.flatten(scale_large)
                scale_middle = tf.layers.flatten(scale_middle)
                scale_small = tf.layers.flatten(scale_small)

                net = tf.concat([net, scale_large, scale_middle, scale_small], axis=-1)

        return net

    def _question_module(self, ques_tensor, reuse=False, LSTM_or_GRU='GRU'):
        """
        :param ques_tensor: [batch_size, length, dim]
        :return:
        """
        with tf.variable_scope("question_extract", reuse=reuse):
            def rnn_cell_lstm_gru(LSTM_or_GRU='GRU'):
                if LSTM_or_GRU == 'LSTM':
                    cell = tf.nn.rnn_cell.LSTMCell(num_units=self.config.state_size, activation=tf.nn.relu)
                else:
                    cell = tf.contrib.rnn.GRUCell(num_units=self.config.state_size, activation=tf.nn.relu)
                return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.config.dropout_prob)

            embedded_sentence = tf.nn.dropout(tf.nn.tanh(ques_tensor, name="embedded_sentence"),
                                              keep_prob=self.config.dropout_prob)

            layers = [rnn_cell_lstm_gru(LSTM_or_GRU) for _ in range(self.config.RNN_layers)]
            multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)

            _outputs, _final_state = tf.nn.dynamic_rnn(multi_layer_cell, embedded_sentence,
                                               dtype=tf.float32)
            #TODO 取最后一个时序输出作为结果，取output的最后一个时序或者直接取state，结果是一样的
            # last = _outputs[:, -1, :]
        return _outputs[:, -1, :]

    def _question_label(self, ques_vector):
        fc_1 = tf.nn.relu(self.fc_layer(ques_vector, 512, name="q_fc1"))
        fc_2 = tf.nn.relu(self.fc_layer(fc_1, 128, name="q_fc2"))
        fc_3 = self.fc_layer(fc_2, 3, name="q_fc3")
        label_result = tf.nn.softmax(fc_3)
        return fc_3, label_result

    def _fuse_concat(self, image_tensor, text_tensor):
        features = tf.nn.dropout(tf.concat([image_tensor, text_tensor], axis=-1),  # [n, 1, 1, 4096]
                                 keep_prob=self.config.dropout_prob)
        return features

    def _fuse_attention_san(self, image_tensor, question_tensor, name='', reuse=False):
        with tf.variable_scope("attention_" + name) as scope:
            if reuse:
                scope.reuse_variables()
            # print(image_tensor.shape, question_tensor.shape)
            ques_tensor = tf.tile(tf.expand_dims(question_tensor, axis=-2), [1, image_tensor.shape[-2], 1])
            img = tf.nn.tanh(image_tensor)            # [batch size, 224*224, out_dim]
            ques = tf.nn.tanh(ques_tensor)        # [batch size, 1, out_dim]

            IQ = tf.nn.dropout(tf.nn.tanh(img + ques), self.config.dropout_prob)
            temp = tf.layers.dense(IQ, 1)
            temp = tf.reshape(temp, [-1, temp.shape[1]])  # [batch size, 784*1]

            # softmax获得注意力
            p = tf.nn.softmax(temp)  # [batch size, 784]
            p_exp = tf.expand_dims(p, axis=-1)  # [batch size, 784, 1]
            att_layer = tf.reduce_sum(p_exp * image_tensor, axis=1)  # [batch size, 2048]

            # 最终的注意力结果
            final_out = att_layer + question_tensor  # [batch size, 2048]

            return p, final_out

    def _fuse_attention(self, image_tensor, text_tensor, name="attention", reuse=tf.AUTO_REUSE):
        '''注意力机制采用一维的处理方式'''
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            # 先将图像特征映射成和文本一样的大小
            # img_f = tf.nn.tanh(tf.layers.dense(image_tensor, self.config.state_size))
            que_f = tf.nn.tanh(text_tensor)
            img_f = tf.nn.tanh(image_tensor)
            # print(img_f.shape, que_f.shape)
            IQ_features = tf.nn.dropout(tf.nn.tanh(tf.concat([img_f, que_f], axis=-1)),  # [n, 1, 1, 4096]
                                     keep_prob=self.config.dropout_prob)

            attn_fc1 = tf.layers.dense(IQ_features, self.config.image_height * self.config.image_width / 4,
                                       name='attn_fc1', reuse=tf.AUTO_REUSE)
            attn_fc1r = tf.nn.relu(attn_fc1)
            attn_fc2 = tf.layers.dense(attn_fc1r, self.config.image_height * self.config.image_width / 8,
                                       name='attn_fc2', reuse=tf.AUTO_REUSE)

            attention_map = tf.reshape(attn_fc2, (self.config.batch_size, int(self.config.image_height * self.config.image_width / 8), 1))
            attention_map = tf.nn.softmax(attention_map, axis=1, name="attention_map")

            flatten_attn = tf.layers.flatten(attention_map)

        return attention_map, flatten_attn

    def _ans_predict(self,fused_feature, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("answer") as scope:
            if reuse:
                scope.reuse_variables()

            # softmax
            fc1 = tf.nn.dropout(tf.nn.relu(self.fc_layer(fused_feature, 1024, name="a_fc1")),
                                keep_prob=self.config.dropout_prob)
            fc2 = self.fc_layer(fc1, self.config.num_class, name="a_fc2")

            answer_prob = tf.nn.softmax(fc2)
            return fc2, answer_prob

    # layers -----------------------------------------------------------------------------------------------------------
    def conv2d_layer(self, input_tensor, filters, kernel_size=(3, 3), stride=1, name="conv", padding='SAME'):
        with tf.variable_scope(name):
            weights = tf.get_variable('conv_weights',
                                      [kernel_size[0], kernel_size[1], input_tensor.get_shape()[-1], filters],
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('conv_bias', [filters], initializer=tf.constant_initializer(0.0))

            # print(input_tensor.shape, weights.shape)
            conv = tf.nn.conv2d(input_tensor, weights, strides=[1, stride, stride, 1], padding=padding)
            conv = tf.nn.bias_add(conv, biases)
            return conv

    def deconv2d_layer(self, input_tensor, filters, output_size,
                       kernel_size=(5, 5), stride=2, name="deconv2d"):
        with tf.variable_scope(name):
            h, w = output_size
            weights = tf.get_variable('deconv_weights',
                                      shape=[kernel_size[0], kernel_size[1],
                                             filters, input_tensor.get_shape()[-1]],
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('biases', [filters], initializer=tf.constant_initializer(0.0))

            output_dims = [self.config.batch_size, h, w, filters]
            deconv = tf.nn.conv2d_transpose(input_tensor, weights, strides=[1, stride, stride, 1],
                                            output_shape=output_dims)
            deconv = tf.nn.bias_add(deconv, biases)
            return deconv

    def fc_layer(self, input_tensor, neurons, name="fc"):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            weights = tf.get_variable('fc_weights', [input_tensor.get_shape()[-1], neurons],
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('fc_biases', [neurons], initializer=tf.constant_initializer(0.0))

            output = tf.matmul(input_tensor, weights) + biases
            return output

    def resnet_arg_scope(self, is_training=True,  # 训练标记
                         weight_decay=0.0001,  # 权重衰减速率
                         batch_norm_decay=0.997,  # BN的衰减速率
                         batch_norm_epsilon=1e-5,  # BN的epsilon默认1e-5
                         batch_norm_scale=True):  # BN的scale默认值

        batch_norm_params = {  # 定义batch normalization（标准化）的参数字典
            'is_training': is_training,
            'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon,
            'scale': batch_norm_scale,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
        }

        with slim.arg_scope(  # 通过slim.arg_scope将[slim.conv2d]的几个默认参数设置好
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(weight_decay),  # 权重正则器设置为L2正则
                weights_initializer=slim.variance_scaling_initializer(),  # 权重初始化器
                activation_fn=tf.nn.relu,  # 激活函数
                normalizer_fn=slim.batch_norm,  # 标准化器设置为BN
                normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:  # ResNet原论文是VALID模式，SAME模式可让特征对齐更简单
                    return arg_sc

    # functions --------------------------------------------------------------------------------------------------------
    def load(self, chpt_dir):
        import re
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(chpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(chpt_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(self.config.checkpoint_path, "model.ckpt"), global_step=step)

