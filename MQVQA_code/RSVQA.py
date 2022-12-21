import tensorflow as tf

import os
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v2
import numpy as np
from tqdm import tqdm
import time
from skimage import filters, exposure, io, transform

class mqvqa:
    def __init__(self, config, sess):
        self.config = config
        self.sess = sess

    def build_model(self):
        print('\nBuilding Model')
        self.image_input = tf.placeholder(tf.float32, [self.config.batch_size, self.config.image_height,
                                                       self.config.image_width, self.config.image_dim])
        self.ques_input = tf.placeholder(tf.float32, [self.config.batch_size, self.config.max_word_length,
                                                      self.config.word_embedding_dim])
        self.ans = tf.placeholder(tf.int32, [self.config.batch_size, ])

        # extract the feature from image and text
        ques_vector = self._question_module(self.ques_input, reuse=False)
        image_vector = self._image_extract_resnet(self.image_input)
        # print(image_vector.shape, ques_vector.shape)
        # compute the attention
        fused_feature = self._fuse_concat(image_vector, ques_vector)

        # attention interaction
        temp1, ans_predict = self._ans_predict(fused_feature)
        # print(ans_predict.shape, self.ans.shape)
        # compute loss
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ans, logits=temp1)
        loss = tf.reduce_mean(cross_entropy)

        return loss, ans_predict

    def train(self, data):
        if not os.path.exists(self.config.checkpoint_path):
            os.makedirs(self.config.checkpoint_path)
        if not os.path.exists(self.config.loss_npy_path):
            os.makedirs(self.config.loss_npy_path)

        loss, predicts = self.build_model()
        print("Model Loaded")

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

        train_op = tf.train.AdamOptimizer(self.config.init_lr).minimize(loss)
        # train_ob = tf.train.AdamOptimizer(self.init_lr).minimize(answer_loss)
        loss_record = []

        self.sess.run(tf.global_variables_initializer())
        all_data = data.read_data_from_csv()
        batches = int(len(all_data) * 0.7 / self.config.batch_size)

        for epoch in tqdm(range(start_epoch, self.config.epoches)):
            start_time = time.time()
            for batch in tqdm(range(batches)):
                # 准备数据
                a_batch_data = all_data[batch * self.config.batch_size: (batch + 1) * self.config.batch_size]
                image_set, question_set, answer_set, label_set = a_batch_data['image_path'].tolist(), a_batch_data[
                    'question'].tolist(), a_batch_data['answer'].tolist(), a_batch_data['ques_label'].tolist()
                read_images, read_questions, read_answers, read_labels = data.read_a_batch_data(image_set, question_set,
                                                                                                answer_set, label_set,
                                                                                                onehot=False)

                _, result_los, result_ans = self.sess.run([train_op, loss, predicts], feed_dict={
                    self.image_input: read_images,
                    self.ques_input: read_questions,
                    self.ans: read_answers})

                loss_record.append(result_los)

                if batch % 50 == 0:
                    print("the epoch is %d" % epoch, "the batch is %d" % batch, "and the loss is %f" % result_los)
                    print("true ans is:", read_answers, "predict ans is :", np.argmax(result_ans, 1))

                    # io.imshow(lab_img_red[0, :, :, 0])
                    # io.show()
                    # self.save(self.checkpoint_path, counter)
                    # counter += 1
            print("this %d -th epoch cost time: %4.4f" % (epoch, time.time() - start_time))
            np.save(os.path.join(self.config.loss_npy_path, str(epoch) + '.npy'), np.array(loss_record))
            self.save(self.config.checkpoint_path, epoch)

    def validation(self, data):
        losses, predicts = self.build_model()
        print("Model Loaded")

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
                                                                                            answer_set, label_set)

            _, answer_get = self.sess.run([losses, predicts], feed_dict={self.image_input: read_images,
                                                                    self.ques_input: read_questions,
                                                                    self.ans: read_answers})
            # print(predicts.shape, answer_get.shape)
            # print(len(answer_get))
            top_ans = np.argmax(answer_get, 1)
            true_ans_num = read_answers

            print(top_ans, true_ans_num)
            # 找出答案对应的ans
            for i in range(self.config.batch_size):
                top_ans[i] += 1 if top_ans[i] == 0 else 0
                pred_ans = id2ans[top_ans[i]]
                true_ans = id2ans[true_ans_num[i]]
                result.append([pred_ans, true_ans])

                all_count += 1
                if true_ans_num[i] == top_ans[i]:
                    true_count += 1

        print("accuracy is %f" % (true_count / all_count))
        print(true_count, all_count)
        print("time cost is:", round(time.time() - start_time, 2), "s")

        if not os.path.exists(self.config.eval_result):
            os.makedirs(self.config.eval_result)
        np.save(os.path.join(self.config.eval_result, "eval_result.npy"), result)

    def test(self, pretrain_file, test_image_pathA, test_image_pathB, test_question):

        ################################### build model ###############################################################
        print('\nBuilding Test Model')
        self.image_A = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.image_dim])
        self.image_B = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.image_dim])
        self.quest = tf.placeholder(tf.float32, [1, self.max_word_length, self.word_embedding_dim])

        # 变化检测模块
        change_mask = self._change_module(self.image_A, self.image_B)
        image_vector = self._image_module(change_mask, self.image_A, self.image_B)  # 1 1 1 2048

        # 文本特征处理模块
        quest_vector = self._question_module(self.quest, reuse=False)  # 1 2048

        # 答案预测模块
        antention, answers = self._fuse_image_text(image_vector, quest_vector)

        ########################################### build finished ####################################################

        # 加载训练好的模型
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, os.path.join(self.checkpoint_path, pretrain_file))

        # 将图像和问题进行编码，输入到网络，然后查看输出即可
        bef_img_red = transform.resize(io.imread(test_image_pathA), [224, 224, 3])
        bef_img_red = np.expand_dims(bef_img_red, axis=0)
        aft_img_red = transform.resize(io.imread(test_image_pathB), [224, 224, 3])
        aft_img_red = np.expand_dims(aft_img_red, axis=0)

        ques = np.zeros([1, self.max_word_length, self.word_embedding_dim])
        # ques_red = test_question.split('?')[0]
        ques_vec = np.load(os.path.join(self.look_up_table_path, 'ques_used.npy')).item()[test_question]
        ques[0, :, :] = ques_vec

        attn, ans_pred = self.sess.run([antention, answers], feed_dict={self.image_A: bef_img_red,
                                                                        self.image_B: aft_img_red,
                                                                        self.quest: ques})

        # output and save
        top_ans = np.argmax(ans_pred)
        print(attn.shape)
        print(top_ans)

    # module -----------------------------------------------------------------------------------------------------------
    def _image_extract_resnet(self, image):
        # 图像特征处理模块
        with slim.arg_scope(self.resnet_arg_scope(is_training=True)):
            net, _ = resnet_v2.resnet_v2_152(image, reuse=tf.AUTO_REUSE)

        net = tf.reshape(net, [self.config.batch_size, net.shape[-1]])
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

    def _fuse_concat(self, image_tensor, text_tensor):
        features = tf.nn.dropout(tf.concat([image_tensor, text_tensor], axis=-1),  # [n, 1, 1, 4096]
                                 keep_prob=self.config.dropout_prob)
        return features

    def _ans_predict(self, fused_feature, reuse=tf.AUTO_REUSE):
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


