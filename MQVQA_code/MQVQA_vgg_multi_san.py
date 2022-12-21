import tensorflow as tf

import os
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v2
import numpy as np
from tqdm import tqdm
import time
from skimage import filters, exposure, io, transform

from data.vgg16 import vgg_16

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
        image_vector = self._image_extract_vgg_multi(self.image_input, reuse=False)
        # print("shape from image and ques", image_vector.shape, ques_vector.shape)
        # compute the attention
        weight1, att1 = self._fuse_attention_san(image_vector, ques_vector, name='attn_1', reuse=False)
        weight2, att2 = self._fuse_attention_san(att1, ques_vector, name='attn_2', reuse=False)
        # att_l3, att = self._fuse_attention_san(att, ques_vector, name='attn_3', reuse=False)
        # att_l4, att = self._fuse_attention_san(att, ques_vector, name='attn_4', reuse=False)

        # attention interaction
        # att = tf.nn.dropout(tf.layers.dense(att, 512), self.config.dropout_prob)
        # att = tf.nn.relu(att, name='ans_relu')
        # print(att2.shape, weight1.shape)
        tempt1 = tf.nn.dropout(att2, self.config.dropout_prob)
        att = tf.layers.dense(tempt1, self.config.num_class)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ans, logits=att)
        loss = tf.reduce_mean(loss)
        predict_ans = tf.nn.softmax(att)
        # att = tf.layers.dense(att, self.config.num_class)                            # [batch size, ans_num]
        #
        # predict_ans = tf.nn.softmax(att)                                    # [batch size, ans_num]

        # compute loss
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ans, logits=ans_predict)
        # loss = tf.reduce_mean(cross_entropy)

        attention_layers = [weight1, weight2]
        return loss, attention_layers, predict_ans

    def train(self, data):
        if not os.path.exists(self.config.checkpoint_path):
            os.makedirs(self.config.checkpoint_path)
        if not os.path.exists(self.config.loss_npy_path):
            os.makedirs(self.config.loss_npy_path)

        losses, attention, predicts = self.build_model()
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

        self.sess.run(tf.global_variables_initializer())
        all_data = data.read_data_from_csv()
        batches = int(len(all_data) * 0.7 / self.config.batch_size)

        for epoch in tqdm(range(start_epoch, self.config.epoches)):
            start_time = time.time()
            for batch in tqdm(range(batches)):
                # 准备数据
                a_batch_data = all_data[batch * self.config.batch_size: (batch + 1) * self.config.batch_size]
                image_set, question_set, answer_set, label_set = a_batch_data['image_path'].tolist(), a_batch_data['question'].tolist(), \
                                                                 a_batch_data['answer'].tolist(), a_batch_data['ques_label'].tolist()
                read_images, read_questions, read_answers, read_labels = data.read_a_batch_data(image_set, question_set,
                                                                                                answer_set, label_set)

                _, result_los, result_att, result_ans = self.sess.run([train_op, losses, attention, predicts], feed_dict={
                    self.image_input: read_images,
                    self.ques_input: read_questions,
                    self.ans: read_answers})

                loss_record.append(result_los)

                if batch % 10 == 0:
                    print("the epoch is %d" % epoch, "the batch is %d" % batch, "and the loss is %f" % result_los)

                    print("true ans is:", read_answers, "predict ans is :", np.argmax(result_ans, 1))
                    # print(result_att.shape)
                    #
                    # io.imshow(lab_img_red[0, :, :, 0])
                    # io.show()
                    # self.save(self.checkpoint_path, counter)
                    # counter += 1
            print("this %d -th epoch cost time: %4.4f" % (epoch, time.time() - start_time))
            np.save(os.path.join(self.config.loss_npy_path, str(epoch) + '.npy'), np.array(loss_record))
            self.save(self.config.checkpoint_path, epoch)

    def validation(self, data):
        loss, attention_layers, predicts = self.build_model()
        print("Model Loaded")

        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, os.path.join(self.config.checkpoint_path, self.config.pretrain_file))

        start_time = time.time()
        result = []

        ans2id = np.load(self.config.ans_id_path).item()
        id2ans = {k+1: v for k, v in enumerate(ans2id)}

        true_count, all_count = 0, 0
        # prepare data
        all_data = data.read_data_from_csv()
        star_num = int(len(all_data) * 0.7 / self.config.batch_size)
        num_iters = int(len(all_data) * 0.3 / self.config.batch_size)
        for batch in tqdm(range(num_iters)):
            # TODO 对最后一个batch的处理
            if (batch + 1) * self.config.batch_size + star_num >= len(all_data):
                batch -= 1
            # 拿出一个batch的数据
            a_batch_data = all_data[batch * self.config.batch_size + star_num: (batch + 1) * self.config.batch_size + star_num]
            image_set, question_set, answer_set, label_set = a_batch_data['image_path'].tolist(), a_batch_data[
                'question'].tolist(), a_batch_data['answer'].tolist(), a_batch_data['ques_label'].tolist()
            read_images, read_questions, read_answers, read_labels = data.read_a_batch_data(image_set, question_set,
                                                                                            answer_set, label_set)

            _, answer_get = self.sess.run([loss, predicts], feed_dict={self.image_input: read_images,
                                                                         self.ques_input: read_questions,
                                                                         self.ans: read_answers})
            # print(predicts.shape, answer_get.shape)
            # print(len(answer_get))
            top_ans = np.argmax(answer_get, 1)

            # if top_ans == 0 or top_ans >= self.config.num_class:
            #     top_ans_new = 1
            # else:
            #     top_ans_new = top_ans
            true_ans_num = read_answers

            print(top_ans, true_ans_num)
            # 找出答案对应的ans
            for i in range(self.config.batch_size):
                top_ans[i] += 1 if top_ans[i] == 0 else 0
                pred_ans = id2ans[top_ans[i]]
                true_ans = id2ans[true_ans_num[i]]
                # result.append([pred_ans, true_ans])
                #         # true_ques = id2ques[question[i]]
                #         true_id = ids[i]
                all_count += 1
                if true_ans_num[i] == top_ans[i]:
                    true_count += 1

        print("accuracy is %f" % (true_count / all_count))
        print(true_count, all_count)
        print("time cost is:", round(time.time() - start_time, 2), "s")

    def test(self, pretrain_file, test_question, test_image, data):

        ################################### build model ###############################################################
        print('\nBuilding Test Model')
        self.image_input = tf.placeholder(tf.float32, [1, self.config.image_height, self.config.image_width, self.config.image_dim])
        self.ques_input = tf.placeholder(tf.float32, [1, self.config.max_word_length, self.config.word_embedding_dim])

        ques_vector = self._question_module(self.ques_input, reuse=False)
        image_vector = self._image_extract_vgg_multi(self.image_input, reuse=False)

        weight1, att1 = self._fuse_attention_san(image_vector, ques_vector, name='attn_1', reuse=False)
        weight2, att2 = self._fuse_attention_san(att1, ques_vector, name='attn_2', reuse=False)
        # att_l3, att = self._fuse_attention_san(att, ques_vector, name='attn_3', reuse=False)
        # att_l4, att = self._fuse_attention_san(att, ques_vector, name='attn_4', reuse=False)

        ans_predict = self._ans_predict(att2)

        ########################################### build finished ####################################################

        # 加载训练好的模型
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, os.path.join(self.config.checkpoint_path, pretrain_file))

        # 将图像和问题进行编码，输入到网络，然后查看输出即可
        question_embed = data.question_embedding(test_question)
        image_embed =data.read_image_from_path(test_image)

        attn1, attn2, ans_pred = self.sess.run([weight1, weight2, ans_predict], feed_dict={
            self.image_input: image_embed, self.ques_input: question_embed})

        # output and save
        top_ans = np.argmax(ans_pred)
        print(attn2.shape)
        print(top_ans)

    # module -----------------------------------------------------------------------------------------------------------
    # 用vgg和label提取
    def _image_extract_vgg_label(self, image, label, reuse=False, driven=False):
        with tf.variable_scope("image_extract", reuse=reuse):
            vgg = vgg_16(self.config.vgg16_weight_path, self.config.image_height, self.config.image_width)
            vgg.build(image)

            x = tf.layers.dense(tf.layers.flatten(vgg.pool5), self.config.state_size)

            if driven:
                x1 = tf.layers.flatten(vgg.conv1_2) * label[:, 0]                  # 224*224*64
                x2 = tf.layers.flatten(vgg.conv3_3) * label[:, 1]                  # 56*56*256
                x3 = tf.layers.flatten(vgg.conv5_3) * label[:, 2]                  # 14*14*512

                x = tf.concat([x, x1, x2, x3], axis=-1)

        return x

    # 用vgg多尺度提取
    def _image_extract_vgg_multi(self, image, reuse=False):
        with tf.variable_scope("image_extract", reuse=reuse):
            vgg = vgg_16(self.config.vgg16_weight_path, self.config.image_height, self.config.image_width)
            vgg.build(image)

            # x1 = tf.layers.flatten(vgg.conv1_2)                   # 224*224*64
            # x1 = tf.reshape(vgg.conv1_2, [-1, vgg.conv1_2.shape[1] * vgg.conv1_2.shape[2], vgg.conv1_2.shape[3]])
            # x2 = tf.layers.flatten(vgg.conv3_3)                   # 56*56*256
            x2 = tf.reshape(vgg.conv3_3, [-1, vgg.conv3_3.shape[1] * vgg.conv3_3.shape[2], vgg.conv3_3.shape[3]])

            # x3 = tf.layers.flatten(vgg.conv5_3)                   # 14*14*512
            x3 = tf.image.resize_images(vgg.conv5_3, [vgg.conv3_3.shape[1], vgg.conv3_3.shape[2]])
            x3 = tf.reshape(x3, [-1, x3.shape[1] * x3.shape[2], vgg.conv5_3.shape[3]])

            x = tf.concat([x2, x3], axis=-1)
            x = tf.layers.dense(x, 2048)

        return x

    # # 用resnet和label提取
    # def _image_extract_resnet_label(self, image, label, driven=False):
    #     # 图像特征处理模块
    #     with slim.arg_scope(self.resnet_arg_scope(is_training=True)):
    #         net, end_points = resnet_v2.resnet_v2_152(image, reuse=tf.AUTO_REUSE)
    #
    #         if driven:
    #             # net = tf.layers.flatten(net)                                                          # 1， 1， 2048
    #             scale_large = end_points['resnet_v2_152/conv1'] * label[:, 0]                           # 112，112，64
    #             scale_middle = end_points['resnet_v2_152/block1/unit_2/bottleneck_v2'] * label[:, 1]    # 56， 56， 256
    #             scale_small = end_points['resnet_v2_152/block3/unit_35/bottleneck_v2'] * label[:, 2]    # 14， 14， 1024
    #
    #             # 拉伸为一维向量处理
    #             net = tf.layers.flatten(net)
    #             scale_large = tf.layers.flatten(scale_large)
    #             scale_middle = tf.layers.flatten(scale_middle)
    #             scale_small = tf.layers.flatten(scale_small)
    #
    #             net = tf.concat([net, scale_large, scale_middle, scale_small], axis=-1)
    #
    #     return net

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

        # with tf.variable_scope("question_extract", reuse=reuse):
        #     rnn = tf.nn.rnn_cell
        #
        #     # 构建一个2层的LSTM
        #     lstm1 = rnn.BasicLSTMCell(self.config.state_size)
        #     lstm_drop1 = rnn.DropoutWrapper(lstm1, output_keep_prob=1 - self.config.dropout_prob)
        #     lstm2 = rnn.BasicLSTMCell(self.config.state_size)
        #     lstm_drop2 = rnn.DropoutWrapper(lstm2, output_keep_prob=1 - self.config.dropout_prob)
        #     final = rnn.MultiRNNCell([lstm_drop1, lstm_drop2])
        #
        #     # 构建state
        #     state = final.zero_state(self.config.batch_size, tf.float32)
        #
        #     with tf.variable_scope("word_embed", reuse=reuse):
        #         for i in range(self.config.max_question_length):
        #             if i == 0:
        #                 ques_emb_linear = tf.zeros([self.config.batch_size, self.config.word_embedding_dim])
        #             else:
        #                 tf.get_variable_scope().reuse_variables()
        #                 ques_emb_linear = tf.nn.embedding_lookup(self.embeddings, ques_tensor[:, i - 1])
        #
        #             # LSTM based question model
        #             ques_emb_drop = tf.nn.dropout(ques_emb_linear, 1 - self.config.dropout_prob)
        #             ques_emb = tf.tanh(ques_emb_drop)
        #
        #             output, state = final(ques_emb, state)
        #     # print(state.shape)
        #     question_emb = tf.reshape(tf.transpose(state, [2, 1, 0, 3]), [self.config.batch_size, -1])
        #
        #     # state_drop = tf.nn.dropout(state, 1 - self.drop_out_rate)
        #     # state_linear = tf.nn.xw_plus_b(state_drop, self.embed_state_W, self.embed_state_b)
        #     # state_emb = tf.tanh(state_linear)
        #
        #     return question_emb
    #
    # def _question_label(self, ques_vector):
    #     fc_1 = tf.nn.relu(self.fc_layer(ques_vector, 512, name="q_fc1"))
    #     fc_2 = tf.nn.relu(self.fc_layer(fc_1, 128, name="q_fc2"))
    #     fc_3 = self.fc_layer(fc_2, 3, name="q_fc3")
    #     label_result = tf.nn.softmax(fc_3)
    #     return label_result
    #
    # def _fuse_concat(self, image_tensor, text_tensor):
    #     features = tf.nn.dropout(tf.concat([image_tensor, text_tensor], axis=-1),  # [n, 1, 1, 4096]
    #                              keep_prob=self.config.dropout_prob)
    #     return features

    def _fuse_attention_san(self, image_tensor, question_tensor, name='', reuse=False):
        with tf.variable_scope("attention_" + name) as scope:
            if reuse:
                scope.reuse_variables()

            ques_tensor = tf.tile(tf.expand_dims(question_tensor, axis=-2), [1, image_tensor.shape[-2], 1])
            img = tf.nn.tanh(image_tensor)            # [batch size, 224*224, out_dim]
            ques = tf.nn.tanh(ques_tensor)        # [batch size, 1, out_dim]

            # 连接问题和图像
            # ques = tf.expand_dims(ques, axis=-2)                                # [batch size, 1, out_dim]
            # IQ1 = tf.nn.dropout(tf.concat([img, ques], axis=-1), self.config.dropout_prob)
            # IQ2 = tf.layers.dense(IQ1, self.config.state_size)

            # softmax获得注意力
            # p = tf.nn.softmax(IQ2)
            # p = tf.reduce_sum(p, axis=2)
            # p_exp = tf.expand_dims(p, axis=-1)
            # att_layer = p_exp * image_tensor

            # print("p shape, att_layer shape", p.shape, att_layer.shape)

            # # # 最终的注意力结果
            # final_out = att_layer + question_tensor

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
            # # # 将文本特征变成四维的，和图像一样的大小
            # # text_tensor = tf.reshape(text_tensor, (self.config.batch_size, 1, 1, self.config.state_size))         # [n, 1, 1, 2048]
            #
            # # Now both the features from the resnet and lstm are concatenated along depth axis
            # features = tf.nn.dropout(tf.concat([image_tensor, text_tensor], axis=-1),                # [n, 1, 1, 4096]
            #                          keep_prob=self.config.dropout_prob)
            #
            # # attention_map = tf.reshape(features, (self.batch_size, 32 * 32, 4))
            #
            # conv1 = tf.nn.dropout(self.conv2d_layer(features, filters=3000, kernel_size=(1, 1), name="attention_conv1"),
            #                       keep_prob=self.config.dropout_prob)
            # conv2 = self.conv2d_layer(conv1, filters=2048, kernel_size=(1, 1), name="attention_conv2")
            #
            # # Flatenning each attention map to perform softmax
            #
            # attention_map = tf.reshape(conv2, (self.config.batch_size, 32 * 32, 2))                # n, 1024, 2
            # attention_map = tf.nn.softmax(attention_map, axis=1, name="attention_map")
            #
            # image = tf.reshape(image_tensor, (self.config.batch_size, 32*32, 1, 2))              # n, 1024, 2
            # attention = tf.tile(tf.expand_dims(attention_map, 2), (1, 1, 2048, 1))      # n, 1024, 2048, 2
            # # image = tf.tile(image, (1, 1, 1, 2))                                        # n, 1024, 2048, 2
            # weighted = image * attention
            # weighted_average = tf.reduce_mean(weighted, 1)                              # n, 1024, 2
            #
            # # Flatten both glimpses into a single vector
            # weighted_average = tf.reshape(weighted_average, (self.config.batch_size, 2048 * 2))            # n, 4096
            # attention_output = tf.nn.dropout(tf.concat([weighted_average, text_tensor], 1), self.config.dropout_prob)

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

            # # 如果需要两层注意力
            # img_f2 = tf.nn.tanh(flatten_attn)
            # # print(img_f.shape, que_f.shape)
            # IQ_features2 = tf.nn.dropout(tf.nn.tanh(tf.concat([img_f2, que_f], axis=-1)),  # [n, 1, 1, 4096]
            #                             keep_prob=self.config.dropout_prob)
            #
            # attn_fc1_2 = tf.layers.dense(IQ_features2, self.config.image_height * self.config.image_width / 4,
            #                            name='attn_fc1_2', reuse=tf.AUTO_REUSE)
            # attn_fc1r_2 = tf.nn.relu(attn_fc1_2)
            # attn_fc2_2 = tf.layers.dense(attn_fc1r_2, self.config.image_height * self.config.image_width / 8,
            #                            name='attn_fc2_2', reuse=tf.AUTO_REUSE)
            #
            # attention_map_2 = tf.reshape(attn_fc2_2, (
            # self.config.batch_size, int(self.config.image_height * self.config.image_width / 8), 1))
            # attention_map_2 = tf.nn.softmax(attention_map_2, axis=1, name="attention_map2_2")
            # flatten_attn_2 = tf.layers.flatten(attention_map_2)

        return attention_map, flatten_attn

    def _ans_predict(self, fused_feature, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("answer") as scope:
            if reuse:
                scope.reuse_variables()

            # tf.nn.dropout(att, self.config.dropout_prob)
            tempt1 = tf.nn.dropout(tf.nn.relu(fused_feature), keep_prob=self.config.dropout_prob)
            tempt2 = tf.layers.dense(tempt1, self.config.num_class, reuse=tf.AUTO_REUSE)  # [batch size, ans_num]
            tempt3 = tf.nn.dropout(tf.nn.relu(tempt2), keep_prob=self.config.dropout_prob)
            predict_ans = tf.nn.softmax(tempt3)  # [batch size, ans_num]
            # # softmax
            # fc1 = tf.nn.dropout(tf.nn.relu(self.fc_layer(fused_feature, 1024, name="a_fc1")),
            #                     keep_prob=self.config.dropout_prob)
            # fc2 = self.fc_layer(fc1, self.config.num_class, name="a_fc2")

            # answer_prob = tf.nn.softmax(fc2)
            return predict_ans

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


