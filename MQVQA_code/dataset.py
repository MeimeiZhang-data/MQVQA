import pandas as pd
from skimage import io, transform
import os
import numpy as np
import csv
import re

class VQA_data():
    def __init__(self, configs, first_time=False):
        self.config = configs
        # first time should run write2csv
        if first_time:
            self.write2csv()
            self.build_ans_vocabulary()
            self.glove100_used()

        self.ques_used = np.load(os.path.join(self.config.look_up_table_path, 'ques_used.npy')).item()

    # 首次运行模型需要制作csv文件，词汇表，以及加载需要用的Glove向量----------------------------------------------------
    # 0. 制作csv表
    def write2csv(self):
        files = os.listdir(self.config.text_path)

        with open(self.config.csv_path, mode='a', encoding='utf-8', newline='') as f:
            csv_writer = csv.DictWriter(f, fieldnames=['image_path', 'question', 'answer', 'ques_label'])
            csv_writer.writeheader()
            for file in files:
                each_text_path = os.path.join(self.config.text_path, file)
                each_image_path = os.path.join(self.config.image_path, file.split('.')[0]+'.png')
                num_qa, ques_set, answ_set, label_set = self.text2qa(each_text_path)
                for k in range(num_qa):
                    iqa_tuple = {
                        'image_path': each_image_path,
                        'question': ques_set[k],
                        'answer': answ_set[k],
                        'ques_label': label_set[k]
                    }
                    csv_writer.writerow(iqa_tuple)

    # 0. 读取txt中的问题和答案和标签
    def text2qa(self, file_path):
        """
        将单个text文本中的句子分成问题和答案
        :param file_path:传入的text文本路径
        :return: 返回问题的数量，问题和答案
        """
        with open(file_path, "rb") as f:
            data = f.readlines()

        questions, answers, labels = [], [], []
        num_qa_pair = int(len(data)/2)                                           # 判断有多少个qa_pair

        for i in range(num_qa_pair):                                             # 然后循环读取每组QA
            # 问题中的书写可能不规范。先将问题做预处理，将中文？改为英文？
            ques = re.sub("？", "?", data[i*2].decode().lower().split('\r')[0])
            questions.append(ques.split('?')[0])
            answers.append(data[i*2+1].decode().lower().split('\r')[0])
            word_set = ques.split('?')[0].split()
            if 'scene' in word_set or 'theme' in word_set:
                label = 0
            elif 'many' in word_set or 'object' in word_set or 'shape' in word_set:
                label = 2
            else:
                label = 1
            labels.append(label)

        return num_qa_pair, questions, answers, labels

    # 0. 制作答案的词汇表
    def build_ans_vocabulary(self):
        # read all answers
        all_ans = []
        with open(self.config.csv_path, 'r') as f:
            reader = csv.reader(f)
            for sample in list(reader):
                all_ans.append(sample[-2])

        all_ans.remove('answer')
        ans2id = {ans: num + 1 for num, ans in enumerate(list(set(all_ans)))}
        np.save(self.config.ans_id_path, ans2id)

    # 0. 制作Glove向量
    def glove100_used(self):
        all_data = self.read_data_from_csv()

        ques_used, word_used = {}, {}

        # 对问题进行去重
        ques_temp = all_data['question']
        unique_ques = set(ques_temp)

        # 对单词进行去重
        word_temp = []
        for q in unique_ques:
            word_in_q = q.split()
            for word in word_in_q:
                word_temp.append(word)
        unique_word = set(word_temp)

        # 分别建立问题表和词汇表，建立每个问题和单词之间的映射
        word_embeddings = {}
        with open(self.config.glove_path, 'r', encoding='utf—8') as glove:  # 以gbk编码读取
            for line in glove.readlines():
                line = list(line.split())
                # c = np.array(line).dtype        #此时的词向量是字符串格式需要后期转化为float型
                word = line[0]
                coefs = np.asarray(line[1:], dtype='float32')
                word_embeddings[word] = coefs

        for embed_w in unique_word:
            word_used[embed_w] = word_embeddings[embed_w]

        # print(word_used)
        for embed_q in unique_ques:
            all_ques_word = embed_q.split()
            temp = np.zeros((self.config.max_word_length, self.config.word_embedding_dim))
            for q_th in range(len(all_ques_word)):
                temp[q_th, :] = word_used[all_ques_word[q_th]]
            ques_used[embed_q] = temp

        # 保存问题表和词汇表的结果，保存为npy结果
        if not os.path.exists(self.config.look_up_table_path):
            os.makedirs(self.config.look_up_table_path)

        np.save(os.path.join(self.config.look_up_table_path, 'ques_used.npy'), ques_used)
        np.save(os.path.join(self.config.look_up_table_path, 'word_used.npy'), word_used)
    # ------------------------------------------------------------------------------------------------------------------
    # 1. 从csv读取数据并构建
    def read_data_from_csv(self):
        return pd.read_csv(self.config.csv_path)

    # 2. 读取一个batch的图像数据
    def read_image_from_path(self, image_path_set):
        img = np.zeros([self.config.batch_size, self.config.image_height, self.config.image_width, self.config.image_dim])
        for i in range(len(image_path_set)):
            # print("img path is ", image_path_set[i])
            temp = io.imread(image_path_set[i])
            img[i, :, :, :] = transform.resize(temp, [self.config.image_height, self.config.image_width, self.config.image_dim])
        return img

    # 3. 制作一个batch的问题嵌入
    def question_embedding(self, question_set):
        # 用Glove方法
        all_ques = np.zeros([self.config.batch_size, self.config.max_word_length, self.config.word_embedding_dim])
        # print(len(question_set), question_set)
        for i in range(self.config.batch_size):
            ques = question_set[i]
            ques_vec = self.ques_used[ques]
            #     for order, word in enumerate(ques_word):
            #         word_vec = word_embeddings[word]
            all_ques[i, :, :] = ques_vec
        #
        # # print(all_ques.shape)
        return all_ques

    # 4. 制作一个batch的答案向量
    def ans2vec(self, answer_set, onehot=False):
        ans2id = np.load(self.config.ans_id_path).item()
        id = []
        for answer in answer_set:
            ans_id = ans2id[answer]
            id.append(ans_id)

        # 再将id转化为onehot形式
        new_id = np.zeros((self.config.batch_size, self.config.num_class))
        if onehot:
            for i in range(len(answer_set)):
                new_id[i, id[i]] = 1
            id = new_id

        return id

    # 5. 制作一个batch的label向量
    def label2vec(self, label_set, onehot=False):
        id = []
        for l in label_set:
            id.append(l)

        labels = np.zeros((self.config.batch_size, self.config.label_class))
        # labels[:, ] = label_set[:]
        if onehot:
            for i in range(len(label_set)):
                if label_set[i] == 0:
                    labels[i] = [1, 0, 0]
                elif label_set[i] == 1:
                    labels[i] = [0, 1, 0]
                else:
                    labels[i] = [0, 0, 1]
            id = labels

        return id

    # 6. 读取一个batch的数据
    def read_a_batch_data(self, image_set, question_set, answer_set, label_set, onehot=False):
        images = self.read_image_from_path(image_set)
        quests = self.question_embedding(question_set)
        answes = self.ans2vec(answer_set, onehot=onehot)
        labels = self.label2vec(label_set, onehot=False)

        return images, quests, answes, labels


if __name__ == '__main__':
    from configure import config
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    conf = config()

    data = VQA_data(conf)