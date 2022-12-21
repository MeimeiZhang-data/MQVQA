import tensorflow as tf
import argparse

from RSVQA_config import config
from RSVQA_dataset import VQA_data
from RSVQA_model_RSVQA import mqvqa

tf.reset_default_graph()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='eval', help='train/eval/test')

    # if eval, use this parameter
    parser.add_argument('--pretrain_file', type=str, default='model.ckpt-2',
                        help='If sepcified,load a pretrained model from this file')

    # if test, use these parameters
    parser.add_argument('--test_image', type=str, default='data/image/14282.png',
                        help='the test image path')
    parser.add_argument('--test_question', type=str, default='what does the image show ',
                        help='define a question for predict an image')

    args = parser.parse_args()

    conf = config()
    conf.phase = args.phase
    conf.pretrain_file = args.pretrain_file

    conf.test_image = args.test_image
    conf.test_question = args.test_question

    data = VQA_data(conf)

    con = tf.ConfigProto(allow_soft_placement=True)
    with tf.device('/gpu:0'):
        with tf.Session(config=con) as sess:
            if conf.phase == 'train':                     # 训练阶段
                model = mqvqa(conf, sess)
                model.train(data)
                # model.plot_loss()

            elif conf.phase == 'eval':                    # 验证阶段
                # data = VQA_file(config, statistic=False)
                model = mqvqa(conf, sess)
                model.validation(data)

            else:                                           # 测试阶段
                model = mqvqa(conf, sess)
                model.test(conf.pretrain_file, conf.test_image, conf.test_question, data)


if __name__ == '__main__':
    main()
