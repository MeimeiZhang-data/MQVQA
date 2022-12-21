class config():
    def __init__(self):
        # vgg16 path
        backbone_name = 'MQVQA_attn_step4'
        self.vgg16_weight_path = 'data/vgg16.npy'

        # about input path
        self.image_path = 'data/image/'
        self.text_path = 'data/text/'
        self.csv_path = 'data/data.csv'
        self.glove_path = 'data/glove.6B.100d.txt'

        # about save
        self.ans_id_path = 'data/ans2id.npy'
        self.look_up_table_path = 'data/look_up_table/'
        self.checkpoint_path = 'save_file/checkpoint/' + backbone_name + '/'
        self.loss_npy_path = 'save_file/loss/' + backbone_name + '/'
        self.eval_result = 'save_file/eval/' + backbone_name + '/'
        self.test_result = 'save_file/test/' + backbone_name + '/'

        # about input
        self.image_width = 224
        self.image_height = 224
        self.image_dim = 3
        self.max_word_length = 16
        self.label_class = 3

        # about net
        self.word_embedding_dim = 100
        self.dropout_prob = 0.6
        self.state_size = 2048
        self.init_lr = 0.0001
        self.RNN_layers = 2
        self.reuse = False
        self.num_class = 128
        # self.feature_map_shape = 12544       # 112*112

        # about hyper parameters
        self.alpha = 1000000
        self.batch_size = 1
        self.epoches = 10


