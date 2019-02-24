class DQNConfig(object):

    def __init__(self, FLAGS):

        # ----------- Agent Params
        self.scale = 50
        # self.display = False

        self.max_step = 400 * self.scale
        self.memory_size = 50 * self.scale

        self.batch_size = 16
        # self.random_start = 30
        self.cnn_format = 'NCHW'
        self.discount = 0.6 # epsilon in RL (decay index)
        self.target_q_update_step = 1 * self.scale
        self.learning_rate = 0.001
        self.learning_rate_minimum = 0.00025
        self.learning_rate_decay = 0.96
        self.learning_rate_decay_step = 4 * self.scale

        self.ep_end = 0.20
        self.ep_start = 1.
        self.ep_end_t = 4. * self.memory_size # encourage some new actions

        self.history_length = 4
        self.train_frequency = 4
        self.learn_start = 4. * self.scale

        # self.min_delta = -1
        # self.max_delta = 1

        # self.double_q = False
        # self.dueling = False

        self.test_step = 4 * self.scale
        # self.save_step = self.test_step * 10 # unused yet

        # ----------- Environment Params
        self.env_name = 'Act_Perc'
        self.Lua_PATH = r'../affordance_model/infer.lua'
        # In furthur case you can just use local infomation with 64*64 or 128*128 pixel.
        self.screen_width  = 128
        self.screen_height = 128
        self.scene_num = 24
        self.test_scene_num = 6
        self.max_reward = 1.
        self.min_reward = -1.

        # ----------- Model Params
        self.inChannel = 4 # RGBD
        # self.action_repeat = 4
        self.ckpt_dir = r'./dqn/checkpoint'
        self.model_dir = r'./dqn/model'
        self.is_train = True
        self.is_sim = True
        self.end_metric = 0.85
        
        if FLAGS.use_gpu == False:
            self.cnn_format = 'NHWC'
        else:
            self.cnn_format = 'NCHW'

        if FLAGS.is_train == False:
            self.is_train = False

        if FLAGS.is_sim == False:
            self.is_sim = False
            
    def list_all_member(self):
        params = {}
        for name,value in vars(self).items():
            params[name] = value
        return params
