"""
    -- This agent is utilizing the CNN+single U-net layer as for the Deep part
    -- and the input state is 1-channel screen
    -- we need to convert to 4-channel screen
"""
from __future__ import print_function
import os
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from functools import reduce

from .base import BaseModel
from .history import History
from .replay_memory import ReplayMemory
from .ops import linear, conv2d, max_pool, deconv2d, crop_and_concat, clipped_error
from .utils import get_time, save_pkl, load_pkl

class Agent(BaseModel):
    def __init__(self, config, environment, sess):
        super(Agent, self).__init__(config)
        self.sess = sess
        self.weight_dir = r'../weights'

        self.env = environment
        self.history = History(self.config)
        self.memory = ReplayMemory(self.config, self.model_dir)

        with tf.variable_scope('step'):
            self.step_op = tf.Variable(0, trainable=False, name='step')
            self.step_input = tf.placeholder('int32', None, name='step_input')
            self.step_assign_op = self.step_op.assign(self.step_input)

        self._build_dqn()

    def train(self):
        """
            -- Train Model Process
        """
        # eval : In a session, computes and returns the value of this variable.
        start_step = self.step_op.eval() # begin with 0
        start_time = time.time()

        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        max_avg_ep_reward = 0
        ep_rewards, actions = [], []

        # TODO: random init the senario
        screen, reward, action, terminal = self.env.new_random_game()

        for _ in range(self.history_length):
            self.history.add(screen)

        for self.step in tqdm(range(start_step, self.max_step), ncols=70, initial=start_step):
            if self.step == self.learn_start:
                num_game, self.update_count, ep_reward = 0, 0, 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                ep_rewards, actions = [], []

            # 1. predict
            action = self.predict(self.history.get())
            # 2. act
            # TODO: Add to simulation
            screen, reward, terminal = self.env.act(action, is_training=True)
            # 3. observe & store
            self.observe(screen, reward, action, terminal)
            # 4. learn
            self.learn()

            if terminal:
                # 注意！这个代码里训练部分没有epoch的概念，每次结束后，接着之前的step / memory 继续去尝试得到新的场景
                # 所以才会出现memory中间有terminal的情况，在history_length周围有重新开始一次仿真的话，就不get这个sample
                # 加上epoch（本质就是场景重新开始）也不能让step重置，也就是说初始的learn_start的step满足之后，就不会再回到learn_start的懵懂阶段了
                # TODO: random init the senario
                screen, reward, action, terminal = self.env.new_random_game()

                num_game += 1
                ep_rewards.append(ep_reward)
                ep_reward = 0.
            else:
                ep_reward += reward

            # 注意 ep_reward属于在每次simulation里的总和
            # 把每次simulation得到总reward存成list放在ep_rewards里，在test_step来的时候存下来
            # 而 total_reward属于在test_step里的总和
            actions.append(action)
            total_reward += reward

            if self.step >= self.learn_start:
                if self.step % self.test_step == self.test_step - 1:
                    avg_reward = total_reward / self.test_step # get in each action，所以在所有action数目里平均
                    avg_loss = self.total_loss / self.update_count # get total_loss in each mini-batch optimizer，所以在minibatch跑过的次数里平均
                    avg_q = self.total_q / self.update_count # get in each mini-batch optimizer

                    try:
                        # 在simulation的次数上平均
                        max_ep_reward = np.max(ep_rewards)
                        min_ep_reward = np.min(ep_rewards)
                        avg_ep_reward = np.mean(ep_rewards)
                    except:
                        max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

                    print('\navg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d' \
                        % (avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, num_game))

                    if max_avg_ep_reward * 0.9 <= avg_ep_reward:
                        # test之后得到一个比较好的结果，把这个model存下来
                        self.step_assign_op.eval({self.step_input: self.step + 1})
                        self.save_model(self.step + 1)

                        max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)
                    
                    self.inject_summary({
                        'average.reward': avg_reward,
                        'average.loss': avg_loss,
                        'average.q': avg_q,
                        'episode.max reward': max_ep_reward,
                        'episode.min reward': min_ep_reward,
                        'episode.avg reward': avg_ep_reward,
                        'episode.num of game': num_game,
                        'episode.rewards': ep_rewards,
                        'episode.actions': actions,
                        'training.learning_rate': self.learning_rate_op.eval({self.learning_rate_step: self.step}),}
                        , self.step)

                    # 注意这些信息都是每个test_step的轮回里进行存取读出的
                    num_game = 0
                    total_reward = 0.
                    self.total_loss = 0.
                    self.total_q = 0.
                    self.update_count = 0
                    ep_reward = 0.
                    ep_rewards = []
                    actions = []

    def predict(self, s_t, test_ep=None):
        """
            -- According to the estimation result -> get the prediction action (or exploration instead)
        """
        # TODO: 这里s_t是3-D的tensor，要转换成[1,3-D]shape的4D tensor吗？？
        # TODO: it seems this epsilon is a little complex?
        ep = test_ep or (self.ep_end + max(0., (self.ep_start - self.ep_end)
                * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))

        if random.random() < ep:
            # Exploration
            action = random.randrange(0, 48*48) 
        else:
            # Don't need sess.run(self.q_action, feed_dict={xxx})? -> because we run a session in the main function
            # q_action is batch_size(1) * 1 (action is the pixel index(TODO:notice convert to 48*48 coor))
            action = self.q_action.eval({self.s_t: [s_t]})[0]

        return action

    def observe(self, screen, reward, action, terminal):
        """
            Add the action result into history(used to get the current result(next state) by the action)
            -- Notice the history is not used for mini-batch training!!
        """
        reward = max(self.min_reward, min(self.max_reward, reward)) # TODO: need to regularization??

        self.history.add(screen)
        self.memory.add(screen, reward, action, terminal)

    def learn(self):
        """
            Learn from the memory storage every train_frequency (mini-batch loss GD)
            and update target network's weights every target_q_update_step
        """
        if self.step > self.learn_start:
            if self.step % self.train_frequency == 0:
                self.q_learning_mini_batch()

            if self.step % self.target_q_update_step == self.target_q_update_step - 1:
                self.update_target_q_network()

    def q_learning_mini_batch(self):
        """
            Mini batch GD from memory storage
        """
        if self.memory.count < self.history_length:
            return
        else:
            s_t, action, reward, s_t_plus_1, terminal = self.memory.sample()

        t = time.time()

        # notice get eval in the new state
        # s_t_plus_l已经是batch_size*x*x*x的4-D tensor了
        q_t_plus_1 = self.target_q_flat.eval({self.target_s_t: s_t_plus_1})
        # q_t_plus_l = batch_size * 48^2(pixel number)

        terminal = np.array(terminal) + 0.
        max_q_t_plus_1 = np.max(q_t_plus_1, axis=1) #对每个sample要得到所有动作里最大的q_value
        # if terminal -> then reward
        # if not -> reward + decay * max (q)
        target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward

        _, q_t, loss, summary_str = self.sess.run([self.optim, self.q_flat, self.loss, self.q_summary], {
            self.target_q_t: target_q_t,
            self.action: action,
            self.s_t: s_t, # 把s_t喂进去是为了得到q_flat，这样才能优化
            self.learning_rate_step: self.step,})

        self.writer.add_summary(summary_str, self.step)
        self.total_loss += loss
        self.total_q += q_t.mean()
        self.update_count += 1 # 记录优化次数

    """
        ############ Build The Network ############
    """
    def _build_dqn(self):
        """
            Build the Deep Q-table network
            U-net based
        """
        self.w = {} # a dict save each layer weights and bias for estimation network
        self.t_w = {} # a dict save each layer weights and bias for reality network

        # initializer = tf.contrib.layers.xavier_initializer()
        initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu

        # training network U-Net
        with tf.variable_scope('prediction'):
            if self.cnn_format == 'NHWC':
                # a 4-D tensor
                self.s_t = tf.placeholder('float32',
                    [None, self.screen_height, self.screen_width, self.history_length], name='s_t')
            else:
                self.s_t = tf.placeholder('float32',
                    [None, self.history_length, self.screen_height, self.screen_width], name='s_t')

            # Dowmsampling
            # s_t = None*64*64*4(history_length)
            self.l1, self.w['l1_1_w'], self.w['l1_1_b'] = conv2d(self.s_t,
                64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='l1_1')
            # l1 = None*62*62*64
            self.l2, self.w['l1_2_w'], self.w['l1_2_b'] = conv2d(self.l1,
                64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='l1_2')
            # l2 = None*60*60*64

            self.l3 = max_pool(self.l2, [2, 2], [1, 1], self.cnn_format, name='m1')
            # l3 = None*30*30*64

            # Bottom layer
            self.l4, self.w['l2_1_w'], self.w['l2_1_b'] = conv2d(self.l3,
                128, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='l2_1')
            # l4 = None*28*28*128
            self.l5, self.w['l2_2_w'], self.w['l2_2_b'] = conv2d(self.l4,
                128, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='l2_2')            
            # l5 = None*26*26*128

            # Upsampling
            self.l6, self.w['U1_w'] = deconv2d(self.l5, 
                [1, 1], [2, 2], initializer, activation_fn, self.cnn_format, name='U1')
            # l6 = None*52*52*64

            # Cascading (l2, l6)
            self.l7 = crop_and_concat(self.l2, self.l6, self.cnn_format)
            # l7 = None*52*52*128

            # Conv2d->output
            self.l8, self.w['l3_1_w'], self.w['l3_1_b'] = conv2d(self.l7,
                64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='l3_1')
            # l8 = None*50*50*64
            self.l9, self.w['l3_2_w'], self.w['l3_2_b'] = conv2d(self.l8,
                self.history_length, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='l3_2')
            # l9 = None*48*48*4(history_length)
            self.q, self.w['q_w'], self.w['q_b'] = conv2d(self.l9,
                1, [1, 1], [1, 1], initializer, activation_fn, self.cnn_format, name='lq')
            # q = None*48*48*1
            # [ref] https://www.jianshu.com/p/f9b0c2c74488 https://blog.csdn.net/qq_18293213/article/details/72423592

            # 将输出沿着batch size那一层展开，为了后面可以找batch里每个sample的最大value(即操作点所在的位置)
            # dim of l3_flat = batch_size * (H*W*C)
            self.q = tf.squeeze(self.q)
            # q = None*48*48
            shape = self.q.get_shape().as_list()
            self.q_flat = tf.reshape(self.q, [-1, reduce(lambda x, y: x * y, shape[1:])])
            # q_flat = None*48^2
            # Output dims of q_flat is batchsize * pixel_number
            # Find every max q_flat -> action index for each sample in batch
            self.q_action = tf.argmax(self.q_flat, dimension=1)

            # for show in tf board
            q_summary = []
            # average q value for each action over all the samples
            avg_q = tf.reduce_mean(self.q_flat, 0)
            for idx in range(self.q_flat.get_shape().as_list()[1]):
                # idx is range in action size (pixel number)
                q_summary.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))
                self.q_summary = tf.summary.merge(q_summary, 'q_summary')

        # target network
        # The structure is the same with eval network
        with tf.variable_scope('target'):
            if self.cnn_format == 'NHWC':
                # a 4-D tensor
                self.target_s_t = tf.placeholder('float32',
                    [None, self.screen_height, self.screen_width, self.history_length], name='target_s_t')
            else:
                self.target_s_t = tf.placeholder('float32',
                    [None, self.history_length, self.screen_height, self.screen_width], name='target_s_t')

            # Dowmsampling
            # s_t = None*64*64*4(history_length)
            self.target_l1, self.t_w['l1_1_w'], self.t_w['l1_1_b'] = conv2d(self.target_s_t,
                64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='target_l1_1')
            # l1 = None*62*62*64
            self.target_l2, self.t_w['l1_2_w'], self.t_w['l1_2_b'] = conv2d(self.target_l1,
                64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='target_l1_2')
            # l2 = None*60*60*64

            self.target_l3 = max_pool(self.target_l2, [2, 2], [1, 1], self.cnn_format, name='target_m1')
            # l3 = None*30*30*64

            # Bottom layer
            self.target_l4, self.t_w['l2_1_w'], self.t_w['l2_1_b'] = conv2d(self.target_l3,
                128, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='target_l2_1')
            # l4 = None*28*28*128
            self.target_l5, self.t_w['l2_2_w'], self.t_w['l2_2_b'] = conv2d(self.target_l4,
                128, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='target_l2_2')            
            # l5 = None*26*26*128

            # Upsampling
            self.target_l6, self.t_w['U1_w'] = deconv2d(self.target_l5, 
                [1, 1], [2, 2], initializer, activation_fn, self.cnn_format, name='target_U1')
            # l6 = None*52*52*64

            # Cascading (l2, l6)
            self.target_l7 = crop_and_concat(self.target_l2, self.target_l6, self.cnn_format)
            # l7 = None*52*52*128

            # Conv2d->output
            self.target_l8, self.t_w['l3_1_w'], self.t_w['l3_1_b'] = conv2d(self.target_l7,
                64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='target_l3_1')
            # l8 = None*50*50*64
            self.target_l9, self.t_w['l3_2_w'], self.t_w['l3_2_b'] = conv2d(self.target_l8,
                self.history_length, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='target_l3_2')
            # l9 = None*48*48*4(history_length)
            self.target_q, self.t_w['q_w'], self.t_w['q_b'] = conv2d(self.target_l9,
                1, [1, 1], [1, 1], initializer, activation_fn, self.cnn_format, name='target_lq')
            # q = None*48*48*1
            # [ref] https://www.jianshu.com/p/f9b0c2c74488 https://blog.csdn.net/qq_18293213/article/details/72423592

            # 将输出沿着batch size那一层展开，为了后面可以找batch里每个sample的最大value(即操作点所在的位置)
            # dim of l3_flat = batch_size * (H*W*C)
            self.target_q = tf.squeeze(self.target_q)
            # q = None*48*48
            shape = self.target_q.get_shape().as_list()
            self.target_q_flat = tf.reshape(self.target_q, [-1, reduce(lambda x, y: x * y, shape[1:])])
            # q_flat = None*48^2
            # Output dims of q_flat is batchsize * pixel_number
            self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
            self.target_q_with_idx = tf.gather_nd(self.target_q_flat, self.target_q_idx)

        # Used to Set target network params from estimation network (let the t_w_input = w, then assign t_w with t_w_input)
        with tf.variable_scope('pred_to_target'):
            self.t_w_input = {}
            self.t_w_assign_op = {}

            for name in self.w.keys():
                # t_w_input <= w_value
                self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
                self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

        # optimizer
        with tf.variable_scope('optimizer'):
            # 注意有placeholder的都是等待sess.run时输入的
            self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
            self.action = tf.placeholder('int64', [None], name='action')

            # convert to batch_size * action_size matrix
            # e.g. batch = 3, action = 4 init = [3,2,3]
            # [0,0,0,1] [0,0,1,0] [0,0,0,1] -> stands for choosing the 4-th/3-rd/4-th action each
            # although here the batch size is not assigned (none)
            # TODO: action_size is not availble easily for now: 48*48
            action_one_hot = tf.one_hot(self.action, 48*48, 1.0, 0.0, name='action_one_hot')
            # Only set the chosen action to have value, with others = none
            # reduction_indices = axis -> q_acted = batch_size * 1
            # 利用one-hot方法只保留采取动作对应的q值
            q_acted = tf.reduce_sum(self.q_flat * action_one_hot, reduction_indices=1, name='q_acted')

            self.delta = self.target_q_t - q_acted

            self.global_step = tf.Variable(0, trainable=False)

            self.loss = tf.reduce_mean(clipped_error(self.delta), name='loss')
            self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
            # 目前设置minimum和initial相等，所以没有learning rate decay
            self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
                tf.train.exponential_decay(
                    self.learning_rate,
                    self.learning_rate_step,
                    self.learning_rate_decay_step,
                    self.learning_rate_decay,
                    staircase=True))
            # TODO: We can change the momentum and epsilon params
            self.optim = tf.train.RMSPropOptimizer(
                self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)
        
        # display all the params in the tfboard by summary
        with tf.variable_scope('summary'):
            # save every Mini_batch GD
            scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', \
                'episode.max reward', 'episode.min reward', 'episode.avg reward', 'episode.num of game', 'training.learning_rate']

            self.summary_placeholders = {}
            self.summary_ops = {}

            for tag in scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
                self.summary_ops[tag]  = tf.summary.scalar("%s/%s" % (self.env_name, tag), self.summary_placeholders[tag])

            histogram_summary_tags = ['episode.rewards', 'episode.actions']

            for tag in histogram_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
                self.summary_ops[tag]  = tf.summary.histogram(tag, self.summary_placeholders[tag])

            self.writer = tf.summary.FileWriter('../logs', self.sess.graph)

        tf.global_variables_initializer().run()

        self._saver = tf.train.Saver(self.w.values() + [self.step_op], max_to_keep = 10, keep_checkpoint_every_n_hours=1.0)
        #　这个saver会覆盖base model里生成的saver

        # What happened? TODO: if none? (just from the initial)
        self.load_model()
        self.update_target_q_network()

    def update_target_q_network(self):
        """
            Assign estimation network weights to target network. (not simultaneous)
        """
        for name in self.w.keys():
            self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})

    # Unused
    def save_weight_to_pkl(self):
        """
            -- Save estimation network weights to pkl file
        """
        if not os.path.exists(self.weight_dir):
            os.makedirs(self.weight_dir)

        for name in self.w.keys():
            save_pkl(self.w[name].eval(), os.path.join(self.weight_dir, "%s.pkl" % name))

    # Unused
    def load_weight_from_pkl(self, cpu_mode=False):
        with tf.variable_scope('load_pred_from_pkl'):
            self.w_input = {}
            self.w_assign_op = {}
   
            for name in self.w.keys():
                self.w_input[name] = tf.placeholder('float32', self.w[name].get_shape().as_list(), name=name)
                self.w_assign_op[name] = self.w[name].assign(self.w_input[name])

        for name in self.w.keys():
            self.w_assign_op[name].eval({self.w_input[name]: load_pkl(os.path.join(self.weight_dir, "%s.pkl" % name))})

        # every time upload the saved weights -> means begin a new training so we should assign the target-network
        self.update_target_q_network()

    def inject_summary(self, tag_dict, step):
        """
            add infos to summary -> saved in ../logs
            # 每 test_step执行一次(存scalar，但不存histogram)
            # 和MiniBatch每次优化时存的东西不一样(存histogram，但不存scalar)
            # 这个函数可以让你通过tag_dict来指定要存什么东西，即存什么scalar
            调用的是tf.variable_scope('summary')的图
        """
        summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], 
            {self.summary_placeholders[tag]: value for tag, value in tag_dict.items()})
        for summary_str in summary_str_lists:
            self.writer.add_summary(summary_str, self.step)

    def play(self, n_step=10000, n_episode=100, test_ep=None, render=False):
        """
            -- Use for testing simulation
            -- use history here, not memory. Although I don't know why we need history?
        """
        if test_ep == None:
            test_ep = self.ep_end

        if not self.display:
            tmp_dir = '../tmp/%s-%s' % (self.env_name, get_time())
            # TODO: env start
            self.env.env.monitor.start(tmp_dir)

        best_reward, best_idx = 0, 0
        for idx in range(n_episode):
            # TODO: env start feedback
            print("="*30)
            print(" [*] Test Episode %d" %idx, " begins ")
            screen, reward, action, terminal = self.env.new_random_game()
            current_reward = 0

            # initial add
            for _ in range(self.history_length):
                self.history.add(screen)

            end_step = 0
            for end_step in tqdm(range(n_step), ncols=70):
                # 1. predict
                action = self.predict(self.history.get(), test_ep)
                # 2. act
                # TODO: Get the feedback
                screen, reward, terminal = self.env.act(action, is_training=False)
                # 3. observe
                self.history.add(screen)

                current_reward += reward
                if terminal:
                    break

            if current_reward > best_reward:
                best_reward = current_reward
                best_idx = idx

            print(" End in step : %d" %(end_step))
            print(" Best episode : [%d] Best reward : %d" % (best_idx, best_reward))
            print("="*30)

        if not self.display:
            # TODO: Close the simulation
            self.env.env.monitor.close()
            #gym.upload(tmp_dir, writeup='https://github.com/devsisters/DQN-tensorflow', api_key='')