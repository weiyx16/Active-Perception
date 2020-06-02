"""
    Tensorflow = 1.4.1-gpu
    Python 3.6.3
    Torch7
    CUDA 8.0
    Cudnn 6.0
"""

import random
import tensorflow as tf 
from dqn.agent import Agent
from simulation.environment import DQNEnvironment
from experiment.environment import REALEnvironment
from config import DQNConfig
import pprint
pp = pprint.PrettyPrinter().pprint

"""
    -- Params config
"""
flags = tf.app.flags
# Params / Default value / Help

# Etc
flags.DEFINE_boolean('use_gpu', True, 'Whether to use gpu or not')
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_boolean('is_sim', True, 'Whether test in simulation or true ur')
# Notice the affordance model need about 2G Ram of the GPU so... you had better use less than 2/3 in 8G titan
flags.DEFINE_string('gpu_fraction', '60/100', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')

FLAGS = flags.FLAGS

# Set random seed
tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

def calc_gpu_fraction(fraction_string):
    idx, num = fraction_string.split('/')
    idx, num = float(idx), float(num)

    # fraction = 1 / (num - idx + 1)
    fraction = idx/num
    print(" [*] Use GPU Fraction is : %.4f" % fraction)
    return fraction

def main(_):
    if FLAGS.gpu_fraction == '':
        raise ValueError("--gpu_fraction should be defined")
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction = calc_gpu_fraction(FLAGS.gpu_fraction))

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        config = DQNConfig(FLAGS) or FLAGS
        print("\n [*] Current Configuration")
        pp(config.list_all_member())

        # Notice before the process 
        # Code in remoteApi.start(19999) in Vrep otherwise it may cause some unpredictable problem
        
        if not tf.test.is_gpu_available() and FLAGS.use_gpu:
            raise Exception("use_gpu flag is true when no GPUs are available")

        if config.is_train:
            env = DQNEnvironment(config)
            agent = Agent(config, env, sess)
            agent.train()
        else:
            if config.is_sim:
                env = DQNEnvironment(config)
                agent = Agent(config, env, sess)
                agent.play()
                agent.randomplay()
            else:
                env = REALEnvironment(config)
                agent = Agent(config, env, sess)
                agent.exp_play()

        env.close()

if __name__ == '__main__':
    tf.app.run()
