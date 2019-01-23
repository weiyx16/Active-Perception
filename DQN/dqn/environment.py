"""
  Simulation env.
  # That is the api with simulation file
"""
import random
# from .utils import rgb2gray, imresize


class DQNEnvironment(object):
    def __init__(self, config):

        pass
    
    def begin(self):
        pass
    
    def over(self):
        pass

    def act(self, action, is_training=True):
        pass
        # return camera_data(screen), reward, terminal

    def new_scene(self):
        pass
        # return camera_data(screen), reward(0), action(-1), terminal