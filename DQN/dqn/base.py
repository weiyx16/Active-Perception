import os
import tensorflow as tf

class BaseModel(object):
    """
        Abstract object representing an Reader model.
    """
    def __init__(self, config):
        self._saver = None
        self.config = config.list_all_member()

        for attr in self.config:
            if attr.startswith('__'):
                continue
            name = attr if not attr.startswith('_') else attr[1:]
            setattr(self, name, self.config[attr])

        self.config = config

    def save_model(self, step = None):
        print(" [*] Saving checkpoints...")
        model_name = type(self).__name__

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver.save(self.sess, self.checkpoint_dir, global_step=step)

    def load_model(self):
        print(" [*] Loading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print(" [*] Load SUCCESS: %s" % ckpt.model_checkpoint_path)
            return True
        else:
            print(" [!] Load FAILED: %s" % self.checkpoint_dir)
            print(" [*] Created model with fresh parameters.")
            self.sess.run(tf.global_variables_initializer())
            return False

    @property
    def checkpoint_dir(self):
        return os.path.join(self.ckpt_dir, self.env_name + '/dqn_model.ckpt')

    @property
    def saver(self):
        if self._saver == None:
            self._saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=1.0)
            return self._saver