import gym
import numpy as np
import universe
from gym.spaces.box import Box
from universe import vectorized
from universe.wrappers import Unvectorize, Vectorize
from gym.wrappers import Monitor

import cv2


def create_atari_env(env_id, create_sub=False):
    env = gym.make(env_id)
    if len(env.observation_space.shape) > 1:
        env = Vectorize(env)
        env = AtariRescale84x84(env)
        env = NormalizedEnv(env)
        env = Unvectorize(env)
    if create_sub:
        self.env = Monitor(self.env, 'tmp/{}'.format(args.env_name), force=True)
    return env


def _process_frame84(frame):
    frame = frame[34:34 + 160, :160]
    frame = cv2.resize(frame, (84, 84))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [1, 84, 84])
    return frame


class AtariRescale84x84(vectorized.ObservationWrapper):

    def __init__(self, env=None):
        super(AtariRescale84x84, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 84, 84])

    def _observation(self, observation_n):
        return [_process_frame84(observation) for observation in observation_n]


class NormalizedEnv(vectorized.ObservationWrapper):

    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def _observation(self, observation_n):
        for observation in observation_n:
            self.num_steps += 1
            self.state_mean = self.state_mean * self.alpha + \
                observation.mean() * (1 - self.alpha)
            self.state_std = self.state_std * self.alpha + \
                observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return [(observation - unbiased_mean) / (unbiased_std + 1e-8) for observation in observation_n]
