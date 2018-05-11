import argparse
import gym
import cv2
from envs import make_atari, WarpFrame, AtariRescale42x42
import random

env = gym.make('Pong-v0')
env = AtariRescale42x42(env)
# env = WarpFrame(env)
done = False
env.reset()
while not done:
    print("step")
    observation, _, done,_ = env.step(random.randint(0, env.action_space.n-1))
    print(observation.shape)
    cv2.imshow('image', observation.reshape(42,42,-1))
    c = cv2.waitKey()
    if c == 27:
        break
