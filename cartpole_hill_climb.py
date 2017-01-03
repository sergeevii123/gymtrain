import gym
import numpy as np
from gym import wrappers
env = gym.make('CartPole-v0')
# env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)

def run_episode(env, parameters):
    observation = env.reset()
    totalreward = 0
    for _ in xrange(200):
        # env.render()
        action = 0 if np.matmul(parameters,observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward


noise_scaling = 0.1
parameters = np.random.rand(4) * 2 - 1
episodes_per_update = 10
bestreward = 0
for _ in xrange(10000):
    newparams = parameters + (np.random.rand(4) * 2 - 1)*noise_scaling
    reward = 0
    for k in xrange(episodes_per_update):
        run = run_episode(env, newparams)
        reward += run
    print _, newparams, reward
    if reward > bestreward:
        bestreward = reward
        parameters = newparams
        if reward == 2000:
            print 'solved'
            # env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)
            for j in xrange(100):
                reward = run_episode(env, parameters)
                print j, reward
            break