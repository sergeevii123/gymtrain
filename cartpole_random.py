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


bestparams = None
bestreward = 0
for _ in xrange(10000):
    parameters = np.random.rand(4) * 2 - 1
    reward = run_episode(env,parameters)
    print _, reward
    if reward > bestreward:
        bestreward = reward
        bestparams = parameters
        # considered solved if the agent lasts 200 timesteps
        if reward == 200:
            print 'solved'
            env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)
            for j in xrange(100):
                reward = run_episode(env, bestparams)
                print j, reward
            break