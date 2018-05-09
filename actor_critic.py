import argparse
import gym
from gym import wrappers
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from envs import make_atari
from models.actor_critic import Policy

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--cont', action='store_true',
                    help='continue from weights')
parser.add_argument('--record', action='store_true',
                    help='save video')

args = parser.parse_args()

env = make_atari('Pong-v0')
env.seed(args.seed)
# if args.record:
# env = wrappers.Monitor(env, './eval/actor_critic', force=True)
torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

model = Policy(env.observation_space.shape[0], env.action_space.n)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs, state_value = model(Variable(state))
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.data[0]


def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.data[0]
        policy_losses.append(-log_prob * Variable(reward))
        value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([r]))))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]

# model.load_state_dict(torch.load('weights/{}.pt'.format("actor_critic_pong")))

running_length = 10
max_reward = -100
for i_episode in count(1):
    state = env.reset()
    current_reward = 0
    for t in range(10000):
        action = select_action(np.array(state))
        state, reward, done, _ = env.step(action)
        # if args.render:
        # env.render()
        model.rewards.append(reward)
        current_reward+=reward
        if done:
            break

    running_length = running_length * 0.99 + t * 0.01
    finish_episode()
    if i_episode % args.log_interval == 0:
        if current_reward > max_reward:
            max_reward = current_reward
            # torch.save(model.state_dict(), 'weights/{}.pt'.format("actor_critic_pong"))
        print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}\tReward: {:.5f}'.format(
            i_episode, t, running_length, current_reward))