import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from envs import make_atari
import random
from gym import wrappers

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 42)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--cont', action='store_true',
                    help='continue from weights')
parser.add_argument('--record', action='store_true',
                    help='save video')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = make_atari('Pong-v0')
env.seed(args.seed)
# if args.record:
# env = wrappers.Monitor(env, './eval/reinforce', force=True)
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self, num_inputs, action_space):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.affine1 = nn.Linear(32*3*11, 256)
        self.affine2 = nn.Linear(256, action_space)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(-1, 32*3*11)
        x = F.elu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy(env.observation_space.shape[0], env.action_space.n)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(Variable(state))
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.data[0]


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

# if args.cont:
policy.load_state_dict(torch.load('weights/{}.pt'.format("reinforce_pong")))

running_length = 10
max_reward = -100
for i_episode in count(1):
    state = env.reset()
    current_reward = 0
    for t in range(10000):
        action = select_action(np.array(state))
        state, reward, done, _ = env.step(action)
        # if args.render:
        env.render()
        policy.rewards.append(reward)
        current_reward+=reward
        if done:
            break

    running_length = running_length * 0.99 + t * 0.01
    finish_episode()
    if i_episode % args.log_interval == 0:
        if current_reward > max_reward:
            max_reward = current_reward
            # torch.save(policy.state_dict(), 'weights/{}.pt'.format("reinforce_pong"))
        print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}\tReward: {:.5f}'.format(
            i_episode, t, running_length, current_reward))
