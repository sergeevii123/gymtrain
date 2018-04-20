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


env = make_atari('SpaceInvaders-v0')
env.seed(args.seed)
if args.record:
    env = wrappers.Monitor(env, './eval/reinforce', force=True)
torch.manual_seed(args.seed)


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

class Policy(nn.Module):
    def __init__(self, num_inputs, action_space):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.affine1 = nn.Linear(32*11*11, 128)
        self.affine2 = nn.Linear(128, action_space)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = x.view(-1, 32*11*11)
        x = F.elu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

policy = Policy(1, 6)
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
# policy.load_state_dict(torch.load('weights/{}.pt'.format("reinforce_invaders")))

running_length = 10
max_reward = 0
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
            # torch.save(policy.state_dict(), 'weights/{}.pt'.format("reinforce_invaders"))
        print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}\tReward: {:.5f}'.format(
            i_episode, t, running_length, current_reward))
