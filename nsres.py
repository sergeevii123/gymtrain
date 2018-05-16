import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from envs import make_atari
import torch.multiprocessing as mp
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import torch.optim as optim
import os
import argparse
import torch.legacy.optim as legacyOptim

class ES(torch.nn.Module):

    def __init__(self, num_inputs, action_space):
        """
        Really I should be using inheritance for the small_net here
        """
        super(ES, self).__init__()
        num_outputs = action_space.n
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.affine1 = nn.Linear(64 * 5 * 5, 512)
        self.actor_linear = nn.Linear(512, num_outputs)

    def forward(self, inputs):
        x = F.selu(self.conv1(inputs))
        x = F.selu(self.conv2(x))
        x = F.selu(self.conv3(x))
        x = x.view(-1, 64 * 5 * 5)
        x = F.selu(self.affine1(x))
        return self.actor_linear(x)

    def count_parameters(self):
        count = 0
        for param in self.parameters():
            count += param.data.numpy().flatten().shape[0]
        return count

    def es_params(self):
        """
        The params that should be trained by ES (all of them)
        """
        return [(k, v) for k, v in zip(self.state_dict().keys(),
                                       self.state_dict().values())]

def do_rollouts(args, models, random_seeds, return_queue, env, are_negative):
    all_returns = []
    all_num_frames = []
    for model in models:
        state = env.reset()
        state = torch.from_numpy(np.array(state))
        this_model_return = 0
        this_model_num_frames = 0
        # Rollout
        done = False
        # for step in range(args.max_episode_length):
        while not done:
            logit = model((Variable(state.unsqueeze(0), volatile=True)))

            prob = F.softmax(logit, dim=-1)
            action = prob.max(1)[1].data.numpy()
            state, reward, done, _ = env.step(action[0])
            this_model_return += reward
            this_model_num_frames += 1
            if done:
                break
            state = torch.from_numpy(np.array(state))
        all_returns.append(this_model_return)
        all_num_frames.append(this_model_num_frames)
    return_queue.put((random_seeds, all_returns, all_num_frames, are_negative))


def perturb_model(args, model, random_seed, env):
    new_model = ES(env.observation_space.shape[0], env.action_space)
    anti_model = ES(env.observation_space.shape[0], env.action_space)
    new_model.load_state_dict(model.state_dict())
    anti_model.load_state_dict(model.state_dict())
    np.random.seed(random_seed)
    for (k, v), (anti_k, anti_v) in zip(new_model.es_params(), anti_model.es_params()):
        eps = np.random.normal(0, 1, v.size())
        v += torch.from_numpy(args.sigma * eps).float()
        anti_v += torch.from_numpy(args.sigma * -eps).float()
    return [new_model, anti_model]


optimConfig = []
averageReward = []
maxReward = []
minReward = []
episodeCounter = []


def gradient_update(args, synced_model, returns, random_seeds, neg_list,
                    num_eps, num_frames, chkpt_dir, unperturbed_results):
    def fitness_shaping(returns):
        """
        A rank transformation on the rewards, which reduces the chances
        of falling into local optima early in training.
        """
        sorted_returns_backwards = sorted(returns)[::-1]
        lamb = len(returns)
        shaped_returns = []
        denom = sum([max(0, math.log(lamb / 2 + 1, 2) -
                         math.log(sorted_returns_backwards.index(r) + 1, 2))
                     for r in returns])
        for r in returns:
            num = max(0, math.log(lamb / 2 + 1, 2) -
                      math.log(sorted_returns_backwards.index(r) + 1, 2))
            shaped_returns.append(num / denom + 1 / lamb)
        return shaped_returns

    def unperturbed_rank(returns, unperturbed_results):
        nth_place = 1
        for r in returns:
            if r > unperturbed_results:
                nth_place += 1
        rank_diag = ('%d out of %d (1 means gradient is uninformative)' % (nth_place, len(returns) + 1))
        return rank_diag, nth_place

    batch_size = len(returns)
    assert batch_size == args.n
    assert len(random_seeds) == batch_size
    shaped_returns = fitness_shaping(returns)
    rank_diag, rank = unperturbed_rank(returns, unperturbed_results)
    if not args.silent:
        print('Episode num: %d\n'
              'Average reward: %f\n'
              'Variance in rewards: %f\n'
              'Max reward: %f\n'
              'Min reward: %f\n'
              'Batch size: %d\n'
              'Max episode length: %d\n'
              'Sigma: %f\n'
              'Learning rate: %f\n'
              'Total num frames seen: %d\n'
              'Unperturbed reward: %f\n'
              'Unperturbed rank: %s\n' %
              (num_eps, np.mean(returns), np.var(returns), max(returns),
               min(returns), batch_size,
               args.max_episode_length, args.sigma, args.lr, num_frames,
               unperturbed_results, rank_diag))

    averageReward.append(np.mean(returns))
    episodeCounter.append(num_eps)
    maxReward.append(max(returns))
    minReward.append(min(returns))
    #
    # pltAvg, = plt.plot(episodeCounter, averageReward, label='average')
    # pltMax, = plt.plot(episodeCounter, maxReward, label='max')
    # pltMin, = plt.plot(episodeCounter, minReward, label='min')
    #
    # plt.ylabel('rewards')
    # plt.xlabel('episode num')
    # plt.legend(handles=[pltAvg, pltMax, pltMin])
    #
    # fig1 = plt.gcf()
    #
    # plt.draw()
    # fig1.savefig('graph.png', dpi=100)

    # For each model, generate the same random numbers as we did
    # before, and update parameters. We apply weight decay once.
    # if args.useAdam:
    # globalGrads = None
    # for i in range(args.n):
    #     np.random.seed(random_seeds[i])
    #     multiplier = -1 if neg_list[i] else 1
    #     reward = shaped_returns[i]
    #
    #     localGrads = []
    #     idx = 0
    #     for k, v in synced_model.es_params():
    #         eps = np.random.normal(0, 1, v.size())
    #         grad = torch.from_numpy((args.n * args.sigma) * (reward * multiplier * eps)).float()
    #
    #         localGrads.append(grad)
    #
    #         if len(optimConfig) == idx:
    #             optimConfig.append({'learningRate': args.lr})
    #         idx = idx + 1
    #
    #     if globalGrads == None:
    #         globalGrads = localGrads
    #     else:
    #         for i in range(len(globalGrads)):
    #             globalGrads[i] = torch.add(globalGrads[i], localGrads[i])
    # # print(globalGrads)
    # idx = 0
    # for k, v in synced_model.es_params():
    #     r, _ = legacyOptim.adam(lambda x: (1, -globalGrads[idx]), v, optimConfig[idx])
    #     v.copy_(r)
    #     idx = idx + 1
    # else:
    #     # For each model, generate the same random numbers as we did
    #     # before, and update parameters. We apply weight decay once.
    for i in range(args.n):
        np.random.seed(random_seeds[i])
        multiplier = -1 if neg_list[i] else 1
        reward = shaped_returns[i]
        for k, v in synced_model.es_params():
            eps = np.random.normal(0, 1, v.size())
            v += torch.from_numpy(args.lr / (args.n * args.sigma) *
                                  (reward * multiplier * eps)).float()
    args.lr *= args.lr_decay

    torch.save(synced_model.state_dict(),
               os.path.join("weights", 'es.pth'))
    return synced_model


def render_env(args, model, env):
    while True:
        state = env.reset()
        state = torch.from_numpy(np.array(state))
        this_model_return = 0
        done = False
        while not done:
            logit = model((Variable(state.unsqueeze(0), volatile=True)))

            prob = F.softmax(logit, dim=-1)
            action = prob.max(1)[1].data.numpy()
            state, reward, done, _ = env.step(action[0, 0])
            env.render()
            this_model_return += reward
            state = torch.from_numpy(np.array(state))
        print('Reward: %f' % this_model_return)


def generate_seeds_and_models(args, synced_model, env):
    """
    Returns a seed and 2 perturbed models
    """
    np.random.seed()
    random_seed = np.random.randint(2 ** 30)
    two_models = perturb_model(args, synced_model, random_seed, env)
    return random_seed, two_models


def train_loop(args, synced_model, env, chkpt_dir):
    def flatten(raw_results, index):
        notflat_results = [result[index] for result in raw_results]
        return [item for sublist in notflat_results for item in sublist]

    print("Num params in network %d" % synced_model.count_parameters())
    num_eps = 0
    total_num_frames = 0
    for _ in range(args.max_gradient_updates):
        processes = []
        return_queue = mp.Queue()
        all_seeds, all_models = [], []
        # Generate a perturbation and its antithesis
        for j in range(int(args.n / 2)):
            random_seed, two_models = generate_seeds_and_models(args,
                                                                synced_model,
                                                                env)
            # Add twice because we get two models with the same seed
            all_seeds.append(random_seed)
            all_seeds.append(random_seed)
            all_models += two_models
        assert len(all_seeds) == len(all_models)
        # Keep track of which perturbations were positive and negative
        # Start with negative true because pop() makes us go backwards
        is_negative = True
        # Add all peturbed models to the queue
        while all_models:
            perturbed_model = all_models.pop()
            seed = all_seeds.pop()
            p = mp.Process(target=do_rollouts, args=(args,
                                                     [perturbed_model],
                                                     [seed],
                                                     return_queue,
                                                     env,
                                                     [is_negative]))
            p.start()
            processes.append(p)
            is_negative = not is_negative
        assert len(all_seeds) == 0
        # Evaluate the unperturbed model as well
        p = mp.Process(target=do_rollouts, args=(args, [synced_model],
                                                 ['dummy_seed'],
                                                 return_queue, env,
                                                 ['dummy_neg']))
        p.start()
        processes.append(p)
        for p in processes:
            p.join()
        raw_results = [return_queue.get() for p in processes]
        seeds, results, num_frames, neg_list = [flatten(raw_results, index) for index in [0, 1, 2, 3]]
        # Separate the unperturbed results from the perturbed results
        _ = unperturbed_index = seeds.index('dummy_seed')
        seeds.pop(unperturbed_index)
        unperturbed_results = results.pop(unperturbed_index)
        _ = num_frames.pop(unperturbed_index)
        _ = neg_list.pop(unperturbed_index)

        total_num_frames += sum(num_frames)
        num_eps += len(results)
        synced_model = gradient_update(args, synced_model, results, seeds,
                                       neg_list, num_eps, total_num_frames,
                                       chkpt_dir, unperturbed_results)
        # if args.variable_ep_len:
        #     args.max_episode_length = int(2 * sum(num_frames) / len(num_frames))


parser = argparse.ArgumentParser(description='ES')
parser.add_argument('--env-name', default='PongDeterministic-v4',
                    metavar='ENV', help='environment')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--lr-decay', type=float, default=1, metavar='LRD',
                    help='learning rate decay')
parser.add_argument('--sigma', type=float, default=0.05, metavar='SD',
                    help='noise standard deviation')
parser.add_argument('--useAdam', action='store_true',
                    help='bool to determine if to use adam optimizer')
parser.add_argument('--n', type=int, default=20, metavar='N',
                    help='batch size, must be even')
parser.add_argument('--max-episode-length', type=int, default=100000,
                    metavar='MEL', help='maximum length of an episode')
parser.add_argument('--max-gradient-updates', type=int, default=100000,
                    metavar='MGU', help='maximum number of updates')
parser.add_argument('--restore', default='', metavar='RES',
                    help='checkpoint from which to restore')
parser.add_argument('--variable-ep-len', action='store_true',
                    help="Change max episode length during training")
parser.add_argument('--silent', action='store_true',
                    help='Silence print statements during training')
parser.add_argument('--test', action='store_true',
                    help='Just render the env, no training')

args = parser.parse_args()

env = make_atari('Pong-v0')
synced_model = ES(env.observation_space.shape[0], env.action_space)
train_loop(args, synced_model, env, "weights")