import gym
import numpy as np
import tensorflow as tf
from gym import wrappers

env = gym.make('CartPole-v0')
# env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)
epsilon = 1e-3

def policy_gradient():
    state = tf.placeholder(dtype=tf.float32, shape=(None, 4))
    actions = tf.placeholder(dtype=tf.float32, shape=(None, 2))
    advantage = tf.placeholder(dtype=tf.float32, shape=(None, 1))

    w1 = tf.Variable(tf.zeros([4, 2]))
    est_probs = tf.nn.softmax(tf.matmul(state, w1))

    acc = tf.reduce_sum(tf.mul(est_probs, actions), reduction_indices=[1])
    log_probs = tf.log(acc)
    loss = -tf.reduce_sum(log_probs * advantage, reduction_indices=[1])
    optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)

    return optimizer, state, actions, advantage, est_probs, loss


def value_gradient():
    state = tf.placeholder(dtype=tf.float32, shape=(None, 4))
    val = tf.placeholder(dtype=tf.float32, shape=(None, 1))

    w1 = tf.Variable(tf.random_normal([4, 10]))
    b1 = tf.Variable(tf.zeros([10]))
    h1 = tf.nn.relu(tf.matmul(state, w1) + b1)

    w2 = tf.Variable(tf.random_normal([10, 1]))
    b2 = tf.Variable(tf.zeros([1]))
    val_est = tf.matmul(h1, w2) + b2

    loss = tf.nn.l2_loss(val_est - val)
    optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
    return optimizer, state, val, val_est, loss


policy_opt, policy_state_var, policy_action_var, policy_advantages_var, policy_action_est, policy_loss = policy_gradient()
value_opt, value_state_var, value_val_var, value_val_est, value_loss = value_gradient()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

gamma = 0.97

pol_loss_hist = []
val_loss_hist = []
reward_hist = []

for epoch in xrange(10000):
    obs = env.reset()
    states = []
    actions = []
    transitions = []
    total_reward = 0.0
    numofep = 1 #int(epoch/300)+1
    for _ in xrange(numofep):
        env.reset()
        for _ in xrange(200):
            probs = sess.run(policy_action_est, feed_dict={policy_state_var: obs.reshape((1, obs.shape[0]))})

            action = 0 if np.random.rand() < probs[0][0] else 1

            action_arr = np.zeros(2)
            action_arr[action] = 1.0

            next_obs, reward, done, _ = env.step(action)

            states.append(obs)
            actions.append(action_arr)
            transitions.append((next_obs, reward))

            total_reward += reward
            obs = next_obs
            if done:
                break
    total_reward /= numofep

    future_rewards = []
    for idx, trans in enumerate(transitions):
        cum_reward = 0.0
        for idx2, future_trans in enumerate(transitions[idx:]):
            _, future_reward = future_trans
            cum_reward += (gamma ** idx2) * future_reward
        future_rewards.append([cum_reward])

    obs_ = [obs for obs, _ in transitions]
    estimated_val = sess.run(value_val_est, feed_dict={value_state_var: obs_})
    advantage = np.array(future_rewards) - estimated_val

    pol_opt, pol_loss = sess.run([policy_opt, policy_loss], feed_dict={policy_state_var: states,
                                                                       policy_action_var: actions,
                                                                       policy_advantages_var: advantage
                                                                       })

    val_opt, val_loss = sess.run([value_opt, value_loss],
                                 feed_dict={value_state_var: states, value_val_var: future_rewards})

    pol_loss_hist.append(pol_loss)
    val_loss_hist.append(val_loss)
    reward_hist.append(total_reward)
    reward100 = sum(reward_hist[-100:]) / 100.0
    if epoch % 10 == 0:
        print epoch, reward100

    if reward100 > 195:
        for i in reward_hist[-100:]:
            print i
        break