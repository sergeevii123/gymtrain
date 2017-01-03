""""
uses policy/value iteration to solve the task
there's a 2 layer neural network that's responsible for estimating the future value of a state
and a linear weight model (wrapped in a softmax) to handle selecting the policy(acitons)
the policy model incorporates the value function, by adjusting the loss score by the difference between the observed reward and the expected reward
actions that perform much better than expected will have a lower "penalty" and thus have their weights less affected than actions where the
observed performance is much worse.
The other interesting component of this model is the exploration function, which ensures that the agent explores at a rate that's proportional
to the probability dist estimated by the policy function e.g. if there's an 80/20 likelikood of choosing actions 0/1, then in practice the
model(throughout training) will choose action 20% of the time (as opposed to defaulting to the more likely value overtime regardless of the dist).
When training this made a large difference in performance
The model usually takes at least 200 to 250 steps to train to last > 500 steps
the approach is based in large part on the work here: http://kvfrans.com/simple-algoritms-for-solving-cartpole/
I plan on extending to include memory replay and other features associated with more modern models
https://github.com/mkowoods/open-ai-gym/blob/master/cartpole%20-%20policy%20gradient.ipynb
"""

import gym
import numpy as np
import tensorflow as tf

env = gym.make('CartPole-v0')



def policy_gradient():
    """
    updates the policy gradient based on receiveing a state tensor and an action tensor
    """
    state = tf.placeholder(dtype=tf.float32, shape=(None, 4))
    actions = tf.placeholder(dtype=tf.float32, shape=(None, 2))
    advantage = tf.placeholder(dtype=tf.float32, shape=(None, 1))

    linear_weights = tf.Variable(tf.zeros((4, 2)))
    linear = tf.matmul(state, linear_weights)
    est_probs = tf.nn.softmax(linear)

    acc = tf.reduce_sum(tf.mul(est_probs, actions), reduction_indices=[1])
    log_probs = tf.log(acc)
    loss = -tf.reduce_sum(log_probs * advantage)
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

    return optimizer, state, actions, advantage, est_probs, loss, linear_weights


def value_gradient():
    """
    a 2 layer Feed Forward Net to estimate the value as a function of the state vector
    """
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


policy_opt, policy_state_var, policy_action_var, policy_advantages_var, policy_action_est, policy_loss, wts = policy_gradient()
value_opt, value_state_var, value_val_var, value_val_est, value_loss = value_gradient()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

gamma = 0.97

pol_loss_hist = []
val_loss_hist = []
reward_hist = []

for epoch in xrange(1000):
    obs = env.reset()
    states = []
    actions = []
    transitions = []
    total_reward = 0.0
    eps = 1.0 / (epoch + 4.0)

    while True:
        probs = sess.run(policy_action_est, feed_dict={policy_state_var: obs.reshape((1, obs.shape[0]))})

        # THIS EXPLORATION FUNCTION IS THE MOST CRITICAL PART TO THE SUCCESS of the model. WHY???
        action = 0 if np.random.rand() < probs[0][
            0] else 1  # intresting approach.. as you get more confident in your answers the model will start to scale down

        action_arr = np.zeros(2)
        action_arr[action] = 1.0

        next_obs, reward, done, _ = env.step(action)

        states.append(obs)
        actions.append(action_arr)
        transitions.append((next_obs, reward, done))

        total_reward += reward

        obs = next_obs
        if done or total_reward > 500:
            break

    future_rewards = []
    for idx, trans in enumerate(transitions):
        obs, rew, done = trans
        cum_reward = 0.0
        for idx2, future_trans in enumerate(transitions[idx:]):
            _, future_reward, _ = future_trans
            # print (gamma**idx2) * future_reward
            cum_reward += (gamma ** idx2) * future_reward

        future_rewards.append([cum_reward])

    # calculate the difference between the true reward and the estimated value
    # if the policy suggests an action, but the true future reward is worse than the predicted result, then it will increase
    # the loss score, which will pull the policy in the other direction

    obs_ = [obs for obs, _, _ in transitions]
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
    if epoch % 1 == 0:
        print 'episode', epoch, 'total_reward', total_reward, pol_loss, val_loss

    if sum(reward_hist[-100:]) / 100.0 > 250:
        break