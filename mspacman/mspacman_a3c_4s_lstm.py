import threading
import multiprocessing
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
import os
import gym
from gym import wrappers


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class BaseNetwork():
    def __init__(self, a_size, scope, trainer):
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(tf.float32, [None, 84, 84, 3])
            self.conv1 = slim.conv2d(inputs=self.inputs, num_outputs=32,
                                     kernel_size=[5, 5], stride=[1, 1], padding='SAME')

            self.pool1 = slim.max_pool2d(inputs=self.conv1, kernel_size=[2, 2])

            self.conv2 = slim.conv2d(inputs=self.pool1, num_outputs=32,
                                     kernel_size=[5, 5], stride=[1, 1], padding='SAME')

            self.pool2 = slim.max_pool2d(inputs=self.conv2, kernel_size=[2, 2])

            self.conv3 = slim.conv2d(inputs=self.pool2, num_outputs=64,
                                     kernel_size=[4, 4], stride=[1, 1], padding='SAME')

            self.pool3 = slim.max_pool2d(inputs=self.conv3, kernel_size=[2, 2])

            self.conv4 = slim.conv2d(inputs=self.pool3, num_outputs=64,
                                     kernel_size=[2, 2], stride=[1, 1], padding='SAME')

            hidden = slim.fully_connected(slim.flatten(self.conv4), 512)

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(512, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.inputs)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)

            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 512])

            self.policy = slim.fully_connected(rnn_out, a_size,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)
            self.value = slim.fully_connected(rnn_out, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)

            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 10e-6))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs + 10e-6) * self.advantages)

                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))


def process_frame(img):
    img = cv2.resize(img, (84, 84))
    # img = img.mean(-1, keepdims=True)
    img = img / 255.0
    return img

class Worker():
    def __init__(self, name, a_size, trainer, model_path, global_episodes, create_submission):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = BaseNetwork(a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)
        self.env = gym.make("MsPacman-v0")
        self.create_submission = create_submission
        self.n_actions = self.env.action_space.n
        if self.name == 'worker_0':
            self.summary_writer = tf.summary.FileWriter(self.model_path + "train")
        # if create_submission:
        #     self.env = wrappers.Monitor(self.env, '/tmp/skiing-experiment-a3c-2', force=True)


    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        values =  rollout[:, 5]

        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.stack(observations),
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages}

        v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                               self.local_AC.policy_loss,
                                               self.local_AC.entropy,
                                               self.local_AC.grad_norms,
                                               self.local_AC.var_norms,
                                               self.local_AC.apply_grads],
                                              feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        curre = 0
        print "Starting worker " + str(self.number)
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False
                s = self.env.reset()
                s = process_frame(s)
                episode_frames.append(s)
                starta = np.random.randint(0, self.n_actions)
                while d is False:

                    if len(episode_buffer) < 4:
                        v, rnn_state = sess.run([self.local_AC.policy, self.local_AC.value],
                                                feed_dict={self.local_AC.inputs: [s]})
                        a = starta
                    else:
                        a_dist, v = sess.run(
                            [self.local_AC.policy, self.local_AC.value],
                            feed_dict={self.local_AC.inputs: np.stack(np.array(episode_buffer)[-4:, 0])})
                        if len(episode_buffer) % 4 == 0:
                            a = np.random.choice(a_dist[0], p=a_dist[0])
                            a = np.argmax(a_dist == a)

                    s1, r, d, _ = self.env.step(a)
                    s1 = process_frame(s1)
                    if d is False:
                        episode_frames.append(s1)
                    else:
                        s1 = s

                    episode_buffer.append([s, a, r, s1, d, v[0, 0]])
                    episode_values.append(v[0, 0])
                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1
                    # if self.e > endE:
                    #     self.e -= stepDrop

                    if len(episode_buffer) == 2 and d is False and self.create_submission is False:
                        value_estimation = sess.run(self.local_AC.value,
                                                    feed_dict={self.local_AC.inputs: np.stack(np.array(episode_buffer)[-4:, 0])})[0, 0]

                        self.train(episode_buffer, sess, gamma, value_estimation)
                        episode_buffer = episode_buffer[-4:]
                        sess.run(self.update_local_ops)

                    if d is True:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                if self.create_submission:
                    print np.mean(self.episode_rewards[-100:])
                else:
                    # print "worker", str(self.number), "ended episode", curre, "random/norm",random_a, norm_a, "reward", episode_reward
                    print "worker", str(self.number), "ended episode", curre, "reward", episode_reward

                # if len(episode_buffer) != 0 and self.create_submission != True:
                #     v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)

                if episode_count != 0 and episode_count % 10 == 0 and self.name == 'worker_0' and self.create_submission != True:
                    saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                    print "Saved Model"
                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()

                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1
                curre += 1


a_size = gym.make("MsPacman-v0").action_space.n
load_model = False
create_submission = False

model_path = './model'

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

if create_submission:
    num_workers = 1
else:
    num_workers = 1#multiprocessing.cpu_count()

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-6)
    master_network = BaseNetwork(a_size, 'global', None)
    workers = []

    for i in range(num_workers):
        workers.append(Worker(i, a_size, trainer, model_path, global_episodes, create_submission))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        print 'Loading Model...'
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    worker_threads = []
    for worker in workers:
        t = threading.Thread(target=(lambda: worker.work(0.99, sess, coord, saver)))
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)