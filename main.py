import gym
import ppaquette_gym_doom
import tflearn
import tensorflow as tf
import numpy as np
import math
import random

env = gym.make('ppaquette/meta-Doom-v0')
env.reset()

total_score = 0

action_length = len(env.action_space.sample())

def get_action():
    action = env.action_space.sample()
    return action
obs = env.step(get_action())
height = len(obs[0])
width = len(obs[0][0])
channels = len(obs[0][0][0])
data = tf.placeholder(tf.float32, [None, height, width, channels])
reward = tf.placeholder(tf.float32, [None, 1])

def make_viewer(obs):
    eyes = obs
    filters_old = 3
    filters = 6
    n_height = height
    n_width = width
    viewer_vars = []
    while filters <= 216:
        filters *= 6
        eyes_filter = tf.Variable(tf.random_normal([5, 5, filters_old, filters]))
        eyes_bias = tf.Variable(tf.random_normal([filters]))
        viewer_vars.extend([eyes_filter, eyes_bias])
        eyes = tf.nn.conv2d(eyes, eyes_filter, strides=[1, 3, 3, 1], padding='SAME')
        eyes = tf.nn.bias_add(eyes, eyes_bias)
        eyes = tf.nn.elu(eyes)
        eyes = tf.nn.max_pool(eyes, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        filters_old = filters
        n_height = math.ceil(n_height / 6.0)
        n_width = math.ceil(n_width / 6.0)

    fc_n = int(n_height * n_width * filters)
    reshaped_eyes = tf.reshape(eyes, [-1, fc_n])
    return (fc_n, reshaped_eyes, viewer_vars)

lstm_size = 256

def make_agent(viewer):
    (fc_n, viewer, _) = viewer
    agent_brain = tf.contrib.rnn.LSTMBlockCell(lstm_size, use_peephole=True)
    state_in = agent_brain.zero_state(1, dtype=tf.float32)
    state_size = agent_brain.state_size
    (c, h) = (
        tf.placeholder_with_default(state_in.c, (1, state_size.c)),
        tf.placeholder_with_default(state_in.h, (1, state_size.h)))

    state_in = (c, h)
    viewer_weights = tf.Variable(tf.random_normal([fc_n, lstm_size]))
    viewer_biases = tf.Variable(tf.random_normal([lstm_size]))
    output, state_out = agent_brain(tf.matmul(viewer, viewer_weights) + viewer_biases, state_in)
    agent_weights = tf.Variable(tf.random_normal([lstm_size, action_length]))
    agent_biases = tf.Variable(tf.random_normal([action_length]))
    agent_out = tf.nn.elu(tf.matmul(output, agent_weights) + agent_biases)+ tf.ones([action_length])
    agent_out = tf.placeholder_with_default(agent_out, agent_out.get_shape())
    return (agent_out, state_in, state_out, agent_brain, [viewer_weights, viewer_biases, agent_weights, agent_biases])

def make_judge(viewer, agent_actions, real_reward, agent_vars):
    (fc_n, viewer, viewer_vars) = viewer
    judge_brain = tf.contrib.rnn.LSTMBlockCell(lstm_size, use_peephole=True)
    state_in = judge_brain.zero_state(1, dtype=tf.float32)
    state_size = judge_brain.state_size
    (c, h) = (
        tf.placeholder_with_default(state_in.c, (1, state_size.c)),
        tf.placeholder_with_default(state_in.h, (1, state_size.h)))

    state_in = (c, h)
    viewer_weights = tf.Variable(tf.random_normal([fc_n, lstm_size]))
    actions_weights = tf.Variable(tf.random_normal([action_length, fc_n]))
    viewer_biases = tf.Variable(tf.random_normal([lstm_size]))
    actions_biases = tf.Variable(tf.random_normal([fc_n]))
    actions = tf.placeholder_with_default(agent_actions, [1, action_length])
    view_and_actions = tf.matmul(tf.add(tf.nn.l2_normalize(viewer, 1), tf.matmul(actions, actions_weights) + actions_biases), viewer_weights)
    output, state_out = judge_brain(view_and_actions + viewer_biases, state_in)

    judge_weights = tf.Variable(tf.random_normal([lstm_size, 1]))
    judge_biases = tf.Variable(tf.random_normal([1]))
    judge_out = tf.matmul(output, judge_weights) + judge_biases
    judge_train = tf.train.AdamOptimizer()
    judge_train = judge_train.minimize(tf.losses.absolute_difference(real_reward, judge_out))
    judge_train_actions = tf.train.AdamOptimizer()
    # judge_train_actions = judge_train_actions.minimize(-judge_out, var_list=agent_vars.extend(viewer_vars))
    judge_train_actions = judge_train_actions.minimize(-judge_out)

    return (judge_out, actions, state_in, state_out, judge_brain, judge_train, judge_train_actions)

(player, agent, judge) = (None, None, None)

with tf.variable_scope('viewer'):
    player = make_viewer(data)

with tf.variable_scope('agent'):
    agent = make_agent(player)
(agent_out, agent_state_in, agent_state_out, agent_brain, agent_vars) = agent

with tf.variable_scope('judge'):
    judge = make_judge(player, agent_out, reward, agent_vars)
(judge_out, actions, judge_state_in, judge_state_out, judge_brain, judge_train, judge_train_actions) = judge

sess = tf.Session()
sess.run(tf.global_variables_initializer())

score = 0

obs = obs[0]

i = 0

avg_loss = 0
prev_rewards = np.zeros(2)

(judge_c, judge_h) = (np.zeros((1, lstm_size)), np.zeros((1, lstm_size)))
(judge_c_in, judge_h_in) = judge_state_in
(judge_c_out, judge_h_out) = judge_state_out

(agent_c, agent_h) = (np.zeros((1, lstm_size)), np.zeros((1, lstm_size)))
(agent_c_in, agent_h_in) = agent_state_in
(agent_c_out, agent_h_out) = agent_state_out


while True:
    (agent_old_c, agent_old_h) = (agent_c, agent_h)
    agent_action, agent_c, agent_h = sess.run([agent_out, agent_c_out, agent_h_out], feed_dict={data: np.expand_dims(obs, axis=0), agent_c_in: agent_old_c, agent_h_in: agent_old_h})
    action = agent_action[0]

    (judge_old_c, judge_old_h) = (judge_c, judge_h)

    random_action = np.array(env.action_space.sample())
    judge_run_random = sess.run(judge_out, feed_dict={actions: np.expand_dims(random_action, axis=0), data: np.expand_dims(obs, axis=0), judge_c_in: judge_old_c, judge_h_in: judge_old_h})
    agent_action = agent_action
    judge_run_agent = sess.run(judge_out, feed_dict={data: np.expand_dims(obs, axis=0), judge_c_in: judge_old_c, judge_h_in: judge_old_h})


    random_is_better = judge_run_random[0][0] > judge_run_agent[0][0]

    threshold = i < 0

    action = random_action if random_is_better or threshold else agent_action[0]

    if random_is_better or threshold:
        print("Taking random action")
    old_obs = obs
    obs, actual_reward, is_finished, info = env.step(action)

    # Trying to make the agent figure out that sitting around and waiting for things to kill it is a bad strategy
    # Otherwise it seems to do nothing because of that zero loss
    if avg_loss == 0.0 and actual_reward == 0.0:
        actual_reward = random.random() * -50.0
    print("Reward this turn: " + str(actual_reward))

    pred_reward, judge_c, judge_h = sess.run([judge_out, agent_c_out, agent_h_out], feed_dict={actions: np.expand_dims(action, axis=0), data: np.expand_dims(obs, axis=0), judge_c_in: judge_old_c, judge_h_in: judge_old_h})
    prev_rewards = np.append(prev_rewards, actual_reward)
    prev_rewards = np.delete(prev_rewards, 0)
    avg_loss = np.mean(prev_rewards)

    sess.run(judge_train, feed_dict={data: np.expand_dims(old_obs, axis=0), reward: np.array([[avg_loss]]), judge_c_in: judge_old_c, judge_h_in: judge_old_h})

    if (not threshold) and (abs(pred_reward[0][0] - avg_loss) < (random.random() * 12.0)):
        print("Running agent training step")
        sess.run(judge_train_actions, feed_dict={data: np.expand_dims(old_obs, axis=0), judge_c_in: judge_old_c, judge_h_in: judge_old_h, agent_c_in: agent_old_c, agent_h_in: agent_old_h})

    print("prediction loss: " + str(abs(pred_reward[0][0] - avg_loss)))
    if is_finished:
        print("Reset")
        env.reset()
        (judge_c, judge_h) = (np.zeros((1, lstm_size)), np.zeros((1, lstm_size)))
        (agent_c, agent_h) = (np.zeros((1, lstm_size)), np.zeros((1, lstm_size)))
    env.render()
    i += 1
