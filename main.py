import vizdoom as vzd
import tensorflow as tf
import numpy as np
import math
import random

# TODO: Get this from args
DEFAULT_CONFIG = "../ViZDoom/scenarios/deadly_corridor.cfg"

game = vzd.DoomGame()
game.load_config(DEFAULT_CONFIG)

game.set_screen_format(vzd.ScreenFormat.RGB24)
game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

game.set_render_hud(True)
game.set_render_minimal_hud(False)

game.init()

game.new_episode()

buttons = game.get_available_buttons()
action_length = len(buttons)

total_score = 0

obs = game.get_state().screen_buffer
# Hard-coding these for now since we hard-code the screen resolution above anyways
height = 480
width = 640
channels = 3
data = tf.placeholder_with_default(tf.zeros([1, height, width, channels]), [None, height, width, channels])
reward = tf.placeholder_with_default(tf.zeros([1, 1]), [None, 1])

def make_viewer(screen_buf_variable):
    eyes = screen_buf_variable
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

lstm_layers = [512, 256]

def make_agent(viewer):
    (fc_n, viewer, _) = viewer
    layer_defs = []
    state_in = []
    for layer_size in lstm_layers:
        layer_defs.append(tf.contrib.rnn.LSTMBlockCell(layer_size, use_peephole=True))
    agent_brain = tf.contrib.rnn.MultiRNNCell(layer_defs)
    zero_state = agent_brain.zero_state(1, dtype=tf.float32)
    state_size = agent_brain.state_size
    for index, size in enumerate(state_size):
        (c, h) = (
            tf.placeholder_with_default(zero_state[index].c, (1, size.c)),
            tf.placeholder_with_default(zero_state[index].h, (1, size.h)))
        state_in.append((c, h))

    viewer_weights = tf.Variable(tf.random_normal([fc_n, lstm_layers[-1]]))
    viewer_biases = tf.Variable(tf.random_normal([lstm_layers[-1]]))
    output, state_out = agent_brain(tf.matmul(viewer, viewer_weights) + viewer_biases, state_in)
    agent_weights = tf.Variable(tf.random_normal([lstm_layers[-1], action_length]))
    agent_biases = tf.Variable(tf.random_normal([action_length]))
    agent_out = tf.nn.softmax(tf.matmul(output, agent_weights) + agent_biases)
    agent_out = tf.placeholder_with_default(agent_out, agent_out.get_shape())
    return (agent_out, state_in, state_out, [viewer_weights, viewer_biases, agent_weights, agent_biases])

def make_judge(viewer, agent_actions, real_reward, agent_vars):
    (fc_n, viewer, viewer_vars) = viewer
    layer_defs = []
    state_in = []
    for layer_size in lstm_layers:
        layer_defs.append(tf.contrib.rnn.LSTMBlockCell(layer_size, use_peephole=True))
    judge_brain = tf.contrib.rnn.MultiRNNCell(layer_defs)
    zero_state = judge_brain.zero_state(1, dtype=tf.float32)
    state_size = judge_brain.state_size
    for index, size in enumerate(state_size):
        (c, h) = (
            tf.placeholder_with_default(zero_state[index].c, (1, size.c)),
            tf.placeholder_with_default(zero_state[index].h, (1, size.h)))
        state_in.append((c, h))

    viewer_weights = tf.Variable(tf.random_normal([fc_n, lstm_layers[-1]]))
    actions_weights = tf.Variable(tf.random_normal([action_length, fc_n]))
    viewer_biases = tf.Variable(tf.random_normal([lstm_layers[-1]]))
    actions_biases = tf.Variable(tf.random_normal([fc_n]))
    actions = tf.placeholder_with_default(agent_actions, [1, action_length])
    view_and_actions = tf.matmul(tf.add(tf.nn.l2_normalize(viewer, 1), tf.matmul(actions, actions_weights) + actions_biases), viewer_weights)
    output, state_out = judge_brain(view_and_actions + viewer_biases, state_in)

    judge_weights = tf.Variable(tf.random_normal([lstm_layers[-1], 1]))
    judge_biases = tf.Variable(tf.random_normal([1]))
    judge_out = tf.matmul(output, judge_weights) + judge_biases
    judge_train = tf.train.AdamOptimizer()
    judge_train = judge_train.minimize(tf.losses.absolute_difference(real_reward, judge_out))
    judge_train_actions = tf.train.AdamOptimizer()
    # judge_train_actions = judge_train_actions.minimize(-judge_out, var_list=agent_vars.extend(viewer_vars))
    judge_train_actions = judge_train_actions.minimize(-judge_out)

    return (judge_out, actions, state_in, state_out, judge_train, judge_train_actions)

(player, agent, judge) = (None, None, None)

with tf.variable_scope('viewer'):
    player = make_viewer(data)

with tf.variable_scope('agent'):
    agent = make_agent(player)
(agent_out, agent_state_in, agent_state_out, agent_vars) = agent

with tf.variable_scope('judge'):
    judge = make_judge(player, agent_out, reward, agent_vars)
(judge_out, actions, judge_state_in, judge_state_out, judge_train, judge_train_actions) = judge

sess = tf.Session()
sess.run(tf.global_variables_initializer())

score = 0

i = 0

avg_loss = 0
prev_rewards = np.zeros(2)

agent_state_old = agent_state = agent_initial_state = sess.run(agent_state_out)
judge_state_old = judge_state = judge_initial_state = sess.run(judge_state_out)

def unpack_state(dest, state_in, state):
    for i, (c, h) in enumerate(state):
        dest[c] = state_in[i].c
        dest[h] = state_in[i].h
    return dest

while True:
    agent_state_old = agent_state
    agent_fetches = {"agent_out": agent_out, "state_out": agent_state_out}
    agent_feed_dict = {data: np.expand_dims(obs, axis=0)}
    agent_feed_dict = unpack_state(agent_feed_dict, agent_state, agent_state_in)
    agent_vals = sess.run(agent_fetches, feed_dict=agent_feed_dict)
    agent_action = np.eye(action_length)[np.argmax(agent_vals["agent_out"], axis=1)][0]
    agent_state = agent_vals["state_out"]
    action = agent_action[0]

    judge_state_old = judge_state

    random_action = random.choice(np.eye(action_length))
    judge_run_random = sess.run(judge_out, feed_dict=unpack_state({actions: np.expand_dims(random_action, axis=0), data: np.expand_dims(obs, axis=0)}, judge_state, judge_state_in))
    judge_run_agent = sess.run(judge_out, feed_dict=unpack_state(unpack_state({data: np.expand_dims(obs, axis=0)}, judge_state, judge_state_in), agent_state_old, agent_state_in))


    random_is_better = judge_run_random[0][0] > judge_run_agent[0][0] and random.random() < 0.5

    threshold = i < 0

    action = random_action if random_is_better or threshold else agent_action

    if random_is_better or threshold:
        print("Taking random action")
    old_obs = obs
    game.make_action(list(action))
    is_finished = game.is_episode_finished()
    if is_finished:
        print("Reset")
        game.new_episode()
        judge_state = judge_state_old = judge_initial_state
        agent_state = agent_state_old = agent_initial_state
    state = game.get_state()
    obs = state.screen_buffer
    actual_reward = game.get_last_reward()
    if avg_loss == 0 and actual_reward == 0:
        actual_reward = random.random() * -0.5
    print("Reward this turn: " + str(actual_reward))

    judge_vals = sess.run({"pred_reward": judge_out, "state_out": judge_state_out}, feed_dict=unpack_state({actions: np.expand_dims(action, axis=0), data: np.expand_dims(obs, axis=0)}, judge_state_old, judge_state_in))
    pred_reward = judge_vals["pred_reward"]
    judge_state = judge_vals["state_out"]
    prev_rewards = np.append(prev_rewards, actual_reward)
    prev_rewards = np.delete(prev_rewards, 0)
    avg_loss = np.mean(prev_rewards)

    sess.run(judge_train, feed_dict=unpack_state({data: np.expand_dims(old_obs, axis=0), reward: np.array([[avg_loss]])}, judge_state_old, judge_state_in))

    if (not threshold) and (abs(pred_reward[0][0] - avg_loss) < (random.random() * 12.0)):
        print("Running agent training step")
        sess.run(judge_train_actions, feed_dict=unpack_state(unpack_state({data: np.expand_dims(old_obs, axis=0)}, agent_state_old, agent_state_in), judge_state_old, judge_state_in))

    print("prediction loss: " + str(abs(pred_reward[0][0] - avg_loss)))
    i += 1
