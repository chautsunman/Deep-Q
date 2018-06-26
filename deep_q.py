import gym
import tensorflow as tf
import numpy as np

ENV_ID = "CartPole-v0"
HIDDEN_UNITS = [64]
LEARNING_RATE = 1e-3
REPLAY_MEMORY_SIZE = 50000
T = 100000
INITIAL_EXPLORATION = 1
FINAL_EXPLORATION = 0.02
FINAL_EXPLORATION_FRAME = 10000
MINIBATCH_SIZE = 32
DISCOUNT_FACTOR = 0.99
UPDATE_TARGET_NETWORK_FREQUENCY = 500

# Q function
def model(state, actions):
    y = state
    for hidden_unit in HIDDEN_UNITS:
        y = tf.layers.dense(y, hidden_unit, activation=tf.nn.relu)
    y = tf.layers.dense(y, actions)
    return y

# build the graph
def build_graph(env):
    with tf.variable_scope("deep_q"):
        # placeholders
        state_ph = tf.placeholder(tf.float32, shape=[None] + list(env.observation_space.shape))
        action_ph = tf.placeholder(tf.int32, shape=[None])
        reward_ph = tf.placeholder(tf.float32, shape=[None])
        next_state_ph = tf.placeholder(tf.float32, shape=[None] + list(env.observation_space.shape))
        done_ph = tf.placeholder(tf.float32, shape=[None])

        # Q
        with tf.variable_scope("q"):
            q = model(state_ph, env.action_space.n)
            q_argmax = tf.argmax(q, axis=1)
        q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="deep_q/q")

        # Q of next state
        with tf.variable_scope("target_q"):
            q_next_state = model(next_state_ph, env.action_space.n)
        target_q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="deep_q/target_q")

        # Q of the selected action
        q_selected_action = tf.reduce_sum(q * tf.one_hot(action_ph, env.action_space.n), axis=1)

        # target Q of the selected action
        q_target = reward_ph + (1 - done_ph) * DISCOUNT_FACTOR * tf.reduce_max(q_next_state, axis=1)

        # loss
        loss = tf.reduce_mean(tf.square(q_target - q_selected_action))

        # optimizer
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

        # training op
        train = optimizer.minimize(loss, var_list=q_vars)

        # update target Q function op
        update_target_q_ops = []
        for target_q_var, q_var in zip(sorted(target_q_vars, key=lambda var: var.name), sorted(q_vars, key=lambda var: var.name)):
            update_target_q_ops.append(target_q_var.assign(q_var))
        update_target_q = tf.group(*update_target_q_ops)

    return state_ph, action_ph, reward_ph, next_state_ph, done_ph, q, q_argmax, loss, train, update_target_q

# replay memory
class ReplayMemory:
    def __init__(self, n):
        self.d = []
        self.n = n
        self.next_idx = 0

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.d) < self.n:
            self.d.append(data)
        else:
            self.d[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.n

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idxs = np.random.randint(len(self.d), size=batch_size)
        for idx in idxs:
            state, action, reward, next_state, done = self.d[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

# get the exploration at time t
def get_exploration(t):
    if t < FINAL_EXPLORATION_FRAME:
        return INITIAL_EXPLORATION + (FINAL_EXPLORATION - INITIAL_EXPLORATION) / FINAL_EXPLORATION_FRAME * t
    else:
        return FINAL_EXPLORATION

def main():
    # environment
    env = gym.make(ENV_ID)

    # agent and graph
    state_ph, action_ph, reward_ph, next_state_ph, done_ph, q, q_argmax, loss, train, update_target_q = build_graph(env)

    # replay memory
    replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE)

    observation = env.reset()
    state = np.array([observation])

    # deep Q algorithm
    with tf.Session() as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())

        # initialize the target Q function with the same weights as the agent
        sess.run(update_target_q)

        for t in range(T):
            print(t)

            # select an action
            if np.random.rand() < get_exploration(t):
                action = env.action_space.sample()
            else:
                action = sess.run(q_argmax, feed_dict={state_ph: state})
                action = action[0]

            # execute the action
            next_observation, reward, done, _ = env.step(action)

            # store the transition
            replay_memory.add(observation, action, reward, next_observation, done)

            if not done:
                observation = next_observation
            else:
                observation = env.reset()
            state = np.array([observation])

            # sample random minibatch
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_memory.sample(MINIBATCH_SIZE)

            # train the agent
            sess.run(train, feed_dict={
                state_ph: state_batch,
                action_ph: action_batch,
                reward_ph: reward_batch,
                next_state_ph: next_state_batch,
                done_ph: done_batch
            })

            # set the weights of the target Q function as the same as the agent every UPDATE_TARGET_NETWORK_FREQUENCY steps
            if (t + 1) % UPDATE_TARGET_NETWORK_FREQUENCY == 0:
                sess.run(update_target_q)

        # play
        for game_idx in range(1, 10 + 1):
            observation = env.reset()
            env.render()
            game_reward = 0
            state = np.array([observation])
            done = False

            while not done:
                action = sess.run(q_argmax, feed_dict={state_ph: state})
                action = action[0]
                next_observation, reward, done, _ = env.step(action)
                env.render()
                game_reward += reward
                state = np.array([next_observation])

            print("Game {} reward: {}".format(game_idx, game_reward))

    env.close()

if __name__ == "__main__":
    main()
