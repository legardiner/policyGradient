import numpy as np
import tensorflow as tf


def expected_rewards(episode_rewards, discount_rate, normalize=True):
    """Given a list of rewards returns a list of cummulative discounted rewards.

    Args:
        episode_rewards (list): a list of floats representing the reward for
            each trajectory in an episode
        discount_rate (float): a float value greater than 0.1 and less than or
            equal to 1 to discount future rewards
        normalize (boolean): a boolean value to indicate whether to normalize
            the rewards

    Returns:
        discounted_episode_rewards (list): a list of cummulative discounted
            rewards
    """
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * discount_rate + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative

    if normalize:
        mean = np.mean(discounted_episode_rewards)
        std = np.std(discounted_episode_rewards)
        discounted_episode_rewards -= mean
        discounted_episode_rewards /= std

    return discounted_episode_rewards.tolist()


class PolicyGradient():
    """Policy Gradient Neural Network

    The softmax function returns probabilities of each potential action. An
    action is sample from this distribution for input to the neural network.
    As the network trains, the probabilities of the actions changes in order
    to achieve a higher reward.

    The network takes the state, actions, and the final discounted cummulative
    episode rewards

    Args:
        learning_rate (float): learning rate for optimizer
        state_size (int): length of the state vector
        action_size (int): number of possible actions
        hidden_state_size (int): number of neurons for the hidden layers
        name (string): name of Policy Gradient object

    Attributes:
        inputs_ (numpy.ndarray): Numpy array of shape (state_size, )
            containing state information for each trajectory
        actions_ (numpy.ndarray): Numpy array of shape (action_size, )
            containg an one-hot encoding action vector for each trajectory
        expected_episode_rewards (numpy.ndarray): Numpy array of the expected
            cummulative discounted rewards
        softmax: Probability of each action
    """
    def __init__(self, learning_rate=0.001, state_size=4, action_size=2,
                 hidden_state_size=16, name="PolicyGradient"):
        with tf.name_scope(name):
            self.inputs_ = tf.placeholder(
                tf.float32, [None, state_size], name="inputs")
            self.actions_ = tf.placeholder(
                tf.int32, [None, action_size], name="actions")
            self.expected_episode_rewards_ = tf.placeholder(
                tf.float32, [None, ], name="expected_episode_rewards")
            self.fc1 = tf.contrib.layers.fully_connected(
                self.inputs_, hidden_state_size,
                weights_initializer=tf.contrib.layers.xavier_initializer())
            self.fc2 = tf.contrib.layers.fully_connected(
                self.fc1, action_size,
                weights_initializer=tf.contrib.layers.xavier_initializer())
            self.fc3 = tf.contrib.layers.fully_connected(
                self.fc2, action_size,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                activation_fn=None)
        with tf.name_scope("softmax"):
            self.softmax = tf.nn.softmax(self.fc3)
        with tf.name_scope("loss"):
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.actions_, logits=self.fc3)
            self.loss = tf.reduce_mean(
                self.cross_entropy * self.expected_episode_rewards_)
        with tf.name_scope("train"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train = self.optimizer.minimize(self.loss)


class ValueFunction():
    """Value Neural Network

    The final fully connected layer returns the expected value given the state
    as inputs and the cummulative discounted episode rewards as labels. This
    value is used a baseline for the policy gradient network to reduce
    variance and improve learning

    Args:
        learning_rate (float): learning rate for optimizer
        state_size (int): length of the state vector
        value_size (int): number of outputs for the value function
        hidden_state_size (int): number of neurons for the hidden layers
        name (string): name of Value Function object

    Attributes:
        inputs_ (numpy.ndarray): Numpy array of shape (state_size, )
            containing state information for each trajectory
        expected_episode_rewards (numpy.ndarray): Numpy array of the expected
            cummulative discounted rewards
    """
    def __init__(self, learning_rate=0.001, state_size=4, action_size=2,
                 output_size=1, hidden_state_size=16, name="ValueFunction"):
        with tf.name_scope(name):
            self.inputs_ = tf.placeholder(
                tf.float32, [None, state_size], name="inputs")
            self.expected_episode_rewards_ = tf.placeholder(
                tf.float32, [None, ], name="expected_episode_rewards")
            self.fc1 = tf.contrib.layers.fully_connected(
                self.inputs_, hidden_state_size,
                weights_initializer=tf.contrib.layers.xavier_initializer())
            self.fc2 = tf.contrib.layers.fully_connected(
                self.fc1, action_size,
                weights_initializer=tf.contrib.layers.xavier_initializer())
            self.fc3 = tf.contrib.layers.fully_connected(
                self.fc2, output_size,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                activation_fn=None)
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.square(
                self.fc3 - self.expected_episode_rewards_))
        with tf.name_scope("train"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train = self.optimizer.minimize(self.loss)
