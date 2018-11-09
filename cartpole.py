import gym
import gym.spaces
import tensorflow as tf
import numpy as np
import argparse
import logging
import os
from helper import expected_rewards, PolicyGradient, ValueFunction

parser = argparse.ArgumentParser(description='Policy gradient reinforcement \
                                 learning model for cartpole game')
parser.add_argument('--num_episodes', default=1000, help='Number of episodes \
                    to be sampled during an epoch')
parser.add_argument('--learning_rate', default=0.01, help='Learning rate for \
                    optimizer')
parser.add_argument('--discount_rate', default=0.95, help='Discount rate for \
                    future rewards')
parser.add_argument('--epochs', default=1000, help='Number of epochs to train')
parser.add_argument('--state_size', default=4, help='Number of state values')
parser.add_argument('--action_size', default=2, help='Number of actions')
parser.add_argument('--output_size', default=1, help='Number of output neurons \
                    for value function network')
parser.add_argument('--hidden_state_size', default=16, help='Number of neurons \
                    in fully connected layers')
parser.add_argument('--baseline', default=True, help='Boolean to use baseline method to \
                    reduce variance')
parser.add_argument('--log_dir', default='logs/cartpole/', help='Path to directory for logs for \
                    tensorboard visualization')
parser.add_argument('--run_num', required=True, help='Provide a run number to correctly log')


def main(args):
    # Load game
    env = gym.make("CartPole-v0")
    # Initialize the game
    state = env.reset()

    # Reset tensorflow graph and initialize networks
    tf.reset_default_graph()
    network = PolicyGradient(learning_rate=float(args.learning_rate),
                             state_size=int(args.state_size),
                             action_size=int(args.action_size),
                             hidden_state_size=int(args.hidden_state_size))
    valueNetwork = ValueFunction(learning_rate=float(args.learning_rate),
                                 state_size=int(args.state_size),
                                 action_size=int(args.action_size),
                                 output_size=int(args.output_size),
                                 hidden_state_size=int(args.hidden_state_size))
    saver = tf.train.Saver()

    # Create directory for logs
    if not os.path.exists(os.path.join(args.log_dir, args.run_num)):
        logging.info("Creating directory {0}".format(os.path.join(args.log_dir, args.run_num)))
        os.mkdir(os.path.join(args.log_dir, args.run_num))

    # Start tensorflow
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(os.path.join(args.log_dir, args.run_num), sess.graph)
        sess.run(tf.global_variables_initializer())
        # Initialize list to track progress
        avg_epoch_rewards = []
        for epoch in range(int(args.epochs)):

            # Initialize episode states, actions, rewards, and total rewards
            epoch_states, epoch_actions, epoch_rewards = [], [], []
            total_episode_rewards = []

            for episode in range(int(args.num_episodes)):
                # Initialize episode states, actions, and rewards
                episode_states, episode_actions, episode_rewards = [], [], []
                # Initialize the game for the episode
                state = env.reset()
                # Run the game/episode until it's done
                while True:
                    # Get distribution of actions from softmax
                    feed = {network.inputs_: state.reshape((1, *state.shape))}
                    action_dist = sess.run(network.softmax, feed_dict=feed)

                    # Sample action from distribution
                    action = np.random.choice(range(action_dist.shape[1]),
                                              p=action_dist.flatten())

                    # Create one hot encoding for action for network input
                    one_hot_action_ = np.zeros(int(args.action_size))
                    one_hot_action_[action] = 1

                    # Take action in game
                    next_state, reward, done, _ = env.step(action)

                    # Store the states, reward, and action
                    episode_states.append(state)
                    episode_rewards.append(reward)
                    episode_actions.append(one_hot_action_)

                    state = next_state

                    if done:
                        # Calculate the discounted cummulative reward
                        expected_episode_rewards = expected_rewards(
                            episode_rewards, float(args.discount_rate))

                        # Calculate and store the total episode reward
                        total_episode_reward = sum(episode_rewards)
                        total_episode_rewards.append(total_episode_reward)

                        # Add episode states, actions, and rewards to epoch
                        epoch_states.append(np.vstack(episode_states))
                        epoch_actions.append(np.vstack(episode_actions))
                        epoch_rewards += expected_episode_rewards

                        break

            # After iterationg through all episodes in an epoch, calculate and store the average total reward 
            avg_epoch_reward = np.mean(total_episode_rewards)
            avg_epoch_rewards.append(avg_epoch_reward)
        
            if args.baseline:
                # Get the expected value given a state
                feed = {valueNetwork.inputs_: np.vstack(epoch_states)}
                V_t = sess.run(valueNetwork.fc3, feed_dict=feed)
                # Apply the baseline to the discounted cummulative rewards
                unadjusted_epoch_rewards = epoch_rewards
                epoch_rewards -= np.hstack(V_t)

                # Train the value network
                feed = {valueNetwork.inputs_: np.vstack(epoch_states),
                        valueNetwork.expected_episode_rewards_: unadjusted_epoch_rewards}
                loss_, _ = sess.run([valueNetwork.loss, valueNetwork.train],
                                    feed_dict=feed)
            # Train the policy gradient
            feed = {network.inputs_: np.vstack(epoch_states),
                    network.actions_: np.vstack(epoch_actions),
                    network.expected_episode_rewards_: epoch_rewards, 
                    network.avg_epoch_reward: avg_epoch_reward}
            loss_, _, summary = sess.run([network.loss, network.train, network.summary_op], feed_dict=feed)

            # Log and save models
            logger.info("Epoch: {0}\tAvg Reward: {1}".format(epoch,
                                                             avg_epoch_reward))
            writer.add_summary(summary, epoch)
            if epoch % 100 == 0:
                    saver.save(sess, "./model/model{0}.ckpt".format(epoch))
                    print("Model Saved")


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    args = parser.parse_args()
    main(args)
