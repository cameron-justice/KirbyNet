#!/usr/bin/env python3.7

from gym import core, spaces
import retro
import tensorflow as tf
import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt
from utils import grayscale
from ReplayMemory import ReplayMemory
from collections import deque


class Kirby():
    def __init__(self, inputShape, actionSpace, epsilon):
        self.epsilon = epsilon
        self.action_space = actionSpace
        self.input_shape = inputShape

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Conv1D(32, kernel_size=7, padding='same', activation='relu', input_shape=inputShape))
        self.model.add(tf.keras.layers.Conv1D(16, kernel_size=3, padding='same', activation='relu'))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(5, activation='relu'))
        self.model.add(tf.keras.layers.Dense(actionSpace, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Nadam(learning_rate=0.002))

    # Uses the image of the screen to determine an action
    def getAction(self, state):
        y = self.model.predict(state)
        return np.argmax(y) + 1 if np.random.random() > self.epsilon else np.random.randint(1,11+1)

    def predict(self, states):
        return self.model.predict(states)

    def fit(self, states, targets, epochs, verbose, batch_size):
        self.model.fit(states, targets, epochs=epochs, verbose=verbose, batch_size=batch_size)

def main():

    env = retro.make(game='KirbysAdventure-Nes', use_restricted_actions=retro.Actions.DISCRETE)
    obs = env.reset()
    input_space = (obs.shape[0], obs.shape[1])
    action_space = 11
    env.action_space = spaces.Discrete(11)
    epsilon = 1.0
    gamma = 0.95
    epsilon_decay = 0.99
    epsilon_min = 0.01
    episodes = 1000

    replay_iterations = 100
    replay_sample_size = 128
    times_window = deque(maxlen=100)
    mean_times = deque(maxlen=episodes)

    kirby = Kirby(input_space, action_space, epsilon)
    target_kirby = Kirby(input_space, action_space, epsilon)
    memory = ReplayMemory(10000, obs.shape[0], obs.shape[1], action_space)

    for episode in range(episodes):
        current_state = env.reset()
        for time in range(500):
            current_state = grayscale(current_state)
            current_state = np.ndarray([1, current_state.shape[0], current_state.shape[1]])
            # Get an action from Q
            action = kirby.getAction(current_state)

            # Perform action
            next_state, reward, done, info = env.step(action)

            # Store the experience
            memory.remember(current_state, action-1, reward, grayscale(next_state), done)
            #update observation
            current_state = next_state

            if done:
                break
            # Show gameplay
            if episode > 950:
                env.render()

        # Lower epsilon
        epsilon = epsilon * epsilon_decay if epsilon > epsilon_min else epsilon_min
        times_window.append(time)
        mean_time = np.mean(times_window)
        mean_times.append(mean_time)
        print('\rEpisode %d/%d - time: %d, mean-time: %d, epsilon: %f' % (episode+1, episodes, time, mean_time, epsilon), end='')

        memory.replay(kirby, target_kirby, replay_iterations, replay_sample_size, gamma)

    env.close()


if __name__ == "__main__":
    main()
