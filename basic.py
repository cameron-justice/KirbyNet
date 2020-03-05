#!/usr/bin/env python3.7

import os
from gym import core, spaces
import retro
import tensorflow as tf
import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt
from utils import preprocess
from ReplayMemory import ReplayMemory
from collections import deque
import sys

class Kirby():
    def __init__(self, inputShape, actionSpace, epsilon):
        self.epsilon = epsilon
        self.action_space = actionSpace
        self.input_shape = inputShape

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Conv2D(32, kernel_size=(7,7), padding='same', activation='relu', input_shape=inputShape))
        self.model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(10, activation='relu'))
        self.model.add(tf.keras.layers.Dense(actionSpace, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Nadam(learning_rate=0.005))

        self.model.summary()

    # Uses the image of the screen to determine an action
    def getAction(self, state):
        state = state.reshape(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        y = self.model.predict(state)
        return np.argmax(y) + 1 if np.random.random() > self.epsilon else np.random.randint(1,11+1)

    def predict(self, states):
        states = states.reshape(len(states), self.input_shape[0], self.input_shape[1], self.input_shape[2])
        return self.model.predict(states)

    def fit(self, states, targets, epochs, verbose, batch_size):
        states = states.reshape(len(states), self.input_shape[0], self.input_shape[1], self.input_shape[2])
        self.model.fit(states, targets, epochs=epochs, verbose=verbose, batch_size=batch_size)

    def save(self, m_file, w_file):
        #j = self.model.to_json()
        #with open(m_file, "w") as json_file:
            #json_file.write(j)

        w = self.model.save_weights(w_file)

    def load(self, m_file, w_file):
        #file = open(m_file, 'r')
        #lmj = file.read()
        #file.close()

       # self.model = tf.keras.models.model_from_json(ljf)

        if os.path.isfile(w_file):
            self.model.load_weights(w_file)
            print("loaded from disk")

def main():
    env = retro.make("KirbysAdventure-Nes-USA",  use_restricted_actions=retro.Actions.DISCRETE)
    obs = env.reset()
    action_space = 11
    env.action_space = spaces.Discrete(11)
    epsilon = 0.5
    gamma = 0.95
    epsilon_decay = 0.99
    epsilon_min = 0.01
    episodes = int(sys.argv[1])
    episode_length = 3000

    img_width = 80 if not "full" in sys.argv else 224
    img_height = 80 if not "full" in sys.argv else 240
    channels = 1 if not "rgb" in sys.argv else 3
    input_space = (img_width, img_height, channels)

    replay_iterations = 25
    replay_sample_size = 64
    times_window = deque(maxlen=100)
    mean_times = deque(maxlen=episodes)

    kirby = Kirby(input_space, action_space, epsilon)
    target_kirby = Kirby(input_space, action_space, epsilon)
    kirby.load("k_model.json", "k_weights.h5")
    target_kirby.load("t_model.json", "t_weights.h5")

    memory = ReplayMemory(2500, img_width, img_height, action_space)

    if os.path.isfile("epsilon.txt"):
        f = open("epsilon.txt", 'r')
        epsilon = float(f.readline())

    for episode in range(episodes):
        current_state = env.reset()
        for time in range(episode_length):
            current_state = preprocess(current_state, channels, img_width, img_height)
            # Get an action from Q
            action = kirby.getAction(current_state)

            # Perform action
            next_state, reward, done, info = env.step(action)

            print(info)

            if action == 6 or action == 7 or action == 8:
                if action == 7:
                    reward += 10*action
                else:
                    reward += 5*action

            if action == 3 or action == 4 or action == 5:
                reward -= 5*action

            reward -= time / 10

            # Store the experience
            memory.remember(current_state, action-1, reward, preprocess(next_state, channels, img_width, img_height), done)
            #update observation
            current_state = next_state

            if done:
                break
            # Show gameplay
            if "--play" in sys.argv:
                env.render()

        # Lower epsilon
        epsilon = epsilon * epsilon_decay if epsilon > epsilon_min else epsilon_min
        times_window.append(time)
        mean_time = np.mean(times_window)
        mean_times.append(mean_time)
        print('\rEpisode %d/%d - time: %d, mean-time: %d, epsilon: %f' % (episode+1, episodes, time, mean_time, epsilon), end='')

        memory.replay(kirby, target_kirby, replay_iterations, replay_sample_size, gamma)

    kirby.save("k_model.json", "k_weights.h5")
    target_kirby.save("t_model.json", "t_weights.h5")
    f = open("epsilon.txt", 'w')
    f.write(str(epsilon))
    env.close()


if __name__ == "__main__":
    main()
