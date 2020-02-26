#!/usr/bin/env python3.7

from gym import core, spaces
import retro
import tensorflow as tf
import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt
from utils import grayscale

actions = {
    0: [0,0,0,0,0,0,0,0,1],
    1: [0,0,0,0,0,0,0,1,0],
    2: [0,0,0,0,0,0,0,1,1],
    3: [0,0,0,0,0,1,0,1,1],
    4: [0,0,0,0,0,1,1,0,0]
}

class Kirby():
    def __init__(self, inputShape):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Conv2D(32,  (7,7), padding='same', activation='relu', input_shape=inputShape))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
        self.model.add(tf.keras.layers.Conv2D(32,  (3,3), padding='same', activation='relu'))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
        self.model.add(tf.keras.layers.Conv2D(64,  (3,3), padding='same', activation='relu'))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
        self.model.add(tf.keras.layers.Conv2D(128, (1,1), padding='valid', activation='relu'))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(16, activation='relu'))
        self.model.add(tf.keras.layers.Dense(5, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Nadam(learning_rate=0.002))

        self.model.summary()

    # Uses the image of the screen to determine an action
    def getAction(self, obs):
        y = self.model.predict(obs)
        i = np.argmax(y)
        return actions.get(i)
       
def main():

    env = retro.make(game='KirbysAdventure-Nes')
    obs = env.reset()
    input_space = (obs.shape[0], obs.shape[1], 1)

    kirby = Kirby(input_space)

    play = True
    while play:
        # Chooses a random action from the space
        # obs: observation of the screen after the action
        # rew: reward gained from action
        # done: if done state reached
        # info: debugging info, using this disqualifies official grading
        obs = grayscale(obs)
        action = kirby.getAction(obs.reshape(1, obs.shape[0], obs.shape[1], 1))
        obs, rew, done, info = env.step(action)
        env.render()
        if done:
            play=false
    env.close()


if __name__ == "__main__":
    main()
