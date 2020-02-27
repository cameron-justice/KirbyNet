#!/usr/bin/env python3

import numpy as np

class ReplayMemory:
    def __init__(self, memory_size, frame_width, frame_height, action_size):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.action_size = action_size
        self.size = 0
        self.maxsize = memory_size
        self.current_index = 0
        self.current_state = np.zeros([memory_size, frame_width, frame_height])
        self.action = [0]*memory_size
        self.reward = np.zeros([memory_size])
        self.next_state = np.zeros([memory_size, frame_width, frame_height])
        self.done = [False]*memory_size

    def remember(self, current_state, action, reward, next_state, done):
        self.current_state[self.current_index, :, :] = current_state
        self.action[self.current_index] = action
        self.reward[self.current_index] = reward
        self.next_state[self.current_index, :, :] = next_state.reshape(1, next_state.shape[0], next_state.shape[1])
        self.done[self.current_index] = done
        self.current_index = (self.current_index+1) % self.maxsize
        self.size = max(self.current_index, self.size)

    def replay(self, model, target_model, num_samples, sample_size, gamma):
        if self.size < sample_size:
            return

        for i in range(num_samples):
            current_sample = np.random.choice(self.size, sample_size, replace=False)

            current_state = self.current_state[current_sample, :, :]
            action = [self.action[j] for j in current_sample]
            reward = self.reward[current_sample]
            next_state = self.next_state[current_sample, :, :]
            done = [self.done[j] for j in current_sample]

            model_targets = model.predict(current_state)

            targets = reward + gamma * np.amax(target_model.predict(next_state))
            targets[done] = reward[done]

            model_targets[range(sample_size), action] = targets

            model.fit(current_state, model_targets, epochs=1, verbose=0, batch_size=sample_size)

        target_model.model.set_weights(model.model.get_weights())
