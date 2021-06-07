"""
Q Learning is based on the Bellman Update Function:
    U(S) = R(S) + gamma * max of E[U(S')]

Since we do not know the actual value of U(S'), we approximate it by max(A) of Q(S, A)
and iteratively update Q(S,A) so it converges to U(S')

Q(S,A) = (1-alpha) * Q(S,A) + alpha * (reward + gamma * max Q(S', A'))

Also, to prevent the agent from stuck in a local maxima, we use the hyper-parameter
epsilon to encourage the agent to explore unseen area.
"""

from time import sleep
import numpy as np
import gym
import random
import math


def print_frames(frames):
    for i, frame in enumerate(frames):
        print(frame['frame'])
        print(f'Timestamp: {i + 1}')
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(0.1)


class Q_Learn:
    def __init__(self, env=gym.make('Taxi-v3'), alpha=0.1, gamma=0.6, epsilon_high=0.1, epsilon_low=0.0005, decay=200):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_high = epsilon_high
        self.epsilon_low = epsilon_low
        self.decay = decay
        self.env = env
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.steps = 0

        # for approximate Q Learning
        self.features = {}
        self.weights = {}

    def reset_steps(self):
        self.steps = 0

    def epsilon_decay(self):
        return self.epsilon_low + (self.epsilon_high - self.epsilon_low) * math.exp(-self.steps / self.decay)

    def get_features(self, state, action):
        # just an example
        self.features['north'] = 1.0
        self.features['south'] = 1.0
        self.features['west'] = 1.0
        self.features['east'] = 1.0

    def get_q_value(self):
        res = 0
        for f in self.features:
            if f not in self.weights:
                self.weights[f] = 1.0
            res += self.features[f] * self.weights[f]
        return res

    def get_q_values(self, state):
        actions = self.env.action_space.n

        res = np.zeros(actions)
        for a in range(actions):
            self.get_features(state, a)
            res[a] = self.get_q_value()
        return res

    def approx_train(self, episodes=10000):
        """
        Training an approximate Q_learning algorithm.
        This approach approximates the Q value by a set of features
        multiplied by their weights. In another word Q = sum of (wi * fi)
        This way we do not really need to update the exact Q values.
        """
        env = self.env
        self.reset_steps()

        for i in range(episodes):
            state = env.reset()

            epochs, penalties, reward = 0, 0, 0
            done = False

            while not done:
                qs = self.get_q_values(state)
                actions = env.action_space.n
                diff = np.zeros(actions)
                epsilon = self.epsilon_decay()

                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()
                else:
                    for a in range(actions):
                        self.get_features(state, a)
                        old_value = self.get_q_value()
                        next_state, reward, done, info = env.step(a)
                        next_max = np.max(self.get_q_values(next_state))
                        diff[a] = next_max - old_value
                    action = np.argmax(diff)

                self.steps += 1
                next_state, reward, done, info = env.step(action)

                for f in self.features:
                    self.weights[f] = self.weights[f] + self.alpha * (reward + self.gamma * diff[action])

                if reward == -10:
                    penalties += 1

                state = next_state
                epochs += 1

            if i % (episodes // 10) == 0:
                print(f'episode: {i}, epochs elapsed: {epochs}, penalties incurred: {penalties}')
                print(f'weights: {self.weights}')

    def train(self, episodes=10000):
        env = self.env
        self.reset_steps()

        for i in range(episodes):
            state = env.reset()

            epochs, penalties, reward = 0, 0, 0
            done = False

            while not done:
                epsilon = self.epsilon_decay()
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state])

                self.steps += 1
                next_state, reward, done, info = env.step(action)

                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])

                new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
                self.q_table[state, action] = new_value

                if reward == -10:
                    penalties += 1

                state = next_state
                epochs += 1

            if i % (episodes // 10) == 0:
                print(f'episode: {i}, epochs elapsed: {epochs}, penalties incurred: {penalties}')

    def act(self, approx=False):
        env = self.env
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0
        frames = []

        done = False

        while not done:
            if not approx:
                action = np.argmax(self.q_table[state])
            else:
                action = np.argmax(self.get_q_values(state))
            state, reward, done, info = env.step(action)

            if reward == -10:
                penalties += 1

            # for animation
            frames.append({
                'frame': env.render(mode='ansi'),
                'state': state,
                'action': action,
                'reward': reward
            })

            epochs += 1

        print(f'epochs elapsed: {epochs}, penalties incurred: {penalties}')
        print_frames(frames)


if __name__ == '__main__':
    q_learn = Q_Learn()
    print('start training...')
    q_learn.train()
    print(f"Animating performance...")
    q_learn.act(approx=False)
