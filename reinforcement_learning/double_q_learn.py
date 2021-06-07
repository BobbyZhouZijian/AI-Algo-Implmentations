"""
Double Q Learning algorithm builds on the idea
of Q learning. It addersses the problem of positive
bias when taking the maximum of the next state-action pair.

Positive bias happens when we use TD (temporal difference) to
obtain an estimation of the return based on the estimated return
from next-step states.

While estimating an estimation does not break convergence, it does
give rise to a problem where uncertainty in the estimation can be
amplified when maximum is taken, espicially positive bias.

To alleviate this issue, we use two q tables and update them based on
the other's next-state estimation, with equal probability (i.e. 0.5 for each
in one episode)

Formula:
    Q1(S,A) <- Q1(S,A) + alpha * (Reward + gamma * max(Q2(S', A')) - Q1(S,A))
    Q2(S,A) <- Q2(S,A) + alpha * (Reward + gamma * max(Q1(S', A')) - Q2(S,A))
"""
from collections import defaultdict
from time import sleep
import numpy as np
import gym
import random
import math


class DoubleQLearn:
    def __init__(self, env=gym.make('Taxi-v3'), alpha=0.1, gamma=0.6, epsilon_high=0.1, epsilon_low=0.0005, decay=200):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_high = epsilon_high
        self.epsilon_low = epsilon_low
        self.decay = decay
        self.env = env
        self.q1_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.q2_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.steps = 0

    def reset_steps(self):
        self.steps = 0

    def epsilon_decay(self):
        return self.epsilon_low + (self.epsilon_high - self.epsilon_low) * math.exp(-self.steps / self.decay)

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
                    action = np.argmax(self.q1_table[state])

                self.steps += 1
                next_state, reward, done, info = env.step(action)

                # equal chance to select q1 and q2
                if random.uniform(0, 1) < 0.5:
                    old_value = self.q1_table[state, action]
                    next_max = np.max(self.q2_table[next_state])

                    alpha = self.alpha_decay(state, action)
                    # alpha = self.alpha
                    new_value = (1 - alpha) * old_value + alpha * (reward + self.gamma * next_max)
                    self.q1_table[state, action] = new_value
                else:
                    old_value = self.q2_table[state, action]
                    next_max = np.max(self.q1_table[next_state])

                    new_value = (1 - alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
                    self.q2_table[state, action] = new_value

                if reward == -10:
                    penalties += 1

                state = next_state
                epochs += 1

            if i % (episodes // 10) == 0:
                print(f'episode: {i}, epochs elapsed: {epochs}, penalties incurred: {penalties}')

    def print_frames(self, frames):
        for i, frame in enumerate(frames):
            print(frame['frame'])
            print(f'Timestamp: {i + 1}')
            print(f"State: {frame['state']}")
            print(f"Action: {frame['action']}")
            print(f"Reward: {frame['reward']}")
            sleep(0.1)

    def act(self):
        env = self.env
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0
        frames = []

        done = False

        while not done:
            action = np.argmax(self.q1_table[state])
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
        self.print_frames(frames)


if __name__ == '__main__':
    dq_learn = DoubleQLearn()
    print('start training...')
    dq_learn.train()
    print(f"Animating performance...")
    dq_learn.act()
