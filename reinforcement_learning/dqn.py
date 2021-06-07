import gym
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class Agent:
    def __init__(self, params):
        for key, val in params.items():
            setattr(self, key, val)
        self.net = NeuralNet(self.state_space, 128, self.action_space)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.replay = []
        self.steps = 0

    def train(self):
        if (len(self.replay)) < self.batch_size:
            return

        samples = random.sample(self.replay, self.batch_size)
        state, action, reward, next_state = zip(*samples)

        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)

        y = reward + self.gamma * torch.max(self.net(state).detach(), dim=1)[0].view(self.batch_size, -1)
        pred = self.net(state).gather(1, action)

        loss = self.loss_fn(pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.toptimizer.step()

    def epsilon_decay(self):
        return self.epsilon_low + (self.epsilon_high - self.epsilon_low) * (math.exp(-self.steps / self.decay))

    def act(self, cur_state):
        self.steps += 1
        epsilon = self.epsilon_decay()
        if random.uniform(0, 1) < epsilon:
            action = random.randrange(self.action_space)
        else:
            state = torch.tensor(cur_state, dtype=torch.float).view(1, -1)
            action = torch.argmax(self.net(state)).item()
        return action

    def add_to_replay(self, state, action, reward, next_state):
        self.replay.append((state, action, reward, next_state))
        while len(self.replay) == self.capacity:
            self.replay.pop(0)


class DQN:
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        params = {
            'gamma': 0.8,
            'epsilon_high': 0.9,
            'epsilon_low': 0.05,
            'decay': 200,
            'lr': 0.001,
            'capacity': 10000,
            'batch_size': 64,
            'state_space': self.env.observation_space.shape[0],
            'action_space': self.env.action_space.n
        }
        self.agent = Agent(params)
        self.score = []
        self.mean = []

    def train(self, episodes=1000):
        env = self.env
        state = env.reset()
        total_reward = 1

        for _ in range(episodes):
            done = False
            while not done:
                action = self.agent.act(state)
                next_state, reward, done, info = env.step(action)

                if done:
                    reward = -1

                self.agent.add_to_replay(state, action, reward, next_state)
                total_reward += reward
                state = next_state
                self.agent.train()

                self.score.append(total_reward)
                self.mean.append(sum(self.score[-100:]) / 100)

    def plot_result(self):
        plt.figure(figsize=(20, 10))
        plt.clf()

        score = self.score
        mean = self.mean

        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(score)
        plt.plot(mean)
        plt.text(len(score) - 1, score[-1], str(score[-1]))
        plt.text(len(mean) - 1, mean[-1], str(mean[-1]))

    def print_result(self):
        print(f'score of the training is {self.score}')
        print(f'mean of the training is {self.mean}')


if __name__ == '__main__':
    dqn = DQN()
    dqn.train()
    # dqn.plot_result()
    dqn.print_result()
