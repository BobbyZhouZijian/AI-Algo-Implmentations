import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, batch_size, n_inputs, n_neurons):
        super(RNN, self).__init__()

        self.rnn = nn.RNNCell(n_inputs, n_neurons)
        self.hx = torch.randn(batch_size, n_neurons)

    def forward(self, x):
        output = []

        for i in range(len(x)):
            self.hx = self.rnn(x[i], self.hx)
            output.append(self.hx)

        return output, self.hx


# test out
batch_size = 4
n_inputs = 3
n_neurons = 5

X_batch = torch.tensor([[[0, 1, 2], [3, 4, 5],
                         [6, 7, 8], [9, 0, 1]],
                        [[9, 8, 7], [0, 0, 0],
                         [6, 5, 4], [3, 2, 1]]
                        ], dtype=torch.float)

model = RNN(batch_size, n_inputs, n_neurons)

output_val, state_val = model(X_batch)

print(output_val)
print(state_val)
