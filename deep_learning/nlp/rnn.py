'''
Reference: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/language_model/main.py
'''

import torch
import time
import argparse
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

from util import Corpus, detach


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.encoder = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, vocab_size)

        self.init_weights()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    
    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)
    
    def forward(self, x, hidden):
        emb = self.encoder(x)
        output, (hidden, c) = self.rnn(emb, hidden)
        output = output.reshape(-1, output.size(2))
        decoded = self.decoder(output)
        return decoded, (hidden, c)
    
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.num_layers, bsz, self.hidden_size)


class ModelTrainer:
    def __init__(
        self,
        num_layers=1,
        batch_size=32,
        seq_length=20,
        embed_size=128,
        hidden_size=128,
        device='cuda'
    ):
        self.device = device
        self.corpus = Corpus()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.model = None
        self.vocab_size = 0
    
    def train(
        self,
        data_path,
        num_epochs=25,
    ):
        num_layers = self.num_layers
        batch_size = self.batch_size
        seq_length = self.seq_length
        embed_size = self.embed_size
        hidden_size = self.hidden_size

        ids = self.corpus.get_data(data_path, batch_size)
        self.vocab_size = len(self.corpus.dictionary)
        num_batchs = ids.size(1) // seq_length

        self.model = RNNModel(self.vocab_size, embed_size, hidden_size, num_layers).to(self.device)

        # loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)


        self.model.train()

        for t in range(num_epochs):
            start = time.time()
            # initialize state
            states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                        torch.zeros(num_layers, batch_size, hidden_size).to(device))

            for i in range(0, ids.size(1) - seq_length, seq_length):
                inputs = ids[:, i:i+seq_length].to(self.device)
                targets = ids[:, (i+1):(i+seq_length+1)].to(self.device)
                
                # detach states in order to do inference
                states = detach(states)
                outputs, states = self.model(inputs, states)
                loss = criterion(outputs, targets.reshape(-1))

                # backprop
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 5)
                optimizer.step()

                step = (i + 1) // seq_length
                if step % 100 == 0:
                    print(f"Epoch: {t+1}/{num_epochs}, Step: {step}/{num_batchs}, Loss: {loss.item()}")
            end = time.time()
            print(f'Time elapsed for Epoch [{t+1}/{num_epochs}]: {end - start}s')
    
    def test(self, data_path, num_samples=1000):
        num_layers = self.num_layers
        hidden_size = self.hidden_size
        with torch.no_grad():
            with open(data_path, 'w') as f:
                state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                        torch.zeros(num_layers, 1, hidden_size).to(device))

                prob = torch.ones(self.vocab_size)
                inputs = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(self.device)

                for i in range(num_samples):
                    output, state = self.model(inputs, state)

                    prob = output.exp()
                    word_id = torch.multinomial(prob, num_samples=1).item()
                    inputs.fill_(word_id)

                    word = self.corpus.dictionary.idx2word[word_id]
                    # parse if meeting end of sentence sign
                    word = word.decode('utf-8')
                    word = '\n' if word == '<eos>' else word + ' '

                    f.write(word)

                    if (i+1) % 100 == 0:
                        print(f"sampled {i+1}/{num_samples} words...")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', required=True, help='training data file path')
    parser.add_argument('--num_epochs', default=10, help='number of traning epochs')
    parser.add_argument('--test_path', default='output.txt', help='testing output file path, optional')
    parser.add_argument('--eval_mode', action='store_true', help='run this in evaluation mode')
    args = parser.parse_args()

    try:
        num_epochs = int(args.num_epochs)
    except TypeError:
        raise Exception('num_epochs argument must be an integer')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device} as the training device...')
    net = ModelTrainer(device=device)
    net.train(args.train_path, num_epochs)
    if args.eval_mode:
        net.test(args.test_path)
        print(f"testing finished, saved to {args.test_path}")
    else:
        pass
            




