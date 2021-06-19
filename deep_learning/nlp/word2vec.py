import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

sample_corpus = [
    'he is a king',
    'she is a queen',
    'she is a man',
    'she is a woman',
    'warsaw is poland capital',
    'berlin is germany cpital',
    'paris is france capital',
]


def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens


def get_vocabulary(corpus, tokenize_corpus):
    vocabulary = []
    for sentence in tokenized_corpus:
        for token in sentence:
            if token not in vocabulary:
                vocabulary.append(token)
    return vocabulary


def get_idx(vocabulary, tokenized_corpus):
    word2idx = {w:idx for (idx, w) in enumerate(vocabulary)}
    idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}
    vocabulary_size = len(vocabulary)
    window_size = 2
    idx_pairs = []

    for sentence in tokenized_corpus:
        indices = [word2idx[word] for word in sentence]

        for center_pos in range(len(indices)):
            for w in range(-window_size, window_size+1):
                context_pos = center_pos + w
                if context_pos < 0 or context_pos >= len(indices) or center_pos == context_pos:
                    continue
                idx_pairs.append((indices[center_pos], indices[context_pos]))
    return word2idx, idx2word, idx_pairs


def train(vocabulary, idx_pairs, num_epochs=200, learning_rate=0.005):
    vocabulary_size = len(vocabulary)
    embedding_dims = vocabulary_size // 3

    w1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
    w2 = Variable(torch.randn(vocabulary_size, embedding_dims).float(), requires_grad=True)

    optimizer = optim.Adam([w1, w2], lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for data, target in idx_pairs:
            optimizer.zero_grad()
            x = torch.zeros(vocabulary_size).float()
            x[data] = 1.0
            z = w2 @ w1 @ x
            y = torch.Tensor([target]).long()
            loss = criterion(z.unsqueeze(0), y)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, loss: {loss.item()}')

    print(f"training done. returning the embedding.")
    return w2


# test out
def similarity(x, y):
    return abs(torch.dot(x, y) / (torch.norm(x) * torch.norm(y)))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path', default='', help='file path for the corpus.')
    args = parser.parse_args()

    if args.corpus_path == '':
        corpus = sample_corpus
    else:
        corpus = []
        with open(args.corpus_path, 'r') as f:
            for line in f:
                corpus.append(line.strip())
    tokenized_corpus = tokenize_corpus(corpus)
    vocab = get_vocabulary(corpus, tokenized_corpus)
    word2idx, idx2word, idx_pairs = get_idx(vocab, tokenized_corpus)
    embedding = train(vocab, idx_pairs)

    # test
    print("testing out the embedding:")
    token1 = idx2word[0]
    token2 = idx2word[1]
    token3 = idx2word[2]
    print(f"similarity between '{token1}' and '{token2}': {similarity(embedding[word2idx[token1]], embedding[word2idx[token2]])}")
    print(f"similarity between '{token1}' and '{token3}': {similarity(embedding[word2idx[token1]], embedding[word2idx[token3]])}")