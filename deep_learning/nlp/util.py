'''
A set of tools for performing nlp related tasks.

Refrence: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/language_model/data_utils.py
'''

import torch


class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __len__(self):
        return len(self.word2idx)


class Corpus:
    def __init__(self):
        self.dictionary = Dictionary()
    
    def get_data(self, path, batch_size=32):
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                # add ending symbol for learning where to terminate
                words = line.split() + ['<eos>']
                tokens += len(words)

                for word in words:
                    self.dictionary.add_word(word)
        
            ids = torch.LongTensor(tokens)
            cur_token = 0

            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[cur_token] = self.dictionary.word2idx[word]
                    cur_token += 1
        
        num_batchs = ids.size(0)
        ids = ids[:num_batchs * batch_size]

        return ids.view(batch_size, -1)
