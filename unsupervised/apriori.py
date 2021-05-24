'''
Apriori algorithm is used for mining frequent itemsets
and devising association rules between different features.

In Apriori, two concepts are particularly important:
    1. Support of A: the ratio of an item A in the dataset D
    2. Confidence of (A -> B) = Support([A,B]) / Support([A])  

In each iteration, we scan through the dataset and find all the items
that meet the minimum Support threshold and discard the rest.

We repeat this process until we get rid of all the items
'''

class Apriori:
    def __init__(self):
        self.train_x = None
        self.train_y = None
        self.frozenset = None
    

    def createC1(self, dataset):
        C1 = []
        for item in dataset:
            for key in item:
                if [key] not in C1:
                    C1.append([key])
        C1.sort()
        return map(self.frozenset, C1)


    def train(self):
        #TODO
        return
