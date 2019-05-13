import numpy as np
#  from prepro_utils import process_caption

class AverageMeter(object):

    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):

    """Computes the precision@k for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

class logger(object):

    """has methods to log a value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.log = []
        self.itr = []

    def update(self, val, itr):
        self.log += [val]
        self.itr += [itr]

    def average(self):
        return float(sum(self.log)) / len(self.log)

class retrieval():

    def __init__(self, scores):
        self.scores = scores
        num_caps = scores.size(1)

    def recall(self,k):
        ys, yi = self.scores.topk(k, dim=1, largest=False)
        return sum(sum(yi==0))

    def recallN(self,k):
        ys, yi = self.scores.topk(k, dim=1, largest=False)
        return (yi<5).float().sum(dim=1).mean()

    def median_rank(self):
        ys, yi = self.scores.sort(dim=1)
        ysx, yix = yi.sort(dim=1)
        return yix[:,0].median()[0][0]

class oracle():

    def __init__(self, score_dict):
        self.score_dict = score_dict

    def __call__(self, metric, k):
        return self.score_dict[metric][0:k].max(axis=0).mean()

def fetch_ngrams(sentence, n=2):
    ngrams = set()
    words = process_caption(sentence)
    for idx in range(len(words)-n+1):
        ngrams.add(' '.join(words[idx:idx+n]))
    return ngrams

def distinct_ngrams(all_sentences, n):
    """
    returns the number of distinct n-grams in a list of sentences
    """
    all_ngrams = set()
    for sent in all_sentences:
        temp_set = fetch_ngrams(sent, n)
        all_ngrams.update(temp_set)

    return len(all_ngrams)
