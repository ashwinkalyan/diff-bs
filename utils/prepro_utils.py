import string
import dill
import json
import gc
import torch
from collections import defaultdict

def process_caption(sent):
    """ Can process any sentence, process_caption for historical reasons. 
    Removes punctuations and converts allwords to lower case
    """
    translator = str.maketrans('', '', string.punctuation)
    return sent.lower().translate(translator).strip().split()

def make_vocab(dataset_path, cap_idx=-1):
    """
    train_captions is a list of lists containing captions
    """
    with open(dataset_path + 'captions_train.json','rb') as f:
        raw_cap = json.load(f)
    if cap_idx >=0:
        train_captions = [[raw_cap[item][cap_idx]] for item in raw_cap.keys()]
    else:
        train_captions = [raw_cap[item] for item in raw_cap.keys()]

    vocab_count = defaultdict(int)
    for item in train_captions:
        for cap in item:
            for word in process_caption(cap):
                vocab_count[word] += 1
    
    vocab = {idx:word for (idx,(word,count)) in enumerate(vocab_count.items()) if count > 4}
    # make vocab indexing continuous
    word_to_idx = {word:idx+1 for (idx,(prev_idx,word)) in enumerate(vocab.items())}
    # add start token
    word_to_idx['<s>'] = 0
    # add unknown token
    word_to_idx['UNK'] = len(word_to_idx)
    # make inverse vocabulary
    idx_to_word = {idx:word for (word,idx) in word_to_idx.items()}
    print('saving vocabulary of size: {}'.format(len(word_to_idx)))
    with open(dataset_path + 'vocab_capidx_' + str(cap_idx) + '.dill','wb') as f:
        dill.dump((word_to_idx, idx_to_word), f)
    return word_to_idx, idx_to_word

def encrypt_sequence(cap, word_to_idx, max_len):
    """
    takes in a string and returns a list of ints along with start, stop tokens with UNK replacement and a mask
    """
    cap = [word_to_idx.get(item,word_to_idx['UNK']) for item in process_caption(cap)]

    return [word_to_idx['<s>']] + cap[0:min(max_len,len(cap))] + [word_to_idx['<s>']]*(max(1,max_len-len(cap)+1)), [0] + [1]*(min(max_len,len(cap))+1) + [0]*(max_len - len(cap))

def decrypt_sequence(cap, idx_to_word):
    """
    takes in a list of ints and returns a string
    """
    return ' '.join([idx_to_word[int(item)] for item in cap])

def encode_target(captions, word_to_idx, max_len):
    target = torch.zeros(len(captions), max_len+2).long()
    mask = torch.zeros(len(captions), max_len+2)
    for idx,cap in enumerate(captions):
        tcap, mcap = encrypt_sequence(cap, word_to_idx, max_len)
        target[idx], mask[idx] = torch.LongTensor(tcap), torch.LongTensor(mcap)
    return target, mask
