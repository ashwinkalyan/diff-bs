import string
from collections import defaultdict

import torch
from torch.distributions import Categorical

from .loss_utils import sequence_nll
from .prepro_utils import decrypt_sequence

from .beam_utils import beam_step, add_diversity

class Decoder:
    """ main inference class for sequence decoding problems
    """

    def __init__(self, decoder_rnn, max_len, word_to_idx):
        """ 
        Inputs:
        decoder: (nn.LSTM) trained model that gives P(y_t | y_<t)
        max_len: (int) T, maximum length of decoding
        word_to_idx: (dict: str->int) vocabulary, contains start token '<s>'
        decoder has field:
        """
        self.decoder_rnn = decoder_rnn
        self.max_len = max_len
        self.word_to_idx = word_to_idx
        try:
            self.start_token = word_to_idx['<s>']
        except KeyError:
            print("word_to_idx does not contain start token '<s>'")
        # create inverse mapping for vocabulary
        self.idx_to_word = {idx:word for (word,idx) in word_to_idx.items()}

    def cell_forward(self, x, h=None):
        return self.decoder_rnn.cell_forward(x, h)

    def idxs_to_seq(self, idxs, idx_to_word = None):
        idx_to_word = self.idx_to_word if idx_to_word == None else idx_to_word
        return decrypt_sequence(idxs, idx_to_word)

    def seq_to_idxs(self, y):
        return encrypt_sequence(y, self.word_to_idx, self.max_len)

    def get_likelihood(self, x, y):
        """ Given initial input x and sequence y = (y_1, ... y_T), returns P(y|x) = \prod_t P(y_t|y<=t, x)
        INPUT: 
        x: (N, D)-dim FloatTensor
        y: (list of ints), length of list = N
        OUTPUT: 
        log_likelihood: (N,)-dim FloatTensor
        """
        y,mask = self.seq_to_idxs(y) # mask
        out,h = self.decoder_rnn(x, y) # forward
        _,sequence_nll = sequence_nll(out, y, mask, normalized = False)
        return -sequence_nll

    def sample(self, x):
        """
        inputs:
        x: (1,D)-dim FloatTensor) initial input
        """
        # forward input
        _,h = self.cell_forward(x)
        # forward start token
        x = torch.LongTensor([self.start_token])
        out = list()
        for t in range(self.max_len+1):
            out.append(x)
            y,h = self.cell_forward(x,h)
            y_dist = Categorical(y.exp())
            x = y_dist.sample()
            if x == self.start_token:
                break
        return out

    def argmax_search(self, x):
        """
        inputs:
        x: (1,D)-dim FloatTensor) initial input
        addn_func: (function) additional term like length penalty
        """
        # forward input
        _,h = self.cell_forward(x)
        # forward start token
        x = torch.LongTensor([self.start_token])
        out = list()
        for t in range(self.max_len+1):
            out.append(x)
            y,h = self.cell_forward(x,h)
            x = y.argmax().unsqueeze(0)
            if x == self.start_token:
                break
        return out

    def beam_search(self, x, beam_size=3):
        """ 
        """
        # length normalized?
        # additional penalities! 
        # sorted outputs
        done_beams = []
        beam_seq = torch.zeros(beam_size, self.max_len).long()
        joint_logprobs = torch.zeros(beam_size)
        if x.is_cuda:
            beam_seq = beam_seq.cuda()
            joint_logprobs = joint_logprobs.cuda()
        # forward input
        _,hidden_states = self.cell_forward(x)
        # forward start token
        next_inputs = torch.LongTensor([self.start_token])
        if x.is_cuda:
            next_inputs = next_inputs.cuda()
        for current_time in range(self.max_len):
            # forward through RNN to get likelihood and hidden states
            if x.is_cuda:
                next_inputs = next_inputs.cuda()
                hidden_states = tuple(st.cuda() for st in hidden_states)
            log_probs,hidden_states = self.cell_forward(next_inputs,hidden_states)
            # step beam search by depth 1
            beam_seq, hidden_states, done_beams_t = beam_step(log_probs, beam_seq, joint_logprobs, hidden_states, current_time)
            # copy over completed beams
            done_beams += done_beams_t
            # construct next set of inputs
            next_inputs = beam_seq[:,current_time].long()

        for idx,beam in enumerate(done_beams):
            done_beams[idx]['seq'] = self.idxs_to_seq(beam['seq'])

        return done_beams, beam_seq

    def diverse_beam_search(self, x, beam_size=3, num_groups=3, div_strength=0.5):

        """ only supports hamming diversity
        """
        if not beam_size % num_groups == 0:
            num_groups = beam_size
        bdash = int(beam_size / num_groups)
        # forward input
        _,hidden_states = self.cell_forward(x)
        # forward start token
        next_inputs = torch.LongTensor([self.start_token])
        if x.is_cuda:
            next_inputs = next_inputs.cuda()
        log_probs,hidden_states = self.cell_forward(next_inputs,hidden_states)
        # initializations
        done_beams = defaultdict(list)
        dict_hidden_states = defaultdict(tuple)
        dict_beam_seq = dict()
        dict_joint_logprobs = dict()
        dict_log_probs = dict()
        # start procedure
        for group_idx in range(num_groups):
            dict_beam_seq[group_idx] = torch.zeros(bdash, self.max_len).long()
            dict_joint_logprobs[group_idx] = torch.zeros(bdash)
            dict_log_probs[group_idx] = log_probs
            dict_hidden_states[group_idx] = hidden_states
        if x.is_cuda:
            for group_idx in range(num_groups):
                dict_beam_seq[group_idx] = dict_beam_seq[group_idx].cuda()
                dict_joint_logprobs[group_idx] = dict_joint_logprobs[group_idx].cuda()
        for current_time in range(self.max_len):
            for group_idx in range(num_groups):
                # modify logprobs for diversity here
                log_probs = add_diversity(dict_log_probs, dict_beam_seq, group_idx, current_time, div_strength)
                dict_beam_seq[group_idx], dict_hidden_states[group_idx], done_beams_t = beam_step(dict_log_probs[group_idx], dict_beam_seq[group_idx], dict_joint_logprobs[group_idx], dict_hidden_states[group_idx], current_time)
                next_inputs = dict_beam_seq[group_idx][:,current_time]
                if x.is_cuda:
                    next_inputs = next_inputs.cuda()
                    dict_hidden_states[group_idx] = tuple(st.cuda() for st in dict_hidden_states[group_idx])
                dict_log_probs[group_idx], dict_hidden_states[group_idx] = self.cell_forward(next_inputs, dict_hidden_states[group_idx])
                done_beams[group_idx] += done_beams_t

        dbeams = list()
        for _,beam_list in done_beams.items():
            dbeams += [{'log_probs':beam['log_probs'], 'seq':self.idxs_to_seq(beam['seq'])} for beam in beam_list]

        return dbeams
