import math
import string
from collections import defaultdict

import torch
from torch.distributions import Categorical

from .decoder import Decoder
from .beam_utils import update_beam
from .sub_utils import SubsetSelector
from .loss_utils import sequence_nll
from .prepro_utils import decrypt_sequence

class SetDecoder(Decoder):
    """ main inference class for trainable sequence decoding problems
    note: could probably be made to inherit from nn.Module as well
    """

    def __init__(self, decoder_rnn, word_to_idx, beam_size=3, max_len=20):
        """ 
        Inputs:
        decoder_rnn: (nn.LSTM) trained model that gives P(y_t | y_<t)
        selector: submodular selector
        max_len: (int) T, maximum length of decoding
        word_to_idx: (dict: str->int) has key, start token '<s>'
        """
        self.decoder_rnn = decoder_rnn
        self.selector = SubsetSelector(decoder_rnn.embed, decoder_rnn.hidden_size)
        self.max_len = max_len
        self.beam_size = beam_size
        self.word_to_idx = word_to_idx
        # create inverse mapping for vocabulary
        self.idx_to_word = {idx:word for (word,idx) in word_to_idx.items()}
        self.start_token = word_to_idx['<s>']

    def get_likelihood(self, x, y):
        pass

    def selective_beam_search(self, x, beam_size=3):
        """ awesome stuff
        """
        done_beams = []
        loss = []
        beam_seq = torch.zeros(beam_size, self.max_len).long()
        # forward input
        _,hidden_states = self.decoder_rnn.cell_forward(x)
        # forward start token
        next_inputs = torch.LongTensor([self.start_token])
        if x.is_cuda:
            next_inputs = next_inputs.cuda()
        for current_time in range(self.max_len):
            to_zero_idx = (hidden_states[0].squeeze(0).sum(1)!=0).float().unsqueeze(0).t() # which hidden states to set to 0 for selector
            hidden_states = (hidden_states[0]*to_zero_idx, hidden_states[1])
            # select next inputs
            _,_, beam_seq_t, loss_t = self.selector(hidden_states[0].squeeze(0), beam_size, temp=1)
            # update beam_seq with selection and look out for completed beams
            done_beams_t, beam_seq, hidden_states = update_beam(beam_seq, beam_seq_t, hidden_states, current_time)
            done_beams += done_beams_t
            loss.append(loss_t.unsqueeze(0))
            next_inputs = beam_seq_t[:,1]
        loss = torch.cat(loss).sum() / (self.max_len * beam_size)
        done_beams = [self.idxs_to_seq(beam) for beam in done_beams]
        return done_beams, loss

    def load_from_cv(self, decoder_cv, selector_cv):
        self.decoder_rnn.load_state_dict(decoder_cv)
        self.selector.dsf.load_state_dict(selector_cv)
        print("Loaded both Decoder RNN and DSF from cv")

    def save_to_cv(self):
        return self.decoder_rnn.state_dict(), self.selector.dsf.state_dict()

    def cuda(self):
        self.decoder_rnn = self.decoder_rnn.cuda()
        self.selector.dsf = self.selector.dsf.cuda()

    def freeze_decoder(self):
        for params in self.decoder_rnn.parameters():
            params.requires_grad = False
        print("Decoder RNN parameters are now frozen")
