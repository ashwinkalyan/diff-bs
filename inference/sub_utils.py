import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from models.dsf import DSF

class SubsetSelector:
    """
    given previous hidden states and vocabulary, 
    construct the input 
    calculate marginal gain, 
    perform soft-greedy and retun subset
    auxillary functions for updating the model with optimizer and projection steps
    """
    def __init__(self, embeddings, hidden_size, num_feature=2):
        self.dsf = DSF(hidden_size, num_feature)
        self.embeddings = embeddings
        self.vocab_size = self.embeddings.num_embeddings
        self.hidden_size = hidden_size

    def __call__(self, hidden_states, beam_size, temp=1, idxes=None, targets=None):
        """
        input:
        hidden_states: (K H)-dim FloatTensor
        embeddings: (V,H)-dim FloatTensor
        beam_size: (int) number of outputs to select
        temp: (int) temperature
        output:
        (VK,2*H)-dim FloatTensor
        """
        input_set, ref_idx = self.construct_inputset(hidden_states, idxes)
        num_input,_ = ref_idx.size()
        A_k = torch.zeros(1, 2*self.hidden_size)
        if hidden_states.is_cuda:
            A_k = A_k.cuda()
        f_A_k = 0
        next_inputs = []
        next_beam_seq = []
        loss = []
        for k in range(beam_size):
            # compute marginal gains
            marginal_gains = self.dsf.set_forward(input_set) - f_A_k
            # sample
            m_dist = Categorical(F.softmax(marginal_gains.squeeze()/temp, dim=0))
            e_k = m_dist.sample() if targets is None else self.find_idx(ref_idx, targets[k])
            # calculate loss
            loss.append(-m_dist.log_prob(e_k).unsqueeze(0))
            # compute value of union set for next round
            A_k = input_set[e_k] + A_k
            f_A_k = self.get_function_value(A_k)
            # store sampled path
            next_inputs.append(input_set[e_k].unsqueeze(0))
            next_beam_seq.append(ref_idx[e_k].unsqueeze(0))
            # modify input set for next step
            cmp_idx = torch.arange(num_input)
            if hidden_states.is_cuda:
                cmp_idx = cmp_idx.cuda()
            input_set = input_set[cmp_idx!=e_k]
            ref_idx = ref_idx[cmp_idx!=e_k]
            num_input -= 1
        next_inputs = torch.cat(next_inputs)
        loss = torch.cat(loss)
        next_beam_seq = torch.cat(next_beam_seq)
        return next_inputs[:,:self.hidden_size], next_inputs[:,self.hidden_size:], next_beam_seq, loss

    def construct_inputset(self, hidden_states, idxes=None):
        """
        input:
        hidden_states: (K H)-dim FloatTensor
        idxes: (V)-dim LongTensor
        output:
        (VK,2*H)-dim FloatTensor
        """
        idxes = torch.arange(self.vocab_size) if idxes is None else idxes
        vocab_size = idxes.numel()
        beam_size,_ = hidden_states.size()
        if hidden_states[0].is_cuda:
            idxes = idxes.cuda()
        embeddings = self.embeddings(idxes)
        x = hidden_states.repeat(1,vocab_size).view(-1,self.hidden_size)
        v = embeddings.repeat(beam_size, 1)
        beam_idxes = torch.arange(beam_size).unsqueeze(1).repeat(1, vocab_size).view(-1).unsqueeze(0)
        token_idxes = idxes.repeat(1, beam_size)
        if idxes.is_cuda:
            beam_idxes = beam_idxes.cuda()
        ref_idx = torch.cat((beam_idxes, token_idxes))
        return torch.cat((x,v),dim=1), ref_idx.transpose_(0,1)

    def get_marginal_gain(self, e, x_set):
        """
        input:
        e: (1,H)-dim FloatTensor
        x_set: (1,H)-dim FloatTensor
        """
        f_xset = self.dsf.forward(x_set)
        f_xset_union_e = self.dsf.forward(x_set+e)
        return f_xset_union_e - f_xset

    def get_function_value(self, x_set):
        return self.dsf.forward(x_set)

    @staticmethod
    def find_idx(tensor, query):
        return ((tensor[:,0]==query[0])*(tensor[:,1]==query[1])).nonzero().squeeze()



