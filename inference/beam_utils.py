import torch
from copy import deepcopy

def beam_step(log_probs, beam_seq, joint_logprobs, hidden_states, time):
    """
    INPUTS:

    log_probs: (B X V)-dim FloatTensor, log probabilities of the next token(s)
    beam_size: (int) beam budget
    beam_seq  - (B X T)-dim LongTensor containing previous selections 
    joint_logprobs - (B,)-dim tensor containing joint probability of each B 
    hidden_states - (tuple), previous hidden states
    OUTPUTS:
    """
    start_token = 0
    beam_size, max_len = beam_seq.size()
    num_layers,_,hidden_size = hidden_states[0].size()
    _,vocab_size = log_probs.size()

    done_beams = []
    ys, yi = (-log_probs).topk(beam_size, largest=False, dim=1)
    candidates = []
    for beam_idx in range(ys.size()[0]):
        for word_idx in range(ys.size()[1]):
            candidates.append((joint_logprobs[beam_idx] + ys[beam_idx, word_idx], yi[beam_idx, word_idx], beam_idx))
    extensions = sorted(candidates)[:beam_size]

    new_hidden_states = tuple(torch.zeros(num_layers, beam_size, hidden_size) for state in hidden_states)
    beam_seq_copy = torch.LongTensor(beam_seq.size())

    for beam_idx, ext in enumerate(extensions):
        got_end_token = (ext[1] == start_token)
        no_end_token_so_far = (sum(beam_seq[ext[2]][:time] == start_token) == 0)
        end_of_time = (time==max_len-1)
        # if end token has been reached
        is_first_end_token = got_end_token and no_end_token_so_far
        is_max_time_without_end_token = end_of_time and no_end_token_so_far
        if is_first_end_token or is_max_time_without_end_token:
            done_beams.append({'log_probs':ext[0], 'seq': beam_seq[ext[2]][:time]})
            # render this beam useless for next time step
            beam_seq_copy[beam_idx] = beam_seq[ext[2]].clone()
            beam_seq_copy[beam_idx, time] = deepcopy(ext[1])
            joint_logprobs[beam_idx] = 1000
        
        # if not, extend beams as always
        else:
            beam_seq_copy[beam_idx] = beam_seq[ext[2]].clone()
            beam_seq_copy[beam_idx, time] = deepcopy(ext[1])
            joint_logprobs[beam_idx] = ext[0]

        # copy over hidden_states
        for state_idx, state in enumerate(hidden_states):
            new_hidden_states[state_idx][:,beam_idx,:] = hidden_states[state_idx][:,ext[2],:].clone()
    
    return beam_seq_copy, new_hidden_states, done_beams

def beam_cell_step(log_probs, joint_logprobs, beam_size):
    """ 
    provides the switch indexes; useful for using beam search to supervise
    """
    _,vocab_size = log_probs.size()
    ys, yi = (-log_probs).topk(beam_size, largest=False, dim=1)
    candidates = []
    for beam_idx in range(ys.size()[0]):
        for word_idx in range(ys.size()[1]):
            candidates.append((joint_logprobs[beam_idx] + ys[beam_idx, word_idx], yi[beam_idx, word_idx], beam_idx))
    extensions = sorted(candidates)[:beam_size]
    switch_idx = [[ext[1],ext[2]] for ext in extensions]
    return torch.LongTensor(switch_idx)

def add_diversity(log_probs, beam_seq, group, time, div_strength):
    """ 
    input:
    log_probs: (FloatTensor), B' X V
    beam_seq: (dict, beam_idx: FloatTensor, B' X T
    """
    _,vocab_size = log_probs[0].size()
    div = torch.zeros(vocab_size)
    for group_idx in range(group):
        idx_tensor = beam_seq[group_idx][:, time]
        div[idx_tensor] += 1
    if log_probs[group].is_cuda:
        div = div.cuda()
    log_probs[group] = log_probs[group] - div_strength*div
    return log_probs

def update_beam(beam_seq, beam_seq_t, hidden_states, time, start_token=0):
    """ update beams
    """
    _, max_len = beam_seq.size()
    beam_size,_ = beam_seq_t.size()
    switch_idx = []
    switch_idx, token = beam_seq_t[:,0], beam_seq_t[:,1]
    beam_seq = beam_seq[switch_idx]
    beam_seq[:,time] = token
    not_done_beams_idx = torch.arange(beam_size)
    done_beams = []
    # copy over hidden_states
    hidden_states = tuple(state[:,switch_idx,:] for state in hidden_states)
    not_done_beams = torch.arange(beam_size)
    for beam_idx, beam in enumerate(beam_seq):
        got_end_token = (beam[time] == start_token)
        no_end_token_so_far = (sum(beam[:time] == start_token) == 0)
        end_of_time = (time+1==max_len)
        is_first_end_token = got_end_token and no_end_token_so_far
        is_max_time_without_end_token = end_of_time and no_end_token_so_far
        if is_first_end_token or is_max_time_without_end_token:
            done_beams.append(beam[:time])
            hidden_states[0][:,beam_idx,:] *= 0 # 0 hidden states so that gain is 0 in next step and is not further continued
            #  not_done_beams_idx = not_done_beams_idx[not_done_beams_idx!=beam_idx]
    #  beam_seq = beam_seq[not_done_beams_idx]
    #  hidden_states = tuple(state[:,not_done_beams_idx,:] for state in hidden_states)
    return done_beams, beam_seq, hidden_states
