import numpy as np
import torch

def sequence_nll(scores, targets, mask):
    """ negative log-likelihood for sequences.
    INPUTS:
    scores: (N,V,T)-dim FloatTensor
    targets: (N,T)-dim LongTensor
    mask: (N,V,T)-dim LongTensor
    normalized: (boolean) If True, loss is normalized by (NXT)
    OUTPUTS:
    ce_loss_each: (N,)-dim FloatTensor, -\log_t P(y_t|y<=t), div by T if normalized
    ce_loss: \sum_{n\in N}ce_loss_each, div by N if normalized
    """
    batch_size = mask.size(1)
    losses_flat = -torch.gather(scores, dim=1, index=targets.view(-1,1))
    #  print(losses_flat.size(), mask.size())
    losses = losses_flat.view(*mask.size()) * mask
    ce_loss_each = losses.sum(0) / mask.sum(0)
    ce_loss = (losses*mask).sum() / mask.sum()
    return ce_loss, ce_loss_each
