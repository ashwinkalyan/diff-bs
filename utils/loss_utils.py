import numpy as np
import torch

def sequence_nll(scores, targets, mask, normalized = False):
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
    batch_size = scores.size(0)
    losses_flat = -torch.gather(scores, dim=1, index=targets.view(-1,1))
    losses = losses_flat.view(*mask.size()) * mask

    ce_loss_each = losses.sum(1)
    ce_loss = ce_loss_each.sum() / batch_size
    if normalized:
        ce_loss_each /= mask.sum(1)
        ce_loss /= (batch_size * mask.sum(1))

    return ce_loss, ce_loss_each

def make_oneHot(num_classes,targets):
    a = np.eye(int(num_classes))
    return torch.Tensor(a[np.array(targets.numpy(),dtype=int)])
