import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions

class DSF(nn.Module):

    def __init__(self, hidden_size, num_features):
        """ 
        No bias as function needs to be normalized i.e. f(\emptyset) = 0
        """
        super(DSF, self).__init__()
        self.hidden2repr = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.fc1 = nn.Linear(hidden_size, num_features, bias=False)
        init.uniform_(self.fc1.weight,0,1)
        self.fc2 = nn.Linear(num_features, 1, bias=False)
        init.uniform_(self.fc2.weight,0,1)

    def forward(self, x_set):
        """
        input:
        x_set: (|S| X H)-dim FloatTensor
        """
        x = F.relu(self.hidden2repr(x_set)).sum(0)
        x = torch.log(x+1)
        x = torch.log(self.fc1(x) + 1)
        x = self.fc2(x)
        return x

    def set_forward(self, set_x):
        """
        K X H, compute gain for all of them
        """
        x = F.relu(self.hidden2repr(set_x))
        x = torch.log(x+1)
        x = torch.log(self.fc1(x) + 1)
        x = self.fc2(x)
        return x

    def project(self):
        self.fc1.weight.data.clamp_(0)
        self.fc2.weight.data.clamp_(0)
