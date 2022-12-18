import torch.nn as nn


# A three-layer fully connected neural network
class MLP(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        """

        :rtype: object
        """
        super(MLP, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.Sigmoid())
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.Sigmoid())
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
