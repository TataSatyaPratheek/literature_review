import torch.nn as nn

class ClassicalDQN(nn.Module):
    """Classical Deep Q-Network baseline"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ClassicalDQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)