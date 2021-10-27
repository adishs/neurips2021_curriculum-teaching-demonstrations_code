import torch
import torch.nn as nn
import torch.nn.functional as F

class policy_network(nn.Module):
    """
    @brief: Neural network to obtain probability distribution over actions (policy).
    """
    def __init__(self, task_embedding_net, action_space=3):
        super(policy_network, self).__init__()
        self.task_embedding_net = task_embedding_net
        self.fc1 = nn.Linear(task_embedding_net.embedding_size, 256)
        self.drop1 = nn.Dropout(p=0.2)
        self.action_out = nn.Linear(256, action_space)


    def forward(self, x):
        x = self.task_embedding_net(x)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = self.action_out(x)
        return x