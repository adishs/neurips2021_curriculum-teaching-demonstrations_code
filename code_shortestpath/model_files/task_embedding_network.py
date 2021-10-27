import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
    @brief: Residual block without normalization layer.
    """
    def __init__(self, kernel_size, in_feats):
        super(ResBlock, self).__init__()
        self.feat_size = in_feats
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(self.feat_size, self.feat_size,
                               kernel_size=self.kernel_size,
                               padding=self.padding)
        self.conv2 = nn.Conv2d(self.feat_size, self.feat_size,
                               kernel_size=self.kernel_size,
                               padding=self.padding)
        self.conv3 = nn.Conv2d(self.feat_size, self.feat_size,
                               kernel_size=self.kernel_size,
                               padding=self.padding)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out += residual
        return F.relu(out)


class task_embedding_net(nn.Module):
    """
    @brief: Convolution network to encode task grids.
    """
    def __init__(self, in_depth, grid_size, task_embedding_size=512):
        super(task_embedding_net, self).__init__()
        self.embedding_size = task_embedding_size

        self.conv1 = nn.Conv2d(in_depth, 32, kernel_size=3, padding=1)
        self.resblock1 = ResBlock(3, 32)
        self.resblock2 = ResBlock(3, 32)
        self.fc1 = nn.Linear(grid_size*grid_size*32, task_embedding_size)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = F.relu(self.fc1(torch.flatten(x, 1)))
        return x
