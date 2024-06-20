import torch
from torch import nn


class NCESoftmaxLoss(nn.Module): #@ MOCO
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        #//print(x, x.shape)
        #//exit()
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).type(torch.LongTensor) ## cuda 的时候是这个样子的 label = torch.zeros([bsz]).cuda().long()
        #//print(label, label.shape)
        loss = self.criterion(x, label)
        #//print(loss, loss.shape)
        return loss


class NCESoftmaxLossNS(nn.Module): #@ E2E
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCESoftmaxLossNS, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        # positives on the diagonal
        label = torch.arange(bsz)## .cuda().long()
        loss = self.criterion(x, label)
        return loss
