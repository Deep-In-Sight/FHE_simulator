import torch
import torch.nn as nn

class CrossEntropyLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1, dim=-1, reshape=True):
        """https://towardsdatascience.com/what-is-label-smoothing-108debd7ef06"""
        
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim
        self.reshape = reshape

    def forward(self, pred, target):
        K = pred.shape[-1]

        if self.reshape:
            pred = pred.view(-1,K)
            target = target.view(-1)

        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (K - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
