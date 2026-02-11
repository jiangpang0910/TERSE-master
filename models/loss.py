import torch
import torch.nn as nn


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, device, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.device = device

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).to(self.device).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()

        return loss

class RegressorLoss(nn.Module):
    def __init__(self):
        super(RegressorLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, inputs, targets):
        return self.mse_loss(inputs, targets)

def EntropyLoss(input_):
    mask = input_.ge(0.0000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = - (torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))

def EntropyLoss_single (input_):
    mask = input_.ge(0.0000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = - ((mask_out * torch.log(mask_out)))
    return entropy