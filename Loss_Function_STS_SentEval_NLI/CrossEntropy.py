import torch.nn as nn

def cross_entropy_loss(logits, labels):
    return nn.CrossEntropyLoss()(logits, labels)