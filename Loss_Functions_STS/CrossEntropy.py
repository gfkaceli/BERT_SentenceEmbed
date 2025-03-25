import torch


def cross_entropy_loss(logits, labels):
    return torch.nn.CrossEntropyLoss()(logits, labels)