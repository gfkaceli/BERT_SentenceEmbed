import torch
import numpy as np


def divided_by_maximum(labels):
    return labels / torch.max(labels)


def sigmoid(labels):
    labels = np.array(labels)
    return 1 / (1 + np.exp(-labels))


def norm_function(norm, labels):
    return globals()[norm](labels)
