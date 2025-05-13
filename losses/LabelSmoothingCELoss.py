import torch
import torch.nn.functional as F

def label_smoothing_cross_entropy_loss(logits, labels, smoothing=0.1):
    confidence = 1.0 - smoothing
    log_probs = F.log_softmax(logits, dim=-1)

    # Initializing true distribution with smoothing value for all classes...
    true_dist = torch.full_like(log_probs, smoothing / (log_probs.size(1) - 1))
    # Setting the true label confidence in the correct class...
    true_dist.scatter_(1, labels.unsqueeze(1), confidence)

    loss = torch.mean(torch.sum(-true_dist * log_probs, dim=-1))
    return loss