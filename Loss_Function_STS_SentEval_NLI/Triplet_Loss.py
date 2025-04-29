import torch
import torch.nn.functional as F


def triplet_loss(anchors, positives, negatives, margin=1.0):
    if anchors is None or positives is None or negatives is None:
        return 0.0

    # Euclidean distance between anchor and positive, and anchor and negative...
    pos_dist = F.pairwise_distance(anchors, positives)
    neg_dist = F.pairwise_distance(anchors, negatives)
    loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
    return loss.mean()