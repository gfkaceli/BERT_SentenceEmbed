import torch
import torch.nn.functional as F

def _pairwise_distances(embeddings, squared=False):
    """Computes the 2D matrix of distances between all the embeddings.
    """
    dot_product = torch.matmul(embeddings, embeddings.t())
    square_norm = torch.diag(dot_product)

    # Computing the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    distances[distances < 0] = 0

    if not squared:
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16
        distances = (1.0 -mask) * torch.sqrt(distances)
    return distances

def _get_anchor_positive_triplet_mask(labels):
    """Returns a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    """
    # Checking that i and j are distinct...
    indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
    indices_not_equal = ~indices_equal

    # Checking if labels[i] == labels[j]...
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    return labels_equal & indices_not_equal

def _get_anchor_negative_triplet_mask(labels):
    """Returns a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    """
    # Check if labels[i] != labels[k]
    return ~(labels.unsqueeze(0) == labels.unsqueeze(1))

def hard_triplet_loss(embeddings, labels, margin=1.0):
    # Calculating pairwise distance matrix - SBERT
    pairwise_dist = _pairwise_distances(embeddings, squared=False)

    # Mask to get the hardest positive distances - SBERT
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()
    anchor_positive_dist = mask_anchor_positive * pairwise_dist
    hardest_positive_dist, _ = anchor_positive_dist.max(dim=1, keepdim=True)

    # Mask to get the hardest negative distances - SBERT
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()
    max_anchor_negative_dist, _ = pairwise_dist.max(dim=1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
    hardest_negative_dist, _ = anchor_negative_dist.min(dim=1, keepdim=True)

    tl = hardest_positive_dist - hardest_negative_dist + margin
    # Ensuring non-negative loss
    tl = F.relu(tl)
    triplet_loss = tl.mean()

    return triplet_loss