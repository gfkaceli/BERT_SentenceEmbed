# losses/hard_triplet.py

import torch
import torch.nn.functional as F

def compute_euclidean_distance(a: torch.Tensor,
                               b: torch.Tensor
                              ) -> torch.Tensor:
    """ Component from Hard_Triplet_Loss.py"""
    return torch.sqrt(torch.sum((a - b) ** 2, dim=-1) + 1e-16)

def _pairwise_distances(embeddings: torch.Tensor,
                        squared: bool = False
                       ) -> torch.Tensor:
    """ Component from HardTripletLoss.py """
    dot = embeddings @ embeddings.t()
    norm = torch.diag(dot)
    dists = norm.unsqueeze(0) - 2 * dot + norm.unsqueeze(1)
    dists.clamp_min_(0)
    if not squared:
        mask = dists.eq(0).float()
        dists = dists + mask * 1e-16
        dists = (1.0 - mask) * torch.sqrt(dists)
    return dists

def _get_anchor_positive_mask(labels: torch.Tensor) -> torch.Tensor:
    idx_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    idx_neq = ~torch.eye(labels.size(0), device=labels.device).bool()
    return idx_eq & idx_neq

def _get_anchor_negative_mask(labels: torch.Tensor) -> torch.Tensor:
    return ~(labels.unsqueeze(0) == labels.unsqueeze(1))

def hard_triplet_loss(
    embeddings: torch.Tensor = None,
    labels: torch.Tensor = None,
    anchor: torch.Tensor = None,
    positive: torch.Tensor = None,
    negative: torch.Tensor = None,
    margin: float = 1.0,
    batch_hard: bool = False
) -> torch.Tensor:
    """
    Unified hard‐triplet loss.

    If batch_hard=True:
      - `embeddings`: (N,D)
      - `labels`:     (N,)
      Mines hardest positive/negative within the batch (SBERT style).

    If batch_hard=False:
      - `anchor`, `positive`, `negative`: each (M,D)
      Applies pointwise hardest‐pair variant (max positive dist, min negative dist).
    """
    if batch_hard:
        if embeddings is None or labels is None:
            raise ValueError("Must pass embeddings+labels when batch_hard=True")
        # compute full pairwise distances
        pdist = _pairwise_distances(embeddings, squared=False)

        # hardest positives: for each row, mask non-positives then take max
        pos_mask = _get_anchor_positive_mask(labels).float()
        pos_dists = pos_mask * pdist
        hardest_pos, _ = pos_dists.max(dim=1, keepdim=True)

        # hardest negatives: mask non-negatives by adding large constant, then take min
        neg_mask = _get_anchor_negative_mask(labels).float()
        max_dist, _ = pdist.max(dim=1, keepdim=True)
        masked_neg = pdist + max_dist * (1.0 - neg_mask)
        hardest_neg, _ = masked_neg.min(dim=1, keepdim=True)

        loss_matrix = hardest_pos - hardest_neg + margin
        loss = F.relu(loss_matrix).mean()
        return loss

    else:
        if anchor is None or positive is None or negative is None:
            raise ValueError("Must pass anchor, positive, negative when batch_hard=False")
        # compute per-sample distances
        pos_d = compute_euclidean_distance(anchor, positive)
        neg_d = compute_euclidean_distance(anchor, negative)
        hp = pos_d.max()    # hardest positive distance
        hn = neg_d.min()    # hardest negative distance
        tl = torch.clamp(hp - hn + margin, min=0.0)
        return tl.mean()