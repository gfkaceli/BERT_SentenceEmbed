import torch
import torch.nn.functional as F


def generate_triplets(embeddings, labels):
    anchors, positives, negatives = [], [], []
    for i, anchor_label in enumerate(labels):
        anchor = embeddings[i]

        # Finding a positive example (same label as anchor)...
        positive_indices = (labels == anchor_label).nonzero(as_tuple=True)[0].tolist()
        # Ensuring positive is not the same as anchor...
        positive_indices.remove(i)

        # Ensuring there is at least one valid positive example...
        if not positive_indices:
            continue

        # Selecting the first positive (no randomization for now)...
        positive_idx = positive_indices[0]
        positive = embeddings[positive_idx]

        # Finding a negative example (different label)...
        negative_indices = (labels != anchor_label).nonzero(as_tuple=True)[0].tolist()

        # Ensuring there is at least one valid negative example...
        if not negative_indices:
            continue

        # Select the first negative
        negative_idx = negative_indices[0]
        negative = embeddings[negative_idx]

        # Adding the triplet...
        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)

    # Ensuring non-empty lists before stacking...
    if anchors and positives and negatives:
        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)
    else:
        return None, None, None


def generate_all_triplets(embeddings, labels):
    anchors, positives, negatives = [], [], []
    for i, anchor_label in enumerate(labels):
        anchor = embeddings[i]

        # Find all positive examples (same label as anchor)
        positive_indices = (labels == anchor_label).nonzero(as_tuple=True)[0].tolist()
        positive_indices.remove(i)  # Exclude the anchor itself

        # Find all negative examples (different label from anchor)
        negative_indices = (labels != anchor_label).nonzero(as_tuple=True)[0].tolist()

        # Generate all valid triplets for this anchor
        for pos_idx in positive_indices:
            positive = embeddings[pos_idx]
            for neg_idx in negative_indices:
                negative = embeddings[neg_idx]

                # Add the triplet
                anchors.append(anchor)
                positives.append(positive)
                negatives.append(negative)

    # Convert lists to tensors for batch processing
    if anchors and positives and negatives:
        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)
    else:
        return None, None, None



def triplet_loss(
        embeddings: torch.Tensor = None,
        labels: torch.Tensor = None,
        anchors: torch.Tensor = None,
        positives: torch.Tensor = None,
        negatives: torch.Tensor = None,
        margin: float = 1.0,
        generate_triplets_flag: bool = False
) -> torch.Tensor:
    """
    Unified triplet loss.

    Modes:
      - generate_triplets_flag=True:
          * Inputs: embeddings (N,D), labels (N,)
          * Internally calls generate_triplets → anchors, positives, negatives.
      - generate_triplets_flag=False:
          * Inputs: anchors, positives, negatives directly.

    Returns:
      Scalar mean triplet loss: max(0, d(a,p) - d(a,n) + margin).
    """
    if generate_triplets_flag:
        if embeddings is None or labels is None:
            raise ValueError("Must pass embeddings+labels when generate_triplets_flag=True")
        anchors, positives, negatives = generate_triplets(embeddings, labels)
    else:
        if anchors is None or positives is None or negatives is None:
            raise ValueError("Must pass anchors, positives, negatives when generate_triplets_flag=False")

    if anchors is None:
        # no valid triplets in batch
        return torch.tensor(0.0, device=embeddings.device if embeddings is not None else anchors.device)

    # compute distances
    pos_dist = F.pairwise_distance(anchors, positives)
    neg_dist = F.pairwise_distance(anchors, negatives)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()