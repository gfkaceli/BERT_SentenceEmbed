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


def triplet_loss(embeddings, labels, margin=1.0):
    anchors, positives, negatives = generate_triplets(embeddings, labels)
    if anchors is None or positives is None or negatives is None:
        return 0.0

    # Euclidean distance between anchor and positive, and anchor and negative...
    pos_dist = F.pairwise_distance(anchors, positives)
    neg_dist = F.pairwise_distance(anchors, negatives)
    loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
    return loss.mean()

def triplet_all_loss(embeddings, labels, margin=1.0):
    anchors, positives, negatives = generate_all_triplets(embeddings, labels)
    if anchors is None or positives is None or negatives is None:
        return 0.0

    # Euclidean distance between anchor and positive, and anchor and negative...
    pos_dist = F.pairwise_distance(anchors, positives)
    neg_dist = F.pairwise_distance(anchors, negatives)
    loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
    return loss.mean()