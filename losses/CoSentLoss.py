import torch
import torch.nn.functional as F


def cosent_loss(embedding1, embedding2, labels, tau=20.0, pair_generate=False):

    if pair_generate:
        embedding1, embedding2, labels = generate_pairs(embedding1, labels)

    # Input preparation...
    labels = (labels[:, None] < labels[None, :]).float()

    # Normalization of Logits...
    embedding1 = F.normalize(embedding1, p=2, dim=1)
    embedding2 = F.normalize(embedding2, p=2, dim=1)

    # Cosine Similarity Calculation...
    # The dot product of these pairs gives the cosine similarity, scaled by a factor of tau to control the sharpness of similarity scores...
    y_pred = torch.sum(embedding1 * embedding2, dim=1) * tau

    # Pairwise cosine similarity difference calculation...
    y_pred = y_pred[:, None] - y_pred[None, :]

    y_pred = (y_pred - (1 - labels) * 1e12).view(-1)

    zero = torch.Tensor([0]).to(y_pred.device)
    y_pred = torch.concat((zero, y_pred), dim=0)
    return torch.logsumexp(y_pred, dim=0)


def generate_pairs(embeddings, labels):
    embedding1_list = []
    embedding2_list = []
    similarity_labels = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            embedding1 = embeddings[i]
            embedding2 = embeddings[j]

            # If the labels are the same, labeling the pair as 1 (similar)...
            if labels[i] == labels[j]:
                similarity_labels.append(1)
            else:
                # If the labels are different, labeling the pair as 0 (dissimilar)...
                similarity_labels.append(0)

            embedding1_list.append(embedding1)
            embedding2_list.append(embedding2)

    embedding1_tensor = torch.stack(embedding1_list)
    embedding2_tensor = torch.stack(embedding2_list)
    labels_tensor = torch.tensor(similarity_labels).to(labels.device)

    return embedding1_tensor, embedding2_tensor, labels_tensor