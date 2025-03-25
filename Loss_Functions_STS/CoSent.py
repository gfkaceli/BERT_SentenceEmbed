import torch
import torch.nn.functional as F


def cosent_loss(embedding1, embedding2, labels, tau=20.0):
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