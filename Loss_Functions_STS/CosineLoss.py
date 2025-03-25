import torch.nn.functional as F
from Loss_Functions_STS.normalization import norm_function


def cosine_similarity_mse_loss(embedding1, embedding2, labels):
    # Calculating the cosine similarity between the pairs of embeddings...
    cos_sim = F.cosine_similarity(embedding1, embedding2)

    # MSE loss...
    squared_difference = (labels - cos_sim) ** 2
    loss = squared_difference.mean()

    return loss

def cosine_similarity_mse_norm(embedding1, embedding2, labels, norm):
    labels_norm = norm_function(norm, labels)
    # Calculating the cosine similarity between the pairs of embeddings...
    cos_sim = F.cosine_similarity(embedding1, embedding2)

    # MSE loss...
    squared_difference = (labels_norm - cos_sim) ** 2
    loss = squared_difference.mean()

    return loss