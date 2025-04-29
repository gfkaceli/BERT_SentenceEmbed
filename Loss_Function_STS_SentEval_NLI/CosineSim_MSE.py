import torch.nn.functional as F

def cosine_similarity_mse_loss(embedding1, embedding2, labels):
    # Calculating the cosine similarity between the pairs of embeddings...
    cos_sim = F.cosine_similarity(embedding1, embedding2)

    # MSE loss...
    squared_difference = (labels - cos_sim) ** 2
    loss = squared_difference.mean()

    return loss