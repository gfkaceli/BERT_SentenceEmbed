import torch.nn.functional as F
from losses.normalization import norm_function
from GeneratePairs import generate_pairs


def cosine_similarity_mse_loss(embedding1, labels, embedding2=None, pair_generate=True):
    # Calculating the cosine similarity between the pairs of embeddings...
    if pair_generate:
        embedding1, embedding2 = generate_pairs(embedding1, labels)
    else:
        if embedding2 is None and pair_generate is False:
            raise ValueError("If pair_generate is False you must pass embedding2 values")

    cos_sim = F.cosine_similarity(embedding1, embedding2)

    # MSE loss...
    squared_difference = (labels - cos_sim) ** 2
    loss = squared_difference.mean()

    return loss

def cosine_similarity_mse_norm(embedding1, embedding2, labels, norm, pair_generate):
    labels_norm = norm_function(norm, labels)


    # Calculating the cosine similarity between the pairs of embeddings...
    cos_sim = F.cosine_similarity(embedding1, embedding2)

    # MSE loss...
    squared_difference = (labels_norm - cos_sim) ** 2
    loss = squared_difference.mean()

    return loss