from Loss_Functions_SentEval.Generate_Pairs import generate_pairs
import torch.nn.functional as F

def cosine_similarity_mse_loss(embeddings, labels):
    embedding1, embedding2, labels = generate_pairs(embeddings, labels)

    # Calculating the cosine similarity between the pairs of embeddings...
    cos_sim = F.cosine_similarity(embedding1, embedding2)

    # MSE loss...
    squared_difference = (labels - cos_sim) ** 2
    loss = squared_difference.mean()

    return loss