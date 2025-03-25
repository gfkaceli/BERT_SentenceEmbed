from data import get_sts_dataset
from transformers import AutoModel, AutoTokenizer
from scipy.stats import spearmanr
import torch.nn.functional as F


def get_model_tokenizer(model_id, device):
    model = AutoModel.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.to(device)
    return model, tokenizer


def extract_embeddings(model, tokenizer, device, sentences, to_numpy=False):
    encodings = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True).to(device)
    # [CLS] token embedding...
    embeddings = model(**encodings).last_hidden_state[:, 0, :]

    if to_numpy:
        embeddings = embeddings.cpu().detach().numpy()
    return embeddings

def calculate_cosine_similarity(embeddings_1, embeddings_2):
    cosine_similarity = F.cosine_similarity(embeddings_1, embeddings_2, dim=1)
    return cosine_similarity

def calculate_Spearman_rank_correlation_coefficient(scores, scores_actual):
    sc, _ = spearmanr(scores, scores_actual)
    return sc
