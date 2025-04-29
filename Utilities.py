import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import get_senteval_dataset
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from scipy.stats import spearmanr
import torch.nn.functional as F

def prepare_dataset(dataset, split=0.3):
    # Dataset Import...
    ds = get_senteval_dataset(dataset)

    # Random Split...
    train_test_split = ds.train_test_split(test_size=split)
    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']
    return train_dataset, test_dataset

def get_model_tokenizer_sts(model_id, device):
    model = AutoModel.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.to(device)
    return model, tokenizer

def get_model_tokenizer_sentEval(model_id, device,loss_type='clf'):
    if loss_type == 'clf':
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
    else:
        model = AutoModel.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.to(device)
    return model, tokenizer


def extract_embeddings_sts(model, tokenizer, device, sentences, to_numpy=False):
    encodings = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True).to(device)
    # [CLS] token embedding...
    embeddings = model(**encodings).last_hidden_state[:, 0, :]

    if to_numpy:
        embeddings = embeddings.cpu().detach().numpy()
    return embeddings

def extract_embeddings_sentEval(model, device, dataloader):
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

            # [CLS] token embeddings...
            embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(embeddings.cpu())
            all_labels.append(batch['labels'].cpu())

    return torch.cat(all_embeddings), torch.cat(all_labels)

def tokenize_dataset_batch(train_dataset, test_dataset, tokenizer, batch_size):
    def tokenize_function(examples):
        return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=128)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Torch format setting...
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Batching using DataLoader...
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader


def calculate_cosine_similarity(embeddings_1, embeddings_2):
    cosine_similarity = F.cosine_similarity(embeddings_1, embeddings_2, dim=1)
    return cosine_similarity

def calculate_Spearman_rank_correlation_coefficient(scores, scores_actual):
    sc, _ = spearmanr(scores, scores_actual)
    return sc


