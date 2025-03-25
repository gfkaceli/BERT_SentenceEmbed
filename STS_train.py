# where we run the main training loops
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import STSDataset
from Utilities import (extract_embeddings,
                       calculate_cosine_similarity,
                       calculate_Spearman_rank_correlation_coefficient,
                       get_model_tokenizer)
from data import get_sts_dataset
from Loss_Functions_STS.CosineLoss import cosine_similarity_mse_loss, cosine_similarity_mse_norm
from Loss_Functions_STS.AngleLoss import angle_loss
from Loss_Functions_STS.CombinedLoss import cosent_ibn_angle
from Loss_Functions_STS.CoSent import cosent_loss
from Loss_Functions_STS.InBatchNegative import in_batch_negative_loss

model_id = "bert-base-uncased"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
sts_datasets = ['STS-B', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'SICK-R']

losses = [
    {'loss_name': 'without_ft', 'loss_type': 'emb', 'loss_kwargs': {}},
    {'loss_name': cosine_similarity_mse_loss, 'loss_type': 'emb', 'loss_kwargs': {}},
    {'loss_name': cosine_similarity_mse_norm, 'loss_type': 'emb', 'loss_kwargs': {'norm': 'divided_by_maximum'}},
    {'loss_name': cosent_loss, 'loss_type': 'emb', 'loss_kwargs': {'tau': 20.0}},
    {'loss_name': in_batch_negative_loss, 'loss_type': 'emb', 'loss_kwargs': {'tau': 20.0}},
    {'loss_name': angle_loss, 'loss_type': 'emb', 'loss_kwargs': {'tau': 1.0}},
    {'loss_name': cosent_ibn_angle, 'loss_type': 'emb', 'loss_kwargs': {'w_cosent': 1, 'w_ibn': 1, 'w_angle': 1,
                                                                          'tau_cosent': 20.0, 'tau_ibn': 20.0,
                                                                          'tau_angle': 1.0}}
]


def train(model, tokenizer, dataset, batch_size, epochs=10, loss_fn=cosine_similarity_mse_loss, **loss_kwargs):
    # Optimizer setting...
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Training loop...
    num_epochs = epochs
    model.train()
    for epoch in range(num_epochs):
        # print(f"Epoch {epoch + 1}/{num_epochs}")
        data_loader = DataLoader(STSDataset(dataset['sentence1'], dataset['sentence2'], dataset['labels']), batch_size=batch_size, shuffle=True)

        for sentence1_texts, sentence2_texts, labels in tqdm(data_loader, desc="Training", leave=False):
            labels = labels.to(device)

            # [CLS] token embedding...
            sentence1_embeddings = extract_embeddings(model, tokenizer, device, sentence1_texts)
            sentence2_embeddings = extract_embeddings(model, tokenizer, device, sentence2_texts)

            # Embedding Loss...
            loss = loss_fn(sentence1_embeddings, sentence2_embeddings, labels)
            if loss == 0.0:
                continue

            # Backpropagation...
            loss.backward()

            # Updating weights...
            optimizer.step()
            optimizer.zero_grad()
    return model

def evaluate_sts(model, tokenizer, test_dataset, batch_size):
    model.eval()

    test_dataset = STSDataset(test_dataset['sentence1'], test_dataset['sentence2'], test_dataset['labels'])
    data_loader = DataLoader(test_dataset, batch_size=batch_size)
    all_embeddings1 = []
    all_embeddings2 = []
    all_labels = []

    with torch.no_grad():
        for sentences1, sentences2, labels in tqdm(data_loader, desc="Extracting embeddings", leave=False):
            embeddings1 = extract_embeddings(model, tokenizer, device, sentences1)
            embeddings2 = extract_embeddings(model, tokenizer, device, sentences2)
            all_embeddings1.append(embeddings1.cpu())
            all_embeddings2.append(embeddings2.cpu())
            all_labels.append(labels.cpu())

    data_embeddings1 = torch.cat(all_embeddings1)
    data_embeddings2 = torch.cat(all_embeddings2)
    data_labels = torch.cat(all_labels)
    data_labels_np = data_labels.numpy()

    cosine_similarities = calculate_cosine_similarity(data_embeddings1, data_embeddings2)
    spearman = calculate_Spearman_rank_correlation_coefficient(cosine_similarities, data_labels_np)
    return spearman

if __name__=="__main__":
    total_runs = 3
    batch_size = 60
    spearman_list = []
    for loss in losses:
        loss_name = loss['loss_name']
        loss_type = loss['loss_type']
        loss_kwargs = loss['loss_kwargs']

        for dataset in sts_datasets:
            print(f'Running: {loss_name} on {dataset}')
            total_spearman = 0.

            for loop_count in range(0, total_runs):
                # Dataset Preparation...
                train_dataset, test_dataset = get_sts_dataset(dataset)

                # Model Preparation...
                model, tokenizer = get_model_tokenizer(model_id, device)

                # Training Loop...
                if loss_name != 'without_ft':
                    model = train(model, tokenizer, train_dataset, batch_size, epochs=5, loss_name=loss_name,
                                  **loss_kwargs)

                # Evaluation loop...
                spearman = evaluate_sts(model, tokenizer, test_dataset, batch_size)
                # print(f'Loop {loop_count} spearman - {spearman}')
                total_spearman += spearman
            spearman_list.append({'loss': loss_name, 'dataset': dataset, 'spearman': total_spearman / total_runs})
            print(spearman_list)




