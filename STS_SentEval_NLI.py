from Loss_Function_STS_SentEval_NLI.CrossEntropy import cross_entropy_loss
from Loss_Function_STS_SentEval_NLI.Label_Smoothed_CE import label_smoothing_cross_entropy_loss
from Loss_Function_STS_SentEval_NLI.Triplet_Loss import triplet_loss
from Loss_Function_STS_SentEval_NLI.Hard_Triplet_Loss import hard_triplet_loss
from Loss_Function_STS_SentEval_NLI.CosineSim_MSE import cosine_similarity_mse_loss
from Loss_Function_STS_SentEval_NLI.CoSent_Loss import cosent_loss
from Loss_Function_STS_SentEval_NLI.InBatchNegative import in_batch_negative_loss
from Loss_Function_STS_SentEval_NLI.AngleLoss import angle_loss
from Loss_Function_STS_SentEval_NLI.Combined import cosent_ibn_angle
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from data import (NLIPairDataset, NLITripletDataset, STSDataset, get_sts_dataset, SentEvalDataset,
                  get_senteval_dataset, get_nli_dataset)
from Utilities import calculate_Spearman_rank_correlation_coefficient, calculate_cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



sts_datasets = ['STS-B', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'SICK-R']
senteval_datasets = ['CR', 'MPQA', 'MR', 'SUBJ']
model_id = 'bert-base-uncased'


losses = [
    {'loss_name': cross_entropy_loss, 'loss_type': 'clf', 'loss_kwargs': {}},
    {'loss_name': label_smoothing_cross_entropy_loss, 'loss_type': 'clf', 'loss_kwargs': {'smoothing': 0.1}},
    {'loss_name': triplet_loss, 'loss_type': 'triplet', 'loss_kwargs': {'margin': 5}},
    {'loss_name': hard_triplet_loss, 'loss_type': 'triplet', 'loss_kwargs': {'margin': 5}},
    {'loss_name': cosine_similarity_mse_loss, 'loss_type': 'pair', 'loss_kwargs': {}},
    {'loss_name': cosent_loss, 'loss_type': 'pair', 'loss_kwargs': {'tau': 20.0}},
    {'loss_name': in_batch_negative_loss, 'loss_type': 'pair', 'loss_kwargs': {'tau': 20.0}},
    {'loss_name': angle_loss, 'loss_type': 'pair', 'loss_kwargs': {'tau': 1.0}},
    {'loss_name': cosent_ibn_angle, 'loss_type': 'pair', 'loss_kwargs': {'w_cosent': 1, 'w_ibn': 1, 'w_angle': 1, 'tau_cosent': 20.0, 'tau_ibn': 20.0, 'tau_angle': 1.0}}
]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_model_tokenizer(model_id, type='clf', num_labels=2):
    if type == 'clf':
        model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)
    else:
        model = AutoModel.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.to(device)
    return model, tokenizer

def extract_embeddings(model, tokenizer, device, sentences, to_numpy=False):
    encodings = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True).to(device)

    if 'Classification' in str(type(model)):
        outputs = model(**encodings, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1][:, 0, :]
    else:
        embeddings = model(**encodings).last_hidden_state[:, 0, :]

    if to_numpy:
        embeddings = embeddings.cpu().detach().numpy()
    return embeddings


def train(model, tokenizer, dataset, batch_size, loss_type='clf', epochs=10, loss_name=cross_entropy_loss, **loss_kwargs):
    # Optimizer setting...
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Training loop...
    num_epochs = epochs
    model.train()
    for epoch in range(num_epochs):
        if loss_type == 'clf':
            data_loader = DataLoader(NLIPairDataset(dataset['premise'], dataset['hypothesis'], dataset['labels']), batch_size=batch_size, shuffle=True)

            for premise_texts, hypothesis_texts, labels in tqdm(data_loader, desc="Training", leave=False):
                inputs = tokenizer(premise_texts, hypothesis_texts, padding=True, truncation=True, return_tensors="pt").to(device)
                labels = labels.to(device)

                # Forward pass...
                outputs = model(**inputs)
                logits = outputs.logits
                loss = loss_name(logits, labels)

                # Backpropagation...
                loss.backward()

                # Updating weights...
                optimizer.step()
                optimizer.zero_grad()
        elif loss_type == 'pair':
            data_loader = DataLoader(NLIPairDataset(dataset['premise'], dataset['hypothesis'], dataset['labels']), batch_size=batch_size, shuffle=True)

            for premise_texts, hypothesis_texts, labels in tqdm(data_loader, desc="Training", leave=False):
                labels = labels.to(device)

                # [CLS] token embedding...
                premise_embeddings = extract_embeddings(model, tokenizer, device, premise_texts)
                hypothesis_embeddings = extract_embeddings(model, tokenizer, device, hypothesis_texts)

                # Embedding Loss...
                loss = loss_name(premise_embeddings, hypothesis_embeddings, labels)
                if loss == 0.0:
                    continue

                # Backpropagation...
                loss.backward()

                # Updating weights...
                optimizer.step()
                optimizer.zero_grad()
        elif loss_type == 'triplet':
            data_loader = DataLoader(NLITripletDataset(dataset['anchor'], dataset['positive'], dataset['negative']), batch_size=batch_size, shuffle=True)

            for anchor_texts, positive_texts, negative_texts in tqdm(data_loader, desc="Training", leave=False):
                # [CLS] token embedding...
                anchor_embeddings = extract_embeddings(model, tokenizer, device, anchor_texts)
                positive_embeddings = extract_embeddings(model, tokenizer, device, positive_texts)
                negative_embeddings = extract_embeddings(model, tokenizer, device, negative_texts)

                # Embedding Loss...
                loss = loss_name(anchor_embeddings, positive_embeddings, negative_embeddings)
                if loss == 0.0:
                    continue

                # Backpropagation...
                loss.backward()

                # Updating weights...
                optimizer.step()
                optimizer.zero_grad()
    return model


def evaluate_sts(model, tokenizer, loss, batch_size):
    model.eval()
    spearman_list = []

    for dataset_name in sts_datasets:
        dataset = get_sts_dataset(dataset_name)
        dataset = STSDataset(dataset['sentence1'], dataset['sentence2'], dataset['labels'])
        data_loader = DataLoader(dataset, batch_size=batch_size)
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
        spearman_list.append({'loss': loss, 'dataset': dataset_name, 'spearman': spearman})
    return spearman_list


def evaluate_senteval(model, tokenizer, loss, batch_size):
    model.eval()
    accuracy_list = []

    for dataset_name in senteval_datasets:
        dataset = get_senteval_dataset(dataset_name)
        dataset = SentEvalDataset(dataset['sentence'], dataset['labels'])
        data_loader = DataLoader(dataset, batch_size=batch_size)
        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for sentences, labels in tqdm(data_loader, desc="Extracting embeddings", leave=False):
                embeddings = extract_embeddings(model, tokenizer, device, sentences)
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels.cpu())

        data_embeddings = torch.cat(all_embeddings)
        data_labels = torch.cat(all_labels)

        data_embeddings_np = data_embeddings.numpy()
        data_labels_np = data_labels.numpy()

        train_embeddings, test_embeddings, train_labels, test_labels = train_test_split(data_embeddings_np, data_labels_np, test_size=0.3)

        # Training a Logistic Regression classifier on the training embeddings...
        lr_clf = LogisticRegression(max_iter=10000)
        lr_clf.fit(train_embeddings, train_labels)

        # Predicting the labels for the test set...
        test_predictions = lr_clf.predict(test_embeddings)
        accuracy = accuracy_score(test_labels, test_predictions)
        accuracy_list.append({'loss': loss, 'dataset': dataset_name, 'accuracy': accuracy})
    return accuracy_list

if __name__ == '__main__':
    total_runs = 3
    batch_size = 60
    ds_length = None
    senteval_results_list = []
    sts_results_list = []
    for loss in losses:
        loss_name = loss['loss_name']
        loss_type = loss['loss_type']
        loss_kwargs = loss['loss_kwargs']

        for loop_count in range(0, total_runs):
            # Dataset Preparation...
            dataset = get_nli_dataset(n=ds_length, type=loss_type)

            # Model Preparation...
            model, tokenizer = get_model_tokenizer(model_id, loss_type)

            # Training Loop...
            model = train(model, tokenizer, dataset, batch_size, loss_type, epochs=10, loss_name=loss_name,
                          **loss_kwargs)

            # Evaluation loop...
            sts_results = evaluate_sts(model, tokenizer, loss_name, batch_size)
            senteval_results = evaluate_senteval(model, tokenizer, loss_name, batch_size)
        senteval_results_list.append(senteval_results)
        sts_results_list.append(sts_results)
        print("Training done......")