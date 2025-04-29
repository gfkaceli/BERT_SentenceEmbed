from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from tqdm import tqdm
import torch
from Loss_Functions_SentEval.CosineSimilarity import cosine_similarity_mse_loss
from Loss_Functions_SentEval.CoSentLoss import cosent_loss
from Loss_Functions_SentEval.HardTripletLoss import hard_triplet_loss
from Loss_Functions_SentEval.TripletLoss import triplet_loss
from Loss_Functions_SentEval.AngleLoss import angle_loss
from Loss_Functions_SentEval.CombinedLoss import cosent_ibn_angle
from Loss_Functions_SentEval.InBatchNegative import in_batch_negative_loss
from Loss_Functions_SentEval.CrossEntropy import cross_entropy_loss
from Loss_Functions_SentEval.LabelSmoothingCELoss import label_smoothing_cross_entropy_loss
from Utilities import extract_embeddings_sentEval, prepare_dataset, get_model_tokenizer_sentEval, tokenize_dataset_batch



model_id = "bert-base-uncased"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
senteval_datasets = ['CR', 'MPQA', 'MR', 'SUBJ']

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


def train(model, train_loader, loss_type='clf', epochs=10, loss_name=cross_entropy_loss, **loss_kwargs):
    # Optimizer setting...
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Training loop...
    num_epochs = epochs
    model.train()
    for epoch in range(num_epochs):
        # print(f"Epoch {epoch + 1}/{num_epochs}")
        for batch in tqdm(train_loader, desc="Training", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}

            if loss_type == 'clf':
                # Cross-Entropy Losses...
                outputs = model(**batch)
                logits = outputs.logits
                loss = loss_name(logits, batch['labels'])
            else:
                # Embedding Loss...
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

                # [CLS] token embedding...
                embeddings = outputs.last_hidden_state[:, 0, :]
                loss = loss_name(embeddings, batch['labels'])
                if loss == 0.0:
                    continue

            # Backpropagation...
            loss.backward()

            # Updating weights...
            optimizer.step()
            optimizer.zero_grad()
    return model

def evaluate_clf(model, test_loader):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass...
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)

            total_correct += (predictions == batch['labels']).sum().item()
            total_samples += batch['labels'].size(0)

    accuracy = total_correct / total_samples
    return accuracy

def evaluate_emb(model, train_loader, test_loader):
    model.eval()

    # Generating embeddings of the train and test sentences...
    train_embeddings, train_labels = extract_embeddings_sentEval(model, device, train_loader)
    test_embeddings, test_labels = extract_embeddings_sentEval(model, device, test_loader)

    train_embeddings_np = train_embeddings.numpy()
    test_embeddings_np = test_embeddings.numpy()
    train_labels_np = train_labels.numpy()
    test_labels_np = test_labels.numpy()

    # Training a Logistic Regression classifier on the training embeddings...
    lr_clf = LogisticRegression(max_iter=10000)
    lr_clf.fit(train_embeddings_np, train_labels_np)

    # Predicting the labels for the test set...
    test_predictions = lr_clf.predict(test_embeddings_np)
    accuracy = accuracy_score(test_labels_np, test_predictions)
    return accuracy

if __name__=="__main__":
    total_runs = 3
    batch_size = 60
    accuracy_list = []
    for loss in losses:
        loss_name = loss['loss_name']
        loss_type = loss['loss_type']
        loss_kwargs = loss['loss_kwargs']

        for dataset in senteval_datasets:
            print(f'Running: {loss_name} on {dataset}')
            total_accuracy = 0

            for loop_count in range(0, total_runs):
                # Dataset Preparation...
                train_dataset, test_dataset = prepare_dataset(dataset)

                # Model Preparation...
                model, tokenizer = get_model_tokenizer_sentEval(model_id, device, loss_type)

                # Tokenize Batch...
                train_loader, test_loader = tokenize_dataset_batch(train_dataset, test_dataset, tokenizer,
                                                                   batch_size=batch_size)

                # Training Loop...
                if loss_name != 'without_ft':
                    model = train(model, train_loader, loss_type, epochs=5, loss_name=loss_name, **loss_kwargs)

                # Evaluation loop...
                if loss_type == 'clf':
                    accuracy = evaluate_clf(model, test_loader)
                else:
                    accuracy = evaluate_emb(model, train_loader, test_loader)
                # print(f'Loop {loop_count} Accuracy - {accuracy}')
                total_accuracy += accuracy
            accuracy_list.append({'loss': loss_name, 'dataset': dataset, 'accuracy': total_accuracy / total_runs})