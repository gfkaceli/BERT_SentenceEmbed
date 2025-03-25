from datasets import load_dataset, concatenate_datasets
import torch

class STSDataset(torch.utils.data.Dataset):
    def __init__(self, sentence1, sentence2, label):
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.sentence1[idx], self.sentence2[idx], self.label[idx]


def get_sts_dataset(dataset_name, split=0.3):
    if dataset_name == 'STS-B':
        dataset_name = 'stsbenchmark'
    elif dataset_name == 'SICK-R':
        dataset_name = 'sickr'
    dataset = load_dataset(f'mteb/{dataset_name.lower()}-sts', split='test')
    dataset = dataset.rename_column('score', 'labels')
    dataset_split = dataset.train_test_split(test_size=split)
    train_dataset, test_dataset = dataset_split['train'], dataset_split['test']
    return train_dataset, test_dataset

def get_senteval_dataset(dataset_name):
    dataset = load_dataset(f'rahulsikder223/SentEval-{dataset_name}')
    concatenated_dataset = concatenate_datasets([dataset['train'], dataset['test']])
    concatenated_dataset = concatenated_dataset.rename_column("label", "labels")
    return concatenated_dataset
