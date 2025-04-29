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

class SentEvalDataset(torch.utils.data.Dataset):
    def __init__(self, sentence, label):
        self.sentence = sentence
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.sentence[idx], self.label[idx]


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

"""NLI SECTION"""

def select_n_nli(nli_dataset, n):
    positive_samples = nli_dataset.filter(lambda example: example['labels'] == 1)
    negative_samples = nli_dataset.filter(lambda example: example['labels'] == 0)
    positive_samples = positive_samples.shuffle(seed=42).select(range(int(n / 2)))
    negative_samples = negative_samples.shuffle(seed=42).select(range(int(n / 2)))
    balanced_dataset = concatenate_datasets([positive_samples, negative_samples]).shuffle(seed=42)
    return balanced_dataset

def get_nli_dataset(n=10000, type='pair', exclude_neutral=True):
    if type == 'pair' or type == 'clf':
        dataset = load_dataset('sentence-transformers/all-nli', 'pair-class')

        # Mapping the labels in such a way so that contradiction is the least similar and entailment is the most similar...
        label_mapping = {
            0: 'contradiction',
            2: 'neutral',
            1: 'entailment'
        }

        def map_labels(example):
            example["label"] = label_mapping[example["label"]]
            return example

        dataset = dataset.map(map_labels)
        if exclude_neutral:
            dataset = dataset.filter(lambda example: example['label'] != 2)
        dataset = dataset.rename_column('label', 'labels')
        return dataset['train'] if n is None else select_n_nli(dataset['train'], n)
    elif type == 'triplet':
        dataset = load_dataset('sentence-transformers/all-nli', 'triplet')
        return dataset['train'] if n is None else dataset['train'].select(range(n))

class NLIPairDataset(torch.utils.data.Dataset):
    def __init__(self, premises, hypotheses, labels):
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.premises[idx], self.hypotheses[idx], self.labels[idx]

class NLITripletDataset(torch.utils.data.Dataset):
    def __init__(self, anchor, positive, negative):
        self.anchor = anchor
        self.positive = positive
        self.negative = negative

    def __len__(self):
        return len(self.anchor)

    def __getitem__(self, idx):
        return self.anchor[idx], self.positive[idx], self.negative[idx]


