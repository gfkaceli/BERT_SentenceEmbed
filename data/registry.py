# data/registry.py

from typing import List, Tuple, Optional
from torch.utils.data import DataLoader, ConcatDataset
from data.data import (
    get_sts_dataset,
    get_senteval_dataset,
    get_nli_dataset,
    STSDataset,
    SentEvalDataset,
    NLIPairDataset,
    NLITripletDataset,
)
import torch

def build_sts_loaders(
    dataset_names: List[str],
    tokenizer,
    batch_size: int,
    split: float = 0.3
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns DataLoaders for STS experiments.
    """
    train_parts, test_parts = [], []
    for name in dataset_names:
        train_hf, test_hf = get_sts_dataset(name, split=split)
        train_parts.append(
            STSDataset(train_hf["sentence1"], train_hf["sentence2"], train_hf["labels"])
        )
        test_parts.append(
            STSDataset(test_hf["sentence1"], test_hf["sentence2"], test_hf["labels"])
        )

    train_ds = ConcatDataset(train_parts)
    test_ds = ConcatDataset(test_parts)

    def collate_fn(batch):
        s1, s2, labels = zip(*batch)
        enc = tokenizer(
            list(s1),
            list(s2),
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        enc["labels"] = torch.tensor(labels)
        return enc

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_dl, test_dl


def build_senteval_loaders(
    dataset_names: List[str],
    tokenizer,
    batch_size: int,
    split: float = 0.3
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns DataLoaders for SentEval experiments.
    """
    train_parts, test_parts = [], []
    for name in dataset_names:
        hf = get_senteval_dataset(name)
        split_ds = hf.train_test_split(test_size=split)
        train_hf, test_hf = split_ds["train"], split_ds["test"]
        train_parts.append(
            SentEvalDataset(train_hf["sentence"], train_hf["labels"])
        )
        test_parts.append(
            SentEvalDataset(test_hf["sentence"], test_hf["labels"])
        )

    train_ds = ConcatDataset(train_parts)
    test_ds  = ConcatDataset(test_parts)

    def collate_fn(batch):
        sents, labels = zip(*batch)
        enc = tokenizer(
            list(sents),
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        enc["labels"] = torch.tensor(labels)
        return enc

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_dl, test_dl


def build_nli_loaders(
    dataset_names: List[str],
    tokenizer,
    batch_size: int,
    loss_type: str,
    n: Optional[int] = 10000,
    split: float = 0.3
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Returns DataLoaders for NLI-based experiments.
    If loss_type is "triplet", only a training loader is returned.
    """
    if loss_type in ("pair", "clf"):
        hf = get_nli_dataset(n=n, type="pair")
        split_ds = hf.train_test_split(test_size=split)
        train_hf, test_hf = split_ds["train"], split_ds["test"]

        train_ds = NLIPairDataset(train_hf["premise"], train_hf["hypothesis"], train_hf["labels"])
        test_ds  = NLIPairDataset(test_hf["premise"],  test_hf["hypothesis"],  test_hf["labels"])

        def collate_fn(batch):
            p, h, labels = zip(*batch)
            enc = tokenizer(
                list(p),
                list(h),
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            enc["labels"] = torch.tensor(labels)
            return enc

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        return train_dl, test_dl

    else:  # triplet
        hf = get_nli_dataset(n=n, type="triplet")
        ds = hf if isinstance(hf, list) else hf["train"]
        trip_ds = NLITripletDataset(ds["anchor"], ds["positive"], ds["negative"])

        def collate_fn(batch):
            a, p, n_ = zip(*batch)
            enc_a = tokenizer(list(a), padding=True, truncation=True, return_tensors="pt")
            enc_p = tokenizer(list(p), padding=True, truncation=True, return_tensors="pt")
            enc_n = tokenizer(list(n_), padding=True, truncation=True, return_tensors="pt")
            return {"anchor": enc_a, "positive": enc_p, "negative": enc_n}

        train_dl = DataLoader(trip_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        return train_dl, None


# Registry mapping experiment type to loader function
DATASET_REGISTRY = {
    "sts":       build_sts_loaders,
    "senteval":  build_senteval_loaders,
    "nli":       build_nli_loaders,
}
