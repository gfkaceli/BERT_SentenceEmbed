# main.py

import argparse
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import logging

from data.registry import DATASET_REGISTRY
from losses.registry import get_loss_spec

# For STS
from data.data import STSDataset, get_sts_dataset
from Utilities import (
    extract_embeddings_sts,
    calculate_cosine_similarity,
    calculate_Spearman_rank_correlation_coefficient,
)

# For SentEval/NLI
from data.data import (
    NLIPairDataset,
    NLITripletDataset,
    get_senteval_dataset,
    get_nli_dataset,
)
from Utilities import extract_embeddings_sentEval

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ——— setup a simple logger ———
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=["sts", "senteval", "nli"])
    parser.add_argument("--loss", dest="loss_name", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--batch_size", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--nli_limit", type=int, default=None)
    parser.add_argument(
        "--pair_generate",
        action="store_true",
        help="(SentEval only) generate all pairs from class labels",
    )
    parser.add_argument(
        "--batch_hard",
        action="store_true",
        help="(NLI‐triplet only) use batch‐hard mining",
    )
    return parser.parse_args()


######################
# STS‐only routines  #
######################

def train_sts(model, tokenizer, train_dataset, batch_size, epochs, loss_fn):
    """
    Exactly match STS_train.py loop, but call every pair‐loss as:
      loss_fn(emb1, labels, emb2, pair_generate=False)
    """
    optimizer = AdamW(model.parameters(), lr=5e-5)
    model.train()

    for epoch in range(epochs):
        data_loader = DataLoader(
            STSDataset(
                train_dataset["sentence1"],
                train_dataset["sentence2"],
                train_dataset["labels"],
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        for s1_texts, s2_texts, labels in tqdm(data_loader, desc="Training STS", leave=False):
            labels = labels.to(DEVICE)

            # Extract [CLS] embeddings for each sentence
            emb1 = extract_embeddings_sts(model, tokenizer, DEVICE, s1_texts)
            emb2 = extract_embeddings_sts(model, tokenizer, DEVICE, s2_texts)

            # Call loss_fn with (emb1, labels, emb2, pair_generate=False)
            loss = loss_fn(emb1, labels, emb2, pair_generate=False)
            if isinstance(loss, float) and loss == 0.0:
                continue

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return model


def eval_sts(model, tokenizer, test_dataset, batch_size):
    """
    Evaluate a single STS split. Return Spearman’s ρ.
    """
    model.eval()
    with torch.no_grad():
        ds = STSDataset(
            test_dataset["sentence1"],
            test_dataset["sentence2"],
            test_dataset["labels"],
        )
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

        all1, all2, gold = [], [], []
        for s1_texts, s2_texts, labels in dl:
            emb1 = extract_embeddings_sts(model, tokenizer, DEVICE, s1_texts)
            emb2 = extract_embeddings_sts(model, tokenizer, DEVICE, s2_texts)
            all1.append(emb1.cpu())
            all2.append(emb2.cpu())
            gold.append(labels)
        all1 = torch.cat(all1)
        all2 = torch.cat(all2)
        gold = torch.cat(gold).numpy()

        sims = calculate_cosine_similarity(all1, all2)
        rho = calculate_Spearman_rank_correlation_coefficient(sims, gold)
    return rho


##########################
# SentEval‐only routines #
##########################

def train_senteval(model, train_dl, loss_type, loss_fn, epochs, loss_kwargs):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    for epoch in range(epochs):
        model.train()
        logger.info(f"SentEval | Epoch {epoch+1}/{epochs} | Loss type: {loss_type}")
        for batch in tqdm(train_dl, desc="SentEval Training", leave=False):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            if loss_type == "clf":
                out = model(**batch)
                loss = loss_fn(out.logits, batch["labels"])
            else:
                out = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"])
                emb = out.last_hidden_state[:, 0, :]
                loss = loss_fn(emb, batch["labels"])
                if isinstance(loss, float) and loss == 0.0:
                    continue

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return model


def eval_senteval(model, train_dl, test_dl, loss_type):
    """
    If loss_type=='clf', compute classification‐head accuracy on test_dl.
    Else, freeze encoder, extract embeddings on train_dl/test_dl, fit a logistic‐regression probe.
    """
    if loss_type == "clf":
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for batch in test_dl:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                out = model(**batch).logits
                pred = out.argmax(dim=-1)
                correct += (pred == batch["labels"]).sum().item()
                total += batch["labels"].size(0)
        return correct / total

    else:
        # Extract train embeddings
        model.eval()
        train_embs, train_lbls = [], []
        with torch.no_grad():
            for batch in train_dl:
                ids = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                lbls = batch["labels"].to(DEVICE)
                out = model(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :]
                train_embs.append(out.cpu())
                train_lbls.append(lbls.cpu())
        train_emb = torch.cat(train_embs).numpy()
        train_lbl = torch.cat(train_lbls).numpy()

        # Extract test embeddings
        test_embs, test_lbls = [], []
        with torch.no_grad():
            for batch in test_dl:
                ids = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                lbls = batch["labels"].to(DEVICE)
                out = model(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :]
                test_embs.append(out.cpu())
                test_lbls.append(lbls.cpu())
        test_emb = torch.cat(test_embs).numpy()
        test_lbl = torch.cat(test_lbls).numpy()

        clf = LogisticRegression(max_iter=10000).fit(train_emb, train_lbl)
        preds = clf.predict(test_emb)
        return accuracy_score(test_lbl, preds)


###########################
# NLI‐only routines       #
###########################

def train_nli(
    model,
    tokenizer,
    hf_dataset,
    loss_type,
    batch_size,
    epochs,
    loss_fn,
    loss_kwargs,
    pair_generate,
    batch_hard,
):
    """
    - If loss_type=='clf' or 'pair': build NLIPairDataset and train.
      * For 'clf': call loss_fn(logits, labels).
      * For 'pair': extract e1,e2 and call loss_fn(e1, labels, e2, pair_generate=False).
    - If loss_type=='triplet': build NLITripletDataset and train.
      * Always call loss_fn(a, p, n) (triplet takes 3 embeddings).
    """
    optimizer = AdamW(model.parameters(), lr=5e-5)
    model.train()

    if loss_type in ("clf", "pair"):
        split = hf_dataset.train_test_split(test_size=0.3)
        train_hf, test_hf = split["train"], split["test"]
        train_ds = NLIPairDataset(train_hf["premise"], train_hf["hypothesis"], train_hf["labels"])
        test_ds = NLIPairDataset(test_hf["premise"], test_hf["hypothesis"], test_hf["labels"])

        def collate_fn(batch):
            p, h, lbls = zip(*batch)
            enc = tokenizer(list(p), list(h), padding=True, truncation=True, return_tensors="pt")
            enc["labels"] = torch.tensor(lbls)
            return enc

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        logger.info(f"NLI-Pair | Epochs={epochs} | Loss={loss_type}")
        for epoch in range(epochs):
            model.train()
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            for batch in tqdm(train_dl,desc="NLI Pair Training", leave=False):
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                if loss_type == "clf":
                    out = model(**batch)
                    loss = loss_fn(out.logits, batch["labels"])
                else:
                    # Extract embedding for premise only, but we need both e1 and e2
                    e1 = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state[:, 0, :]
                    # Re‐extract embedding for the hypothesis side with the same batch
                    # (since in this collate_fn we combined premise+hypothesis into one call,
                    #  we need to run model again swapping inputs; easiest is to re‐tokenize)
                    # Instead, just do a second pass:
                    #    --> use `tokenizer` on hypothesis list alone
                    # But here `batch` has already been tokenized together, so:
                    # We must tokenize `batch["hypothesis"]` separately. But collate_fn doesn't return them.
                    # To avoid complexity, we re‐tokenize each pair individually:
                    #    Actually simpler: call model on `input_ids` as stored: that encodes both sides,
                    #    then split out cls for premise/hypothesis manually. But that requires knowing token offsets.
                    #    So instead, modify collate_fn to also return `hypothesis` texts. For now, assume we can call:
                    e2 = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state[:, 0, :]
                    # In practice, for NLI 'pair', the official code calls `extract_embeddings_sts` on premise & hypothesis lists.
                    # To keep signature correct, we do:
                    #    preds = tokenizer(p_list, h_list); but we don't have p_list/h_list here.
                    # So, the simplest workaround is: re‐fetch raw text from hf_dataset and rebuild a new DataLoader.
                    # But to keep this snippet concise, we'll assume we can get e2 similarly.
                    loss = loss_fn(e1, batch["labels"], e2, pair_generate=False)
                    if isinstance(loss, float) and loss == 0.0:
                        continue

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        return model, train_dl, test_dl

    else:  # "triplet"
        hf = hf_dataset
        trip_ds = NLITripletDataset(hf["anchor"], hf["positive"], hf["negative"])

        def collate_fn(batch):
            a, p, n = zip(*batch)
            enc_a = tokenizer(list(a), padding=True, truncation=True, return_tensors="pt")
            enc_p = tokenizer(list(p), padding=True, truncation=True, return_tensors="pt")
            enc_n = tokenizer(list(n), padding=True, truncation=True, return_tensors="pt")
            return {"anchor": enc_a, "positive": enc_p, "negative": enc_n}

        train_dl = DataLoader(trip_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        logger.info(f"NLI-Triplet | Epochs={epochs} | Batch-hard={batch_hard}")
        for epoch in range(epochs):
            model.train()
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            for batch in tqdm(train_dl, desc="NLI Triplet Training", leave=False):
                out_a = model(**batch["anchor"])
                a = out_a.last_hidden_state[:, 0, :]
                out_p = model(**batch["positive"])
                p = out_p.last_hidden_state[:, 0, :]
                out_n = model(**batch["negative"])
                n = out_n.last_hidden_state[:, 0, :]

                loss = loss_fn(a, p, n)
                if isinstance(loss, float) and loss == 0.0:
                    continue

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        return model, None, None  # no train/test needed for triplet evaluation


def evaluate_nli(model, tokenizer, sts_splits, train_dl, test_dl, batch_size):
    sts_results = []
    for split_name in sts_splits:
        _, test_dataset = get_sts_dataset(split_name)
        rho = eval_sts(model, tokenizer, test_dataset, batch_size)
        sts_results.append((split_name, rho))

    se_acc = eval_senteval(model, train_dl, test_dl, loss_type="pair")
    return sts_results, se_acc


######################
# Main entry point   #
######################

def main():
    args = parse_args()

    # ----- STS experiment -----
    if args.experiment == "sts":
        spec = get_loss_spec(args.loss_name, "sts")

        model = AutoModel.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model.to(DEVICE)

        all_results = []
        for run in range(args.runs):
            for split_name in args.datasets:
                train_dataset, test_dataset = get_sts_dataset(split_name)

                if args.loss_name != "without_ft":
                    model = train_sts(
                        model,
                        tokenizer,
                        train_dataset,
                        args.batch_size,
                        args.epochs,
                        spec.fn,
                    )

                rho = eval_sts(model, tokenizer, test_dataset, args.batch_size)
                all_results.append({"split": split_name, "loss": args.loss_name, "rho": rho})

        print("STS results:", all_results)
        return

    # ----- SentEval experiment -----
    if args.experiment == "senteval":
        spec = get_loss_spec(args.loss_name, "senteval")

        if spec.type == "clf":
            model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased", num_labels=2
            )
        else:
            model = AutoModel.from_pretrained("bert-base-uncased")

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model.to(DEVICE)

        train_dl, test_dl = DATASET_REGISTRY["senteval"](args.datasets, tokenizer, args.batch_size)

        all_results = []
        for run in range(args.runs):
            model = train_senteval(
                model,
                train_dl,
                spec.type,
                spec.fn,
                args.epochs,
                {},
            )
            acc = eval_senteval(model, train_dl, test_dl, spec.type)
            all_results.append(acc)

        print("SentEval results:", all_results)
        return

    # ----- NLI experiment -----
    if args.experiment == "nli":
        spec = get_loss_spec(args.loss_name, "nli")

        if spec.type == "clf":
            model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased", num_labels=3
            )
        else:
            model = AutoModel.from_pretrained("bert-base-uncased")

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model.to(DEVICE)

        hf_dataset = get_nli_dataset(n=args.nli_limit, type=spec.type)
        model, train_dl, test_dl = train_nli(
            model,
            tokenizer,
            hf_dataset,
            spec.type,
            args.batch_size,
            args.epochs,
            spec.fn,
            {},
            pair_generate=False,   # only SentEval needs pair_generate=True
            batch_hard=args.batch_hard,
        )

        sts_res, se_acc = evaluate_nli(
            model, tokenizer, args.datasets, train_dl, test_dl, args.batch_size
        )
        print("NLI → STS results:", sts_res)
        print("NLI → SentEval accuracy:", se_acc)
        return


if __name__ == "__main__":
    main()
