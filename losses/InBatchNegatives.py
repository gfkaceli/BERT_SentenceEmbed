import torch
import torch.nn.functional as F

from .GeneratePairs import generate_pairs


def categorical_crossentropy(y_true, y_pred):
    return -(F.log_softmax(y_pred, dim=1) * y_true).sum(dim=1)

def in_batch_negative_loss(embedding1,labels ,embedding2=None, tau=20.0, negative_weights=0.0,
                           pair_generate=True):
    device = labels.device
    if pair_generate:
        embedding1, embedding2, labels = generate_pairs(embedding1, labels)
    else:
        if embedding2 is None and pair_generate is False:
            raise ValueError("When pair_generate=False you must pass embedding2")

    y_pred = torch.empty((2 * embedding1.shape[0], embedding1.shape[1]), device=device)
    y_pred[0::2] = embedding1
    y_pred[1::2] = embedding2
    y_true = labels.repeat_interleave(2).unsqueeze(1)

    def make_target_matrix(y_true):
        idxs = torch.arange(0, y_pred.shape[0]).int().to(device)
        y_true = y_true.int()
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]

        idxs_1 *= y_true.T
        idxs_1 += (y_true.T == 0).int() * -2

        idxs_2 *= y_true
        idxs_2 += (y_true == 0).int() * -1

        y_true = (idxs_1 == idxs_2).float()
        return y_true

    neg_mask = make_target_matrix(y_true == 0)

    y_true = make_target_matrix(y_true)

    y_pred = F.normalize(y_pred, dim=1, p=2)
    similarities = y_pred @ y_pred.T
    similarities = similarities - torch.eye(y_pred.shape[0]).to(device) * 1e12
    similarities = similarities * tau

    if negative_weights > 0:
        similarities += neg_mask * negative_weights

    return categorical_crossentropy(y_true, similarities).mean()