import torch
from .GeneratePairs import generate_pairs

def angle_loss(embedding1, labels, embedding2=None, tau=1.0, pair_generate=True):
    if pair_generate:
        embedding1, embedding2=generate_pairs(embedding1,labels)
    else:
        if embedding2 is None and pair_generate==False:
            raise ValueError("When pair_generate=False you must pass embedding2 values")
    # Input preparation...
    labels = (labels[:, None] < labels[None, :]).float()

    # Chunking into real and imaginary parts...
    y_pred_re1, y_pred_im1 = torch.chunk(embedding1, 2, dim=1)
    y_pred_re2, y_pred_im2 = torch.chunk(embedding2, 2, dim=1)

    a = y_pred_re1
    b = y_pred_im1
    c = y_pred_re2
    d = y_pred_im2

    z = torch.sum(c**2 + d**2, dim=1, keepdim=True)
    re = (a * c + b * d) / z
    im = (b * c - a * d) / z

    dz = torch.sum(a**2 + b**2, dim=1, keepdim=True)**0.5
    dw = torch.sum(c**2 + d**2, dim=1, keepdim=True)**0.5
    re /= (dz / dw)
    im /= (dz / dw)

    y_pred = torch.concat((re, im), dim=1)
    y_pred = torch.abs(torch.sum(y_pred, dim=1)) * tau
    y_pred = y_pred[:, None] - y_pred[None, :]
    y_pred = (y_pred - (1 - labels) * 1e12).view(-1)
    zero = torch.Tensor([0]).to(y_pred.device)
    y_pred = torch.concat((zero, y_pred), dim=0)
    return torch.logsumexp(y_pred, dim=0)