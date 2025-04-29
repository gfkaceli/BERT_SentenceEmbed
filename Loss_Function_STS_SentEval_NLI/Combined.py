from Loss_Function_STS_SentEval_NLI.CoSent_Loss import cosent_loss
from Loss_Function_STS_SentEval_NLI.InBatchNegative import in_batch_negative_loss
from Loss_Function_STS_SentEval_NLI.AngleLoss import angle_loss

def cosent_ibn_angle(embedding1, embedding2, labels, w_cosent=1, w_ibn=1, w_angle=1, tau_cosent=20.0, tau_ibn=20.0, tau_angle=1.0):
    return (w_cosent * cosent_loss(embedding1, embedding2, labels, tau_cosent) +
            w_ibn * in_batch_negative_loss(embedding1, embedding2, labels, tau_ibn) +
            w_angle * angle_loss(embedding1, embedding2, labels, tau_angle))