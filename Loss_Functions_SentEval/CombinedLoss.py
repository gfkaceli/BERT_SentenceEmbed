from Loss_Functions_SentEval.CoSentLoss import cosent_loss
from Loss_Functions_SentEval.AngleLoss import angle_loss
from Loss_Functions_SentEval.InBatchNegative import in_batch_negative_loss

def cosent_ibn_angle(embeddings, labels, w_cosent=1, w_ibn=1, w_angle=1,
                     tau_cosent=20.0, tau_ibn=20.0, tau_angle=1.0):

    return (w_cosent * cosent_loss(embeddings, labels, tau_cosent) +
            w_ibn * in_batch_negative_loss(embeddings, labels, tau_ibn) +
            w_angle * angle_loss(embeddings, labels, tau_angle))