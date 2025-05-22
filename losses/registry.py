
from dataclasses import dataclass
from typing import Callable, Any, Dict

# --- Core loss implementations ---
from losses.CrossEntropy import cross_entropy_loss
from losses.LabelSmoothingCELoss import label_smoothing_cross_entropy_loss
from losses.CosineMSELoss import cosine_similarity_mse_loss
from losses.CoSentLoss import cosent_loss
from losses.InBatchNegatives import in_batch_negative_loss
from losses.AngleLoss import angle_loss
from losses.CombinedLoss import cosent_ibn_angle
from losses.TripletLoss import triplet_loss
from losses.HardTripletLoss import hard_triplet_loss


# --- Pair‐generation helper ---
from losses.GeneratePairs import generate_pairs

@dataclass
class LossSpec:
    """
    Describes a loss function and its category.
      fn:   the callable to invoke in training
      type: one of "clf", "pair", or "triplet"
    """
    fn: Callable[..., Any]
    type: str

# --- Base registry mapping names to raw LossSpec ---
BASE_LOSS_REGISTRY: Dict[str, LossSpec] = {
    # classification losses
    "cross_entropy":    LossSpec(fn=cross_entropy_loss,            type="clf"),
    "label_smoothing":  LossSpec(fn=label_smoothing_cross_entropy_loss, type="clf"),
    # embedding‐pair losses
    "cosine_mse":       LossSpec(fn=cosine_similarity_mse_loss,    type="pair"),
    "cosent":           LossSpec(fn=cosent_loss,              type="pair"),
    "in_batch_negative":LossSpec(fn=in_batch_negative_loss,   type="pair"),
    "angle":            LossSpec(fn=angle_loss,               type="pair"),
    "cosent_ibn_angle": LossSpec(fn=cosent_ibn_angle,            type="pair"),
    # triplet losses
    "triplet":          LossSpec(fn=triplet_loss,             type="triplet"),
    "hard_triplet":     LossSpec(fn=hard_triplet_loss,        type="triplet"),
}

def _wrap_with_pair_generation(core_fn: Callable[..., Any]) -> Callable[..., Any]:
    """
    Wrap a core pair‐loss so it first generates
    embedding pairs and labels from a batch of embeddings+class labels.
    """
    def wrapped(embeddings, labels, **kwargs):
        e1, e2, pair_labels = generate_pairs(embeddings, labels)
        return core_fn(e1, e2, pair_labels, **kwargs)
    return wrapped

def get_loss_spec(loss_name: str, experiment: str) -> LossSpec:
    """
    Return a LossSpec customized for the given experiment.
    If experiment == "senteval" and the loss is of type "pair",
    we wrap its core implementation to generate pairs internally.
    Otherwise, return the raw LossSpec.
    """
    base = BASE_LOSS_REGISTRY[loss_name]
    if experiment.lower() == "senteval" and base.type == "pair":
        return LossSpec(fn=_wrap_with_pair_generation(base.fn), type=base.type)
    return base