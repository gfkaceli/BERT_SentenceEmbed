

sts_datasets = ['STS-B', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'SICK-R']
senteval_datasets = ['CR', 'MPQA', 'MR', 'SUBJ']
model_id = 'bert-base-uncased'


losses = [
    {'loss_name': 'cross_entropy_loss', 'loss_type': 'clf', 'loss_kwargs': {}},
    {'loss_name': 'label_smoothing_cross_entropy_loss', 'loss_type': 'clf', 'loss_kwargs': {'smoothing': 0.1}},
    {'loss_name': 'triplet_loss', 'loss_type': 'triplet', 'loss_kwargs': {'margin': 5}},
    {'loss_name': 'hard_triplet_loss', 'loss_type': 'triplet', 'loss_kwargs': {'margin': 5}},
    {'loss_name': 'cosine_similarity_mse_loss', 'loss_type': 'pair', 'loss_kwargs': {}},
    {'loss_name': 'cosent_loss', 'loss_type': 'pair', 'loss_kwargs': {'tau': 20.0}},
    {'loss_name': 'in_batch_negative_loss', 'loss_type': 'pair', 'loss_kwargs': {'tau': 20.0}},
    {'loss_name': 'angle_loss', 'loss_type': 'pair', 'loss_kwargs': {'tau': 1.0}},
    {'loss_name': 'cosent_ibn_angle', 'loss_type': 'pair', 'loss_kwargs': {'w_cosent': 1, 'w_ibn': 1, 'w_angle': 1, 'tau_cosent': 20.0, 'tau_ibn': 20.0, 'tau_angle': 1.0}}
]
