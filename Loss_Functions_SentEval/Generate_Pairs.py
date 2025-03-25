import torch

def generate_pairs(embeddings, labels):
    embedding1_list = []
    embedding2_list = []
    similarity_labels = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            embedding1 = embeddings[i]
            embedding2 = embeddings[j]

            # If the labels are the same, labeling the pair as 1 (similar)...
            if labels[i] == labels[j]:
                similarity_labels.append(1)
            else:
                # If the labels are different, labeling the pair as 0 (dissimilar)...
                similarity_labels.append(0)

            embedding1_list.append(embedding1)
            embedding2_list.append(embedding2)

    embedding1_tensor = torch.stack(embedding1_list)
    embedding2_tensor = torch.stack(embedding2_list)
    labels_tensor = torch.tensor(similarity_labels).to(labels.device)

    return embedding1_tensor, embedding2_tensor, labels_tensor