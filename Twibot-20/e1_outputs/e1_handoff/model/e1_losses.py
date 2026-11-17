import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    embeddings,
    labels,
    temperature=0.07
):
    embeddings = F.normalize(embeddings, dim=1)
    labels = labels.contiguous().view(-1)

    similarity = torch.matmul(
        embeddings, embeddings.T
    ) / temperature

    batch_size = labels.shape[0]

    self_mask = torch.eye(
        batch_size,
        device=labels.device,
        dtype=torch.bool
    )

    positive_mask = (
        labels.unsqueeze(0) == labels.unsqueeze(1)
    ) & ~self_mask

    logits_mask = ~self_mask
    similarity = similarity.masked_fill(
        self_mask, float("-inf")
    )

    log_prob = similarity - torch.logsumexp(
        similarity, dim=1, keepdim=True
    )

    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0

    if not valid.any():
        return embeddings.sum() * 0.0

    mean_log_prob = (
        (log_prob.masked_fill(~positive_mask, 0).sum(dim=1))
        / positive_count.clamp(min=1)
    )

    return -mean_log_prob[valid].mean()


def combined_loss(logits, embeddings, labels, alpha=0.2):
    loss_cls = F.cross_entropy(logits, labels)
    loss_supcon = supervised_contrastive_loss(
        embeddings, labels
    )
    total = loss_cls + alpha * loss_supcon

    return total, loss_cls, loss_supcon
