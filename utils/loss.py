import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def focal_loss(labels, logits, alpha, gamma):
    """Compute focal loss between `logits` and ground truth `labels`.

    Args:
        labels: Float tensor of size [batch, num_classes].
        logits: Float tensor of size [batch, num_classes].
        alpha: Float tensor of size [batch_size] for per-example weighting.
        gamma: Float scalar modulating loss from hard and easy examples.

    Returns:
        focal_loss: Float32 scalar representing normalized total loss.
    """
    BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss
    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)
    focal_loss /= torch.sum(labels)
    return focal_loss


def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    """Compute Class Balanced Loss between `logits` and ground truth `labels`.

    Args:
        labels: Int tensor of size [batch].
        logits: Float tensor of size [batch, no_of_classes].
        samples_per_cls: Python list of size [no_of_classes].
        no_of_classes: Total number of classes.
        loss_type: One of "sigmoid", "focal", "softmax".
        beta: Hyperparameter for class balanced loss.
        gamma: Hyperparameter for focal loss.

    Returns:
        cb_loss: Float tensor representing class balanced loss.
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()
    weights = torch.tensor(weights, dtype=torch.float32).cuda()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)

    return cb_loss


def true_metric_loss(true, no_of_classes, scale=1):
    batch_size = true.size(0)
    true = true.view(batch_size, 1)
    true_labels = torch.cuda.LongTensor(true).repeat(1, no_of_classes).float()
    class_labels = torch.arange(no_of_classes).float().cuda()
    phi = (scale * torch.abs(class_labels - true_labels)).cuda()
    y = nn.Softmax(dim=1)(-phi)
    return y


def loss_function(output, labels, loss_type, expt_type, scale):
    if loss_type == 'oe':
        targets = true_metric_loss(labels, expt_type, scale)
        return torch.sum(-targets * F.log_softmax(output, -1), -1).mean()

    elif loss_type == 'focal':
        beta = 0.9999
        gamma = 2.0
        sample = torch.bincount(labels, minlength=expt_type).cpu()
        sample = np.where(sample == 0, 0.0001, sample)
        loss = CB_loss(labels, output, sample, expt_type, "focal", beta, gamma)
        return loss

    else:
        loss_fct = nn.CrossEntropyLoss()
        return loss_fct(output, labels)
