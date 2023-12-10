import torch
import torch.nn.functional as F


def mse_loss(predictions, labels):
    mse = torch.nn.MSELoss()

    loss = mse(predictions, labels)
    return loss


def stage_loss(predictions, labels, teachers_weights):
    loss_func = torch.nn.MSELoss()

    loss1 = 0.
    loss2 = 0.
    loss3 = 0.
    for key in labels.keys():
        loss1 += teachers_weights[key] * loss_func(predictions[0], labels[key][0])
        loss2 += teachers_weights[key] * loss_func(predictions[1], labels[key][1])
        loss3 += teachers_weights[key] * loss_func(predictions[2], labels[key][2])

    return loss1 + loss2 + loss3


def stage_loss_ae(predictions, labels):
    loss_func = torch.nn.MSELoss()
    loss = loss_func(predictions, labels)
    return  loss


def kl_loss(predictions, labels):
    loss_func = torch.nn.BCELoss()
    height = labels.size(1)
    width = labels.size(2)
    min = labels.view(labels.size(0),-1).min(1, keepdim=True)[0]
    max = labels.view(labels.size(0),-1).max(1, keepdim=True)[0]
    if min != max:
        new_labels = (labels.view(labels.size(0),-1)-min)/(max-min)
    else:
        new_labels = labels
    threshold = torch.quantile(new_labels,q=0.25)
    labels = (new_labels.view(labels.size(0),1,height,width) > threshold) * 1.
    loss1 = loss_func(predictions[0].view(labels.size(0),-1), F.adaptive_max_pool2d(labels, (1,1)).view(labels.size(0),-1))
    loss2 = loss_func(predictions[1].view(labels.size(0),-1), F.adaptive_max_pool2d(labels, (4,4)).view(labels.size(0),-1))
    loss3 = loss_func(predictions[2].view(labels.size(0),-1), F.adaptive_max_pool2d(labels, (16,16)).view(labels.size(0),-1))

    return loss1,loss2,loss3


def stage_loss_with_mask(predictions, labels):
    loss_func = torch.nn.MSELoss()

    loss1 = loss_func(predictions[0], F.adaptive_max_pool2d(labels, (1, 1)).squeeze(-1))
    loss2 = loss_func(predictions[1], F.adaptive_max_pool2d(labels, (4, 4)).unsqueeze(1))
    loss3 = loss_func(predictions[2], F.adaptive_max_pool2d(labels, (16, 16)).unsqueeze(1))

    # Hard labels on mask // could be done also soft
    # The loss func could also be replaced with others
    mask_lbl = torch.clone(labels)
    mask_lbl[mask_lbl > 0] = 1.
    loss_mask = loss_func(predictions[3], F.adaptive_max_pool2d(mask_lbl, predictions[3].shape[-2:]).squeeze(-1))

    return loss1 + loss2 + loss3 + loss_mask