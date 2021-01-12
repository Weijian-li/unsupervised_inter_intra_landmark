import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def matching_loss(pred, gt, loss_type="L1", mask=None):
    if isinstance(pred, list):
        loss = 0
        for each_pred in pred:
            each_pred = each_pred.squeeze()
            s = each_pred.size()
            dis = each_pred - gt
            if len(s) == 4:
                dis = torch.abs(dis).sum(3).sum(2).sum(1)
            else:
                dis = torch.abs(dis).sum(2).sum(1)
            loss += dis
        return loss

    s = pred.size()
    if loss_type == "L2":
        dis = pred - gt
        dis = (dis ** 2).sum(2).sum(1)
    elif loss_type == "L1":
        dis = pred - gt
        if len(s) == 4:
            dis = torch.abs(dis).sum(3).sum(2).sum(1)
        else:
            # dis = torch.abs(dis).sum(2) * mask
            dis = torch.abs(dis).sum(2)
            # dis = dis.mean(1)
            dis = dis.sum(1)
    return dis