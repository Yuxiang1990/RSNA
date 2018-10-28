#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# Written by Chao CHEN (chaochancs@gmail.com)
# Created On: 2017-08-11
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, alpha=1, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        nonzero_index = torch.nonzero(targets).squeeze()
        zero_index = torch.nonzero(targets == 0).squeeze()

        p_probs = inputs[nonzero_index]
        n_probs = 1.0 - inputs[zero_index]
        probs = torch.cat([p_probs, n_probs], dim=0)

        log_p = torch.log(probs)
        batch_loss = -self.alpha * (torch.pow((1 - probs), self.gamma)) * log_p + 1e-8
        # print(batch_loss)
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class FocalLoss_clf(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True, device="cpu"):
        super(FocalLoss_clf, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.alpha = self.alpha.to(device)
        self.size_average = size_average

    def forward(self, input, target):
        assert(input.dim() == 2)
        target = target.view(-1, 1)
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            # if self.alpha.type() != input.data.type():
            #     self.alpha = self.alpha.type_as(input.data)
            #     self.alpha = self.alpha.to(input.device)

            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt + 1e-8

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

