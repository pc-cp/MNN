import torch

from network.base_model import ModelBase_ResNet18
from network.heads import (
    BYOLPredictionHead, BYOLProjectionHead
)
import copy
from util.NNMemoryBankModule import NNMemoryBankModule
from util.utils import *
class MNN(nn.Module):
    def __init__(self, dim=128, K=4096, topk=1, dataset='cifar10', bn_splits=8, symmetric=False, lamda=-1.0, norm_nn=False, random_lamda=False):
        super(MNN, self).__init__()

        self.K = K
        self.topk = topk
        self.symmetric = symmetric
        self.dim = dim
        self.lamda = lamda
        self.norm_nn = norm_nn
        self.random_lamda = random_lamda
        # create the encoders
        self.net               = ModelBase_ResNet18(dataset=dataset, bn_splits=bn_splits)
        self.backbone_momentum = copy.deepcopy(self.net)

        self.projection_head = BYOLProjectionHead(input_dim=512, hidden_dim=2048, output_dim=dim)
        self.prediction_head = BYOLPredictionHead(input_dim=dim, hidden_dim=2048, output_dim=dim)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        self.memory_bank = NNMemoryBankModule(size=self.K, topk=self.topk).cuda()

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def _generation_mask(self, batch_size):
        '''
        generation_mask(3)
    Out[5]:
        tensor([[1., 1., 0., 0., 0., 0.],
                [0., 0., 1., 1., 0., 0.],
                [0., 0., 0., 0., 1., 1.]])
        '''
        mask_ = torch.eye(batch_size)
        mask = mask_.repeat(self.topk, 1).reshape(self.topk, batch_size, -1).permute(2, 1, 0).reshape(batch_size, self.topk*batch_size)
        return mask

    def contrastive_loss(self, im_q, im_k, labels, epoch, update=False):
        # compute query features
        z_q = self.projection_head(self.net(im_q))  # queries: NxC
        p_q = self.prediction_head(z_q)

        with torch.no_grad():  # no gradient to keys
            # shuffle
            im_k_, shuffle = batch_shuffle(im_k)
            z_k = self.projection_head_momentum(self.backbone_momentum(im_k_)).clone().detach()  # keys: NxC
            # undo shuffle
            z_k = batch_unshuffle(z_k, shuffle)

        batch_size, _ = z_q.shape
        # Nearest Neighbour
        z_k_nn, purity = self.memory_bank(querys=z_k, keys=z_k, update=update, labels=labels)

        # ================normalized==================
        p_q_norm = nn.functional.normalize(p_q, dim=1)
        z_k_norm = nn.functional.normalize(z_k, dim=1)

        if self.random_lamda:
            #  lamda in (0, 1). not include 0 and 1
            self.lamda = torch.rand(1).cuda()

        if self.norm_nn:
            z_k_topk_norm = z_k_norm.repeat(1, self.topk).reshape(-1, self.dim)
            z_k_nn_norm = nn.functional.normalize(self.lamda*nn.functional.normalize(z_k_nn, dim=1)+(1-self.lamda)*z_k_topk_norm, dim=1)
        else:
            z_k_topk = z_k.repeat(1, self.topk).reshape(-1, self.dim)
            z_k_nn_norm = nn.functional.normalize(self.lamda*z_k_nn+(1-self.lamda)*z_k_topk, dim=1)

        # calculate distance between p_q and z_k_nn, has shape (batch_size, batch_size*topk)
        dist_qk_nn = 2 - 2 * torch.einsum('bc,kc->bk', [p_q_norm, z_k_nn_norm])
        labels = self._generation_mask(batch_size=batch_size).cuda()

        one_label = torch.eye(batch_size).cuda()
        dist_qk = 2 - 2 * torch.einsum('bc,kc->bk', [p_q_norm, z_k_norm])

        loss = (torch.mul(dist_qk_nn, labels).sum(dim=1)/(self.topk) + torch.mul(dist_qk, one_label).sum(dim=1)).mean()

        return loss, purity

    def forward(self, im1, im2, labels, momentum, epoch, tem):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """
        # Updates parameters of `model_ema` with Exponential Moving Average of `model`
        update_momentum(model=self.net, model_ema=self.backbone_momentum, m=momentum)
        update_momentum(model=self.projection_head, model_ema=self.projection_head_momentum, m=momentum)

        loss_12, purity_12 = self.contrastive_loss(im1, im2, update=True, labels=labels, epoch=epoch)
        loss = loss_12
        purity = purity_12

        # compute loss
        if self.symmetric:  # symmetric loss
            loss_21, purity_21 = self.contrastive_loss(im2, im1, update=False, labels=labels, epoch=epoch)
            purity = (purity_12 + purity_21) / 2
            loss = (loss_12 + loss_21) / 2


        return loss, purity