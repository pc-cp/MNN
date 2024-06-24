from network.base_model import ModelBase_ResNet18
from network.heads import (
    BYOLPredictionHead, BYOLProjectionHead
)
import copy
from util.NNMemoryBankModule import NNMemoryBankModule
from util.utils import *
class MNN(nn.Module):
    def __init__(self, dim=128, K=4096, momentum=-1, topk=1, dataset='cifar10', bn_splits=8, symmetric=False, lamda=-1.0, random_lamda=False, norm_nn=False):
        super(MNN, self).__init__()

        self.dim = dim
        self.K = K
        self.momentum = momentum
        self.topk = topk
        self.symmetric = symmetric
        self.lamda = lamda
        self.random_lamda = random_lamda
        self.norm_nn = norm_nn
        # create the encoders
        self.net               = ModelBase_ResNet18(dataset=dataset, bn_splits=bn_splits)
        self.backbone_momentum = copy.deepcopy(self.net)

        self.projection_head = BYOLProjectionHead(input_dim=512, hidden_dim=2048, output_dim=dim)
        self.prediction_head = BYOLPredictionHead(input_dim=dim, hidden_dim=2048, output_dim=dim)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        self.memory_bank = NNMemoryBankModule(size=self.K, topk=self.topk).cuda()

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def contrastive_loss(self, im_q, im_k, labels, update=False):

        # compute query features
        z_q = self.projection_head(self.net(im_q))  # queries: NxC
        p_q = self.prediction_head(z_q)

        with torch.no_grad():  # no gradient to keys
            # shuffle
            im_k_, shuffle = batch_shuffle(im_k)
            z_k = self.projection_head_momentum(self.backbone_momentum(im_k_)).clone().detach()  # keys: NxC
            # undo shuffle
            z_k = batch_unshuffle(z_k, shuffle)

        batch_size, feature_dim = z_q.shape
        # Nearest Neighbour
        z_k_nn, purity, _, _ = self.memory_bank(querys=z_k, keys=z_k, update=update, labels=labels)

        # ================normalized==================
        p_q_norm = nn.functional.normalize(p_q, dim=1)
        z_k_norm = nn.functional.normalize(z_k, dim=1)

        if self.random_lamda:
            #  lamda in (0, 1). not include 0 and 1
            self.lamda = torch.rand(1).cuda()

        if self.norm_nn:
            z_k_topk_norm = z_k_norm.repeat(1, self.topk).reshape(-1, self.dim)
            # norm3, as in R2
            z_k_nn_norm = self.lamda*nn.functional.normalize(z_k_nn, dim=1)+(1-self.lamda)*z_k_topk_norm

            # z_k_nn_norm = nn.functional.normalize(self.lamda*nn.functional.normalize(z_k_nn, dim=1)+(1-self.lamda)*z_k_topk_norm, dim=1)
        else:
            z_k_repeat_topk = z_k.repeat(1, self.topk).reshape(-1, feature_dim)
            z_k_nn_norm = nn.functional.normalize(self.lamda*z_k_nn + (1-self.lamda)*z_k_repeat_topk, dim=1)


        dist_qk = 2 - 2 * torch.einsum('bc,kc->bk', [p_q_norm, z_k_norm])
        one_label = torch.eye(batch_size).cuda()
        # calculate distance between p_q and z_k_nn, has shape (batch_size, batch_size*topk)
        dist_qk_nn = 2 - 2 * torch.einsum('bc,kc->bk', [p_q_norm, z_k_nn_norm])
        pseudo_labels = generation_mask(batch=batch_size, topk=self.topk).cuda()

        loss = (torch.mul(dist_qk, one_label).sum(dim=1) + torch.mul(dist_qk_nn, pseudo_labels).sum(dim=1)/(self.topk)).mean()

        return loss, purity

    def forward(self, im1, im2, labels):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """
        # Updates parameters of `model_ema` with Exponential Moving Average of `model`
        update_momentum(model=self.net,             model_ema=self.backbone_momentum,        m=self.momentum)
        update_momentum(model=self.projection_head, model_ema=self.projection_head_momentum, m=self.momentum)

        if self.symmetric:  # symmetric loss
            loss_21, purity_21 = self.contrastive_loss(im2, im1, update=False, labels=labels)

        loss_12, purity_12 = self.contrastive_loss(im1, im2, update=True, labels=labels)
        loss = loss_12
        purity = purity_12
        # compute losss
        if self.symmetric:  # symmetric loss
            purity = (purity_12 + purity_21)*1.0/2
            loss = (loss_12 + loss_21)*1.0/2

        return loss, purity