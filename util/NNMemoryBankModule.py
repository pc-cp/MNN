""" Nearest Neighbour Memory Bank Module """

# Copyright (c) 2021. Lightly AG and its affiliates.
# All Rights Reserved

import torch

from util.MemoryBankModule import MemoryBankModule

class NNMemoryBankModule(MemoryBankModule):
    """Nearest Neighbour Memory Bank implementation
        This code was taken and adapted from here:
        https://github.com/lightly-ai/lightly/blob/master/lightly/models/modules/nn_memory_bank.py
        https://github.com/ChongjianGE/SNCLR/blob/main/snclr/nn_memory_norm_bank_multi_keep.py

    This class implements a nearest neighbour memory bank as described in the
    NNCLR paper[0]. During the forward pass we return the nearest neighbour
    from the memory bank. But we improve the class so that it returns the
    neighbors of the topk of the query sample instead of 1.

    [0] NNCLR, 2021, https://arxiv.org/abs/2104.14548

    Attributes:
        size:
            Number of keys the memory bank can store. If set to 0, memory bank is not used.
        topk:
            Number of neighbors of the query sample

    Examples:
        >>> memory_bank = NNMemoryBankModule(size=2 ** 16, topk=5)
        >>> z_k_nn, purity, _, _ = self.memory_bank(querys=z_k, keys=z_k, update=update, labels=labels)
    """

    def __init__(self, size: int = 2**16, topk: int = 1):
        super(NNMemoryBankModule, self).__init__(size)
        self.size = size
        self.topk = topk
    # Using query to find top-K neighbors in a bank composed of keys
    def forward(self, querys: torch.Tensor, keys: torch.Tensor, update: bool = False, labels = None):
        """Returns top-K neighbors of query that come from a bank composed of keys

        Args:
            querys: Tensors that need to find their nearest neighbors
            keys:   Sharing labels with query and may be in the queue if update is True
                Usually querys and keys are the same. Here querys is used to find its top-k neighbors in the memory
                bank and add keys to the memory bank after the find is done.

            labels: The true label shared by the current batch of querys and keys, which is used to calculate purity
                    and may also be queued if update is True
            update: If `True` updated the memory bank by adding keys and labels to it
        """

        # If update is True, enqueue the keys and labels, otherwise we just return the keys, memory bank and labels bank
        keys, bank, bank_labels = super(NNMemoryBankModule, self).forward(output=keys, labels=labels, update=update)

        bank = bank.to(keys.device).t() # [feature_dim, size] -> [size, feature_dim]
        bank_labels = bank_labels.to(keys.device)

        query_normed = torch.nn.functional.normalize(querys, dim=1)
        bank_normed = torch.nn.functional.normalize(bank, dim=1)
        similarity_matrix = torch.einsum("nd,md->nm", query_normed, bank_normed)

        _, index_nearest_neighbours = similarity_matrix.topk(self.topk, dim=1, largest=True)

        batch_size, feature_dim = querys.shape
        current_batch_labels = labels.unsqueeze(1).expand(batch_size, self.topk)
        labels_queue = bank_labels.clone().detach()
        labels_queue = labels_queue.unsqueeze(0).expand((batch_size, self.size))
        labels_queue = torch.gather(labels_queue, dim=1, index=index_nearest_neighbours)
        matches = (labels_queue == current_batch_labels).float()
        purity = (matches.sum(dim=1) / self.topk).mean()

        # nearest_neighbours.shape: [batch_size*topk, feature_dim]
        nearest_neighbours = torch.index_select(bank, dim=0, index=index_nearest_neighbours.reshape(-1))
        # nearest_neighbours.shape: [batch_size*(topk+1), feature_dim]
        add_output_to_nearest_neighbours = torch.cat((querys.unsqueeze(dim=1),
                                                      nearest_neighbours.reshape(batch_size, self.topk, feature_dim)),
                                                     dim=1).reshape(-1, feature_dim)

        return nearest_neighbours, purity, bank, add_output_to_nearest_neighbours