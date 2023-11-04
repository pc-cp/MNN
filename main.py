import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from util.meter import *
from util.utils import *
from dataset import data_two
from dataset.data_public import *

# =========== Different algorithmic frameworks ===========
from network.MNN import MNN

import argparse
import os
import time
import datetime

def get_args_parser():
    parser = argparse.ArgumentParser('Pretrain pipeline', add_help=False)
    # ===================== network structure =====================
    parser.add_argument('--name', type=str, default='', help='Name of the algorithm used for pre-training.')
    parser.add_argument('--doc', type=str, default='practice makes perfect',
                        help='To describe what this training is about')
    parser.add_argument('--symmetric', default=False, action='store_true',
                        help='use a symmetric loss function that backprops to both crops')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dim', type=int, default=128, help='Dimension of embedded features')

    parser.add_argument('--cos_mom', default=False, action='store_true', help='cosine schedule for momentum')
    parser.add_argument('--cos_tem', default=False, action='store_true', help='cosine schedule for temperature')
    parser.add_argument('--momentum', type=float, default=0.0, help='momentum for teacher')
    parser.add_argument('--cos_lr', default=True, action='store_false', help='cosine schedule for learning rate')
    parser.add_argument('--begin_lr', type=float, default=0.0, help="""Initial value for the learning rate""")
    parser.add_argument('--base_lr', type=float, default=0.06,
                        help="""Base value (after linear warmup) for the learning rate""")
    parser.add_argument('--final_lr', default=0.0, type=float,
                        help="""Final value (after linear warmup) of the learning rate.""")
    parser.add_argument('--warmup_lr_epochs', default=5, type=int,
                        help='Number of warmup epochs for the learning rate (Default: 5).')
    parser.add_argument('--epochs', type=int, default=200)

    # ===================== data =====================
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--data_path', type=str, default='/mnt/data/dataset',
                        help='path of dataset (default: \'./dataset\')')
    parser.add_argument('--aug_numbers', type=int, default=2, help='The number of data augmentations required by the '
                                                                   'algorithm, typically 2')
    parser.add_argument('--weak', default=False, action='store_true', help='weak aug for teacher')

    # ===================== Parameters for the customization of specific algorithms =====================
    parser.add_argument('--queue_size', type=int, default=4096, help='Queue size')
    parser.add_argument('--topk', default=5, type=int, help='K-nearest neighbors')
    parser.add_argument('--lamda', type=float, default=0.5, help='ratio of synthesis')
    parser.add_argument('--random_lamda', default=False, action='store_true', help='whether use random number to lamda')

    # ===================== hardware setup =====================
    parser.add_argument('--gpuid', default='1', type=str, help='gpuid')
    parser.add_argument('--port', type=int, default=23456)
    parser.add_argument('--seed', type=int, default=1339)

    # ===================== Naming of the output file =====================
    parser.add_argument('--logdir', default='current', type=str, help='log')

    return parser

def train(train_loader, model, optimizer, lr_schedule, epoch, iteration_per_epoch, args):
    ce_losses = AverageMeter('CE', ':.4e')
    lr_ave = AverageMeter('LR', ':.4e')
    # Used for top-k neighbors computation purity
    purity_ave = AverageMeter('PUR', ':.4e')

    # switch to train mode
    model.train()
    start_time = time.time()

    for i, ((ims, labels)) in enumerate(train_loader):
        it = iteration_per_epoch * epoch + i

        for j, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]

        for i in range(len(ims)):
            ims[i] = ims[i].cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # print('labels: ', labels)
        if len(ims) == 2:
            loss, purity = model(ims[0], ims[1], labels=labels)
        else:
            print('error')

        # record loss and learning rate
        ce_losses.update(loss.item(), ims[0].size(0))
        lr_ave.update(lr_schedule[it].item(), ims[0].size(0))
        purity_ave.update(purity.item(), ims[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_time = time.time() - start_time

    return ce_losses.avg, purity_ave.avg, str(datetime.timedelta(seconds=int(epoch_time))), lr_ave.avg

# test using a knn monitor
def online_test(net, memory_data_loader, test_data_loader, args):
    net.eval()
    classes = args.num_classes
    total_top1, total_top5, total_num, feature_bank, target_bank = 0.0, 0.0, 0, [], []
    with torch.no_grad():
        # generate feature bank
        for i, (data, target) in enumerate(memory_data_loader):
            # for data, target in enumerate(memory_data_loader):
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            feature = net(data)
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            target_bank.append(target)
        # [D, N]: D代表每个图像的特征维数, N代表dataset的大小
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        feature_labels = torch.cat(target_bank, dim=0).t().contiguous()
        # [N]
        for i, (data, target) in enumerate(test_data_loader):
            # for data, target in enumerate(memory_data_loader):
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            feature = net(data)
            feature = F.normalize(feature, dim=1)
            # same with moco
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, 200, 0.1)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()

    return total_top1 / total_num * 100


def main():
    setup_seed(args.seed)

    # args.gpuid = '0'
    # args.name = 'moco'
    # args.tem = 0.2
    # args.aug_numbers = 2
    # args.symmetric = True
    # args.momentum = 0.99

    # args.topk = 3
    # args.weak = True
    # args.queue_size = 8
    # args.batch_size = 4
    # args.dataset = 'stl10'

    print(args)

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid


    if args.name == 'mnn':
        model = MNN(dataset=args.dataset, K=args.queue_size, momentum=args.momentum, topk=args.topk, symmetric=args.symmetric, lamda=args.lamda, random_lamda=args.random_lamda)
    else:
        print('The algorithm does not exist.')

    model = model.cuda()
    torch.backends.cudnn.benchmark = True

    if args.aug_numbers == 2:
        aug = 'data_two.'
        crop = 'TwoCrop'
    else:
        print('Error')

    if args.dataset == 'cifar10':
        if args.weak:
            dataset = eval(aug + 'CIFAR10Pair')(root=args.data_path, download=False,
                                                transform=get_contrastive_augment('cifar10'),
                                                weak_aug=get_weak_augment('cifar10'))
        else:
            dataset = eval(aug + 'CIFAR10Pair')(root=args.data_path, download=False,
                                                transform=get_contrastive_augment('cifar10'), weak_aug=None)
        memory_dataset = datasets.CIFAR10(root=args.data_path, download=False, transform=get_test_augment('cifar10'))
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=False,
                                        transform=get_test_augment('cifar10'))
        args.num_classes = 10
    elif args.dataset == 'stl10':
        if args.weak:
            dataset = eval(aug + 'STL10Pair')(root=args.data_path, download=False, split='train+unlabeled',
                                              transform=get_contrastive_augment('stl10'),
                                              # transform=get_weak_augment('stl10'), ablation weak/weak in MNN
                                              weak_aug=get_weak_augment('stl10'))
        else:
            dataset = eval(aug + 'STL10Pair')(root=args.data_path, download=False, split='train+unlabeled',
                                              transform=get_contrastive_augment('stl10'), weak_aug=None)
        memory_dataset = datasets.STL10(root=args.data_path, download=False, split='train',
                                        transform=get_test_augment('stl10'))
        test_dataset = datasets.STL10(root=args.data_path, download=False, split='test',
                                      transform=get_test_augment('stl10'))
        args.num_classes = 10
    elif args.dataset == 'tinyimagenet':
        if args.weak:
            dataset = TinyImageNet(root=args.data_path + '/tiny-imagenet-200', train=True,
                                                 transform=eval(aug + crop)(get_contrastive_augment('tinyimagenet'),
                                                                            get_weak_augment('tinyimagenet')))
        else:
            dataset = TinyImageNet(root=args.data_path + '/tiny-imagenet-200', train=True,
                                                 transform=eval(aug + crop)(get_contrastive_augment('tinyimagenet'),
                                                                            get_contrastive_augment('tinyimagenet')))
        memory_dataset = TinyImageNet(root=args.data_path + '/tiny-imagenet-200', train=True,
                                                    transform=get_test_augment('tinyimagenet'))
        test_dataset = TinyImageNet(root=args.data_path + '/tiny-imagenet-200', train=False,
                                                  transform=get_test_augment('tinyimagenet'))
        args.num_classes = 200
    else:
        if args.weak:
            dataset = eval(aug + 'CIFAR100Pair')(root=args.data_path, download=False,
                                                 transform=get_contrastive_augment('cifar100'),
                                                 weak_aug=get_weak_augment('cifar100'))
        else:
            dataset = eval(aug + 'CIFAR100Pair')(root=args.data_path, download=False,
                                                 transform=get_contrastive_augment('cifar100'), weak_aug=None)
        memory_dataset = datasets.CIFAR100(root=args.data_path, download=False, transform=get_test_augment('cifar100'))
        test_dataset = datasets.CIFAR100(root=args.data_path, train=False, download=False,
                                         transform=get_test_augment('cifar100'))
        args.num_classes = 100

    train_loader = DataLoader(dataset, shuffle=True, num_workers=6, pin_memory=True, batch_size=args.batch_size,
                              drop_last=True)
    memory_loader = DataLoader(memory_dataset, shuffle=False, num_workers=6, pin_memory=True,
                               batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, num_workers=6, pin_memory=True, batch_size=args.batch_size)
    iteration_per_epoch = train_loader.__len__()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=5e-4)
    lr_schedule = cosine_scheduler(
        base_value=args.base_lr * args.batch_size / 256.,  # linear scaling rule
        final_value=args.final_lr,
        epochs=args.epochs,
        niter_per_ep=iteration_per_epoch,
        start_warmup_value=args.begin_lr,
        warmup_epochs=args.warmup_lr_epochs,
    )

    checkpoint_path = 'checkpoints/' + args.name + '-{}-{}.pth'.format(args.dataset, args.logdir)
    print('checkpoint_path:', checkpoint_path)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print(checkpoint_path, 'found, start from epoch', start_epoch)
    else:
        start_epoch = 0
        print(checkpoint_path, 'not found, start from epoch 0')

    if start_epoch >= args.epochs-1:
        print('Done!')
        exit(0)

    model.train()
    best_acc = 0

    for epoch in range(start_epoch, args.epochs):
        train_loss, purity_ave, epoch_time, lr_ave = train(train_loader, model, optimizer, lr_schedule, epoch, iteration_per_epoch, args)
        cur_acc = online_test(model.net, memory_loader, test_loader, args)
        if cur_acc > best_acc:
            best_acc = cur_acc
        print(
            f'Epoch [{epoch}/{args.epochs}]: 200-NN-Best: {best_acc:.2f}, 200-NN: {cur_acc:.2f}, '
            f'Purity: {purity_ave:.4f}, loss: {train_loss:.8f}, time: {epoch_time}, lr: {lr_ave:.4f}')

        if epoch == args.epochs - 1:
            torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                }, checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Pretrain pipeline', parents=[get_args_parser()])
    args = parser.parse_args()
    main()