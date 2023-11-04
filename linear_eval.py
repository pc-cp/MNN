from dataset.data_public import *
from util.meter import *
from util.utils import *

from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
from network.base_model import *
from torchvision import datasets, transforms
import torch
import torch.backends.cudnn as cudnn

import time
import datetime
import argparse
import os
import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser('Linear evaluation pipeline', add_help=False)
    # ===================== network structure =====================
    parser.add_argument('--name', type=str, default='', help='Name of the algorithm used for pre-training.')
    parser.add_argument('--cos_lr', default=False, action='store_true', help='cosine decay mechanism for learning rate')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=30, help='the learning rate')

    # ===================== data =====================
    parser.add_argument('--data_path', type=str, default='/mnt/data/dataset',
                        help='path of dataset (default: \'./dataset\')')
    parser.add_argument('--dataset', type=str, default='cifar10')

    # ===================== hardware setup =====================
    parser.add_argument('--port', type=int, default=23456)
    parser.add_argument('--gpuid', default='0', type=str, help='gpuid')
    parser.add_argument('--seed', type=int, default=1339)

    parser.add_argument('--logdir', default='current', type=str, help='a part of checkpoint\'s name')
    parser.add_argument('--checkpoint', type=str, default='', help='Backbone to be evaluated')
    parser.add_argument('--save-dir', type=str, default='./results', help='path to save the t-sne image')

    return parser


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main():
    setup_seed(args.seed)
    # args.name = 'moco'
    # args.epochs = 100
    # args.batch_size = 256
    # args.logdir = 'cifar10_1'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    args.checkpoint = f'{args.name}-{args.dataset}-{args.logdir}.pth'

    if args.cos_lr:
        args.lr = 1
    else:
        # step
        args.lr = 30
    print(args)
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=args.data_path, download=True, transform=get_train_augment('cifar10'))
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True,
                                        transform=get_test_augment('cifar10'))
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=args.data_path, download=True, transform=get_train_augment('cifar100'))
        test_dataset = datasets.CIFAR100(root=args.data_path, train=False, download=True,
                                         transform=get_test_augment('cifar100'))
        num_classes = 100
    elif args.dataset == 'tinyimagenet':
        train_dataset = TinyImageNet(root=args.data_path + '/tiny-imagenet-200', train=True,
                                     transform=get_train_augment('tinyimagenet'))
        test_dataset = TinyImageNet(root=args.data_path + '/tiny-imagenet-200', train=False,
                                    transform=get_test_augment('tinyimagenet'))
        # train_dataset = TinyImageNet(root='../data/tiny-imagenet-200/train', transform=get_train_augment('tinyimagenet'))
        # test_dataset = TinyImageNet(root='../data/tiny-imagenet-200/val', transform=get_test_augment('tinyimagenet'))
        num_classes = 200
    else:
        train_dataset = datasets.STL10(root=args.data_path, download=True, split='train',
                                       transform=get_train_augment('stl10'))
        test_dataset = datasets.STL10(root=args.data_path, download=True, split='test',
                                      transform=get_test_augment('stl10'))
        num_classes = 10

    pre_train = ModelBase(dataset=args.dataset)
    prefix = 'net.'

    state_dict = torch.load('./checkpoints/' + args.checkpoint, map_location='cpu')['model']
    # print(state_dict)
    for k in list(state_dict.keys()):
        if not k.startswith(prefix):
            del state_dict[k]
        if k.startswith(prefix):
            state_dict[k[len(prefix):]] = state_dict[k]
            del state_dict[k]
    pre_train.load_state_dict(state_dict)
    model = LinearHead(pre_train, dim_in=512, num_class=num_classes)
    model = model.cuda()
    # model = DistributedDataParallel(model.to(local_rank), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=args.lr, momentum=0.9, weight_decay=0, nesterov=True)

    torch.backends.cudnn.benchmark = True

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True)

    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True)

    if args.cos_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * train_loader.__len__())
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)

    best_acc = 0
    best_acc5 = 0
    for epoch in range(args.epochs):
        # ---------------------- Train --------------------------
        losses = AverageMeter('Loss', ':.4e')
        lr_ave = AverageMeter('LR', ':.4e')

        model.eval()
        start_time = time.time()

        for i, (image, label) in enumerate(train_loader):
            image = image.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            out = model(image)
            loss = F.cross_entropy(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), image.size(0))
            lr_ave.update(scheduler.get_last_lr()[-1], image[0].size(0))

            if args.cos_lr:
                scheduler.step()

        if ~args.cos_lr:
            scheduler.step()

        # ---------------------- Test --------------------------
        model.eval()

        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        targets_list = list()
        outputs_list = list()

        with torch.no_grad():
            for i, (image, label) in enumerate(test_loader):
                image = image.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

                # compute output
                output = model(image)

                # record t-SNE
                outputs_np = output.data.cpu().numpy()
                targets_np = label.data.cpu().numpy()
                targets_list.append(targets_np[:, np.newaxis])
                outputs_list.append(outputs_np)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, label, topk=(1, 5))
                top1.update(acc1[0], image.size(0))
                top5.update(acc5[0], image.size(0))

        epoch_time = time.time() - start_time

        sum1, cnt1, sum5, cnt5 = top1.sum, top1.count, top5.sum, top5.count

        top1_acc = sum1.float() / float(cnt1)
        top5_acc = sum5.float() / float(cnt5)

        best_acc = max(top1_acc, best_acc)
        best_acc5 = max(top5_acc, best_acc5)

        # if epoch==0 or epoch > 80:
        # tsne_plot(args.save_dir, np.concatenate(targets_list, axis=0), np.concatenate(outputs_list, axis=0).astype(np.float64), epoch)
        print(
            f'Epoch [{epoch}/{args.epochs}]: Acc-1-Best: {best_acc:.3f}, Acc-1: {top1_acc:.3f}, '
            f'Acc-5-Best: {best_acc5:.3f}, Acc-5: {top5_acc:.3f},'
            f'loss: {losses.avg:.4f}, lr: {lr_ave.avg:.4f}, time: {str(datetime.timedelta(seconds=int(epoch_time)))}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Linear evaluation pipeline', parents=[get_args_parser()])
    args = parser.parse_args()
    main()
