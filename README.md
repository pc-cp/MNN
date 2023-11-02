# MNN: Mixed Nearest Neighbors for Self-Supervised Learning

This is an PyTorch implementation of MNN proposed by our paper [MNN: Mixed Nearest-Neighbors for Self-Supervised Learning](https://arxiv.org/abs/2311.00562). If you find this repo useful, welcome 🌟🌟🌟✨.

> ![figure1](./figures/mnn.png "MNN_overview")
> 

## Requirements
To install requirements:

 ```setup
 # name: d2lpy39
 conda env create -f environment.yml
 ```

## Training
(You need to create the directory './stdout', you can also omit '>stdout/*' so that you can run these commands directly.)
To train the model(s) in the paper, run those commands:

```train
nohup python main.py --name mnn --momentum 0.99 --symmetric --weak --topk 5 --dataset cifar10 --gpuid 0 --logdir cifar10_00 --aug_numbers 2 --random_lamda >stdout/cifar10_00 2>&1 &
nohup python main.py --name mnn --momentum 0.99 --symmetric --weak --topk 5 --dataset cifar100 --gpuid 0 --logdir cifar100_00 --aug_numbers 2 --random_lamda >stdout/cifar100_00 2>&1 &
nohup python main.py --name mnn --momentum 0.996 --symmetric --weak --topk 5 --dataset tinyimagenet --gpuid 0 --logdir tinyimagenet_00 --aug_numbers 2 --queue_size 16384 --random_lamda >stdout/tinyimagenet_00 2>&1 &

```

## Evaluation

To evaluate our model on CIFAR10/100 and Tiny-imagenet, run:
```eval
nohup python linear_eval.py --name mnn --dataset cifar10 --gpuid 0 --logdir cifar10_00 --seed 1339  >stdout/cifar10_00_01 2>&1 &
nohup python linear_eval.py --name mnn --dataset cifar100 --gpuid 0 --logdir cifar100_00 --seed 1339  >stdout/cifar100_00_01 2>&1 &
nohup python linear_eval.py --name mnn --dataset tinyimagenet --gpuid 0 --logdir tinyimagenet_00 --seed 1339  >stdout/tinyimagenet_00_01 2>&1 &

```

## Pre-trained Models

You can download pretrained models here:

- [this link](https://drive.google.com/file/d/1KkRLqGOGvo00mlETowRcL6b38I0lr2Ep/view?usp=sharing) trained on three datasets.
- Download and place in the **"./checkpoints"** directory

## Results

Our model achieves the following performance:

### Image Classification on four datasets

| -             | CIFAR-10  | CIFAR-100 | Tiny ImageNet |
|---------------|-----------|-----------|---------------|
| MSF           | 90.19     | 59.22     | 42.68         |
| **MNN(Ours)** | **91.47** | **67.56** | **50.70**     |
## Contributors and Contact
>📋  If there are any questions, feel free to contact with the authors.
