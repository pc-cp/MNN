#!/bin/bash

echo "pretrain mnn(cifar10_11)"
nohup python main.py \
--name mnn \
--symmetric \
--momentum 0.99 \
--dataset cifar10 \
--aug_numbers 2 \
--weak \
--queue_size 4096 \
--topk 5 \
--random_lamda \
--gpuid 1 \
--seed 1339 \
--logdir cifar10_11 \
>pretrain_output/cifar10_11 2>&1 &
wait
echo "evaluating mnn(cifar10_11)"
nohup python linear_eval.py \
--name mnn \
--dataset cifar10 \
--gpuid 1 \
--seed 1339 \
--logdir cifar10_11 \
>linear_eval_output/cifar10_11_1 2>&1 &
wait
echo "pretrain mnn(cifar100_11)"
nohup python main.py \
--name mnn \
--symmetric \
--momentum 0.99 \
--dataset cifar100 \
--aug_numbers 2 \
--weak \
--queue_size 4096 \
--topk 5 \
--random_lamda \
--gpuid 1 \
--seed 1339 \
--logdir cifar100_11 \
>pretrain_output/cifar100_11 2>&1 &
wait
echo "evaluating mnn(cifar100_11)"
nohup python linear_eval.py \
--name mnn \
--dataset cifar100 \
--gpuid 1 \
--seed 1339 \
--logdir cifar100_11 \
>linear_eval_output/cifar100_11_1 2>&1 &
wait

echo "pretrain mnn(stl10_11)"
nohup python main.py \
--name mnn \
--symmetric \
--momentum 0.996 \
--dataset stl10 \
--aug_numbers 2 \
--weak \
--queue_size 16384 \
--topk 5 \
--random_lamda \
--gpuid 1 \
--seed 1339 \
--logdir stl10_11 \
>pretrain_output/stl10_11 2>&1 &
wait
echo "evaluating mnn(stl10_11)"
nohup python linear_eval.py \
--name mnn \
--dataset stl10 \
--gpuid 1 \
--seed 1339 \
--logdir stl10_11 \
>linear_eval_output/stl10_11_1 2>&1 &
wait


echo "pretrain mnn(tin_11)"
nohup python main.py \
--name mnn \
--symmetric \
--momentum 0.996 \
--dataset tinyimagenet \
--aug_numbers 2 \
--weak \
--queue_size 16384 \
--topk 5 \
--random_lamda \
--gpuid 1 \
--seed 1339 \
--logdir tin_11 \
>pretrain_output/tin_11 2>&1 &
wait
echo "evaluating mnn(tin_11)"
nohup python linear_eval.py \
--name mnn \
--dataset tinyimagenet \
--gpuid 1 \
--seed 1339 \
--logdir tin_11 \
>linear_eval_output/tin_11_1 2>&1 &
wait
