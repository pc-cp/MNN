#!/bin/bash
# This file includes
# the hyperparameter settings for training and evaluating the individual algorithms.
# Note that you need to create new folders for each seed value in advance in the same directory as this file
# to store the file logs during pretraining and evaluation.

create_folders() {
    for seed in 1339 1340 1341; do
        mkdir -p "pretrain_output_$seed"
        mkdir -p "linear_eval_output_$seed"
    done
}

run_experiment() {
    local method=$1
    local dataset=$2
    local logdir=$3
    local seed=$4
    local additional_params=$5

    # Check if additional_params contains --tem
    if [[ $additional_params != *"--tem"* ]]; then
        additional_params="--tem 0.2 $additional_params"
    fi

    echo "pretrain ${method}(${logdir}) with seed ${seed}"
    start_time=$(date +%s)
    nohup python main.py \
        --name ${method} \
        --symmetric \
        --dataset ${dataset} \
        --aug_numbers 2 \
        --gpuid 2 \
        --seed ${seed} \
        --logdir ${logdir} \
        ${additional_params} \
        > "pretrain_output_${seed}/${logdir}" 2>&1 &
    wait
    end_time=$(date +%s)
    pretrain_duration=$((end_time - start_time))
    echo "Pretraining ${method}(${logdir}) took ${pretrain_duration} seconds"

    echo "evaluating ${method}(${logdir}) with seed ${seed}"
    start_time=$(date +%s)
    nohup python linear_eval.py \
        --name ${method} \
        --dataset ${dataset} \
        --gpuid 2 \
        --seed ${seed} \
        --logdir ${logdir} \
        > "linear_eval_output_${seed}/${logdir}_1" 2>&1 &
    wait
    end_time=$(date +%s)
    eval_duration=$((end_time - start_time))
    echo "Evaluating ${method}(${logdir}) took ${eval_duration} seconds"
}

main() {
    create_folders
    for seed in 1339 1340 1341; do

        # MNN
        run_experiment "mnn" "cifar10" "cifar10_11" ${seed} "--momentum 0.99 --queue_size 4096 --topk 5 --random_lamda --weak --norm_nn"
        run_experiment "mnn" "cifar100" "cifar100_11" ${seed} "--momentum 0.99 --queue_size 4096 --topk 5 --random_lamda --weak --norm_nn"
        run_experiment "mnn" "tinyimagenet" "tin_11" ${seed} "--momentum 0.996 --queue_size 16384 --topk 5 --random_lamda --weak --norm_nn"
        run_experiment "mnn" "stl10" "stl10_11" ${seed} "--momentum 0.996 --queue_size 16384 --topk 5 --random_lamda --weak --norm_nn"
    done
}

main

# nohup ./scripts.sh >> scripts.log 2>&1 &
# ps -ef | grep scripts.sh | grep -v grep