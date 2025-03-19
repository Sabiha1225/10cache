
import argparse

import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from deepspeed.accelerator import get_accelerator
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
import deepspeed.comm as dist
import time
import evaluate

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AdamW, get_scheduler
import csv
import pandas as pd

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "facebook/opt-13b"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=16, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["test"], batch_size=16, collate_fn=data_collator
)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

def add_argument():
    parser = argparse.ArgumentParser(description="CIFAR")

    # For train.
    parser.add_argument(
        "-e",
        "--epochs",
        default=1,
        type=int,
        help="number of total epochs (default: 30)",
    )
    parser.add_argument(
        "--param_count_gpu",
        default=10,
        type=int,
        help="number of total param in GPU (default: 10)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=2000,
        help="output logging information at a given interval",
    )

    # For mixed precision training.
    parser.add_argument(
        "--dtype",
        default="fp16",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="Datatype used for training",
    )

    # For ZeRO Optimization.
    #Stage 0, 1, 2, and 3 refer to disabled, optimizer state partitioning, and optimizer+gradient state partitioning, and optimizer+gradient+parameter partitioning,
    parser.add_argument(
        "--stage",
        default=3,
        type=int,
        choices=[0, 1, 2, 3],
        help="Datatype used for training",
    )

    #For MoE (Mixture of Experts).
    parser.add_argument(
        "--moe",
        default=False,
        action="store_true",
        help="use deepspeed mixture of experts (moe)",
    )
    parser.add_argument(
        "--moe-param-group",
        default=False,
        action="store_true",
        help="(moe) create separate moe param groups, required when using ZeRO w. MoE",
    )

    # Include DeepSpeed configuration arguments.
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args


def create_moe_param_groups(model):
    """Create separate parameter groups for each expert."""
    parameters = {"params": [p for p in model.parameters()], "name": "parameters"}
    return split_params_into_different_moe_groups_for_optimizer(parameters)


def get_ds_config(args):
    """Get the DeepSpeed configuration dictionary."""
    ds_config = {
        "train_batch_size": 16,
        "steps_per_print": 2000,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.001,
                "betas": [0.8, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7,
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 0.001,
                "warmup_num_steps": 1000,
            },
        },
        "bf16": {"enabled": args.dtype == "bf16"},
        "fp16": {
            "enabled": args.dtype == "fp16",
            "fp16_master_weights_and_grads": False,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 3,
            "min_loss_scale": 1e-10,
            "initial_scale_power": 34,
        },
        "wall_clock_breakdown": True,
        "zero_optimization": {
            "stage": args.stage,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 50000000,
            "reduce_bucket_size": 50000000,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "offload_optimizer": {
                "device": "nvme",
                "nvme_path": "/mnt/nvme/",
                "pin_memory": True,
                "ratio": 1.0,
                "buffer_count": 5,
                "fast_init": False
            },
            "offload_param": {
                "device": "nvme",
                "nvme_path": "/mnt/nvme/",
                "pin_memory": True,
                "buffer_count": 5,
                "buffer_size": 3e9,
                "max_in_cpu": 0
            }
        },
        "comms_logger": {
            "enabled": True,
            "verbose": False,
            "prof_all": True,
            "debug": False
        },
    }
    return ds_config



def main(args):
    # Initialize DeepSpeed distributed backend.
    deepspeed.init_distributed()

    if torch.distributed.get_rank() != 0:
        torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        torch.distributed.barrier()

    net = model

    # Get list of parameters that require gradients.
    parameters = filter(lambda p: p.requires_grad, net.parameters())

    # If using MoE, create separate param groups for each expert.
    if args.moe_param_group:
        parameters = create_moe_param_groups(net)

    ds_config = get_ds_config(args)
    model_engine, optimizer, __, __ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=parameters,
        #training_data=train_dataloader,
        config=ds_config,
    )

    # Get the local device name (str) and local rank (int).
    local_device = get_accelerator().device_name(model_engine.local_rank)
    local_rank = model_engine.local_rank

    # For float32, target_dtype will be None so no datatype conversion needed.
    target_dtype = None
    if model_engine.bfloat16_enabled():
        target_dtype = torch.bfloat16
    elif model_engine.fp16_enabled():
        target_dtype = torch.half

    # Define the Classification Cross-Entropy loss function.
    criterion = nn.CrossEntropyLoss()

    num_iter = 10
    model_engine.train()
    #t0 = time.time()
    warm_up_epoch = 1
    for epoch in range(warm_up_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            data = {k: v.to(local_device) for k, v in data.items()}
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            labels = data["labels"]
            outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=None)
            loss = criterion(outputs.logits, labels)

            model_engine.backward(loss)
            model_engine.step()
            break
    print("Finished Training warmup")
    
    model_engine.set_prefetch_table_warmup()
    model_engine.preallocate_memory()

    model_engine.train()
    temp = 1
    torch.cuda.synchronize()
    t0 = time.time()
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            data = {k: v.to(local_device) for k, v in data.items()}
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            labels = data["labels"]
            outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            model_engine.backward(loss)
            model_engine.step()
    torch.cuda.synchronize()
    t1 = time.time()
    training_time = t1 - t0
    print(f"Training Time taken: {training_time / args.epochs} s")
    print("Finished Training")
    with open("/home/sabiha/deepspeed_example/training_time_eval.txt", 'a') as file:
        file.write(f"Training Time taken smart-cache with 13b opt model: {training_time / args.epochs} s \n")


if __name__ == "__main__":
    args = add_argument()
    main(args)