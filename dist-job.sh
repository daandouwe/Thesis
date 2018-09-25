#!/usr/bin/env bash

NPROCS=2

# Single node example.
python -m torch.distributed.launch --nproc_per_node=$NPROCS \
    dist-test.py --size $NPROCS

# Multi-node example.
# python -m torch.distributed.launch --nproc_per_node=$NPROCS \
    # --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=1234 \
    # dist-test.py --size $NPROCS
