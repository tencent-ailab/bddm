#!/bin/bash

## example usage: sh train.sh 0,1,2,3 conf.yml

cuda=$1
comma=${cuda//[^,]}
nproc=$((${#comma}+1))

CUDA_VISIBLE_DEVICES=$cuda python3 -m torch.distributed.launch \
	--nproc_per_node $nproc --master_port $RANDOM main.py \
	--command train --config $2
