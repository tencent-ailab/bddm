#!/bin/bash

## example usage (single GPU): sh schedule.sh 0 conf.yml

cuda=$1

CUDA_VISIBLE_DEVICES=$cuda python3 main.py \
	--command schedule --config $2
