#!/bin/bash

# Created on 2018/12/10
# Author: Kaituo XU

stage=2

ngpu=1
dumpdir=data

sample_rate=8000
# Network config
L=40
N=500
hidden_size=500
num_layers=4
bidirectional=1
nspk=2
# Training config
epochs=30
half_lr=0
early_stop=0
max_norm=5
# minibatch
batch_size=128
num_workers=4
# optimizer
optimizer=adam
lr=1e-3
momentum=0
l2=0
# save and visualize
checkpoint=0
print_freq=10
visdom=0
visdom_epoch=0
visdom_id="TasNet Training"

# exp tag
tag="" # tag for managing experiments.

# Directory path of wsj0 including tr, cv and tt
data=/home/work_nfs/ktxu/data/wsj-mix/2speakers/wav8k/min/

. utils/parse_options.sh || exit 1;
. ./cmd.sh
. ./path.sh

if [ $stage -le 1 ]; then
  echo "Stage 1: Generating json files including wav path and duration"
  [ ! -d $dumpdir ] && mkdir $dumpdir
  preprocess.py --in-dir $data --out-dir $dumpdir --sample-rate $sample_rate
fi

if [ -z ${tag} ]; then
  expdir=exp/train_r${sample_rate}_L${L}_N${N}_h${hidden_size}_l${num_layers}_bi${bidirectional}_C${nspk}_epoch${epochs}_half${half_lr}_norm${max_norm}_bs${batch_size}_worker${num_workers}_${optimizer}_lr${lr}_mmt${momentum}_l2${l2}_cv10
else
  expdir=exp/train_${tag}
fi

if [ $stage -le 2 ]; then
  echo "Stage 2: Training"
  ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
    train.py \
    --train-dir $dumpdir/cv10/ \
    --valid-dir $dumpdir/cv10/ \
    --sample_rate $sample_rate \
    --L $L \
    --N $N \
    --hidden_size $hidden_size \
    --num_layers $num_layers \
    --bidirectional $bidirectional \
    --nspk $nspk \
    --epochs $epochs \
    --half-lr $half_lr \
    --early-stop $early_stop \
    --max-norm $max_norm \
    --batch-size $batch_size \
    --num-workers $num_workers \
    --optimizer $optimizer \
    --lr $lr \
    --momentum $momentum \
    --l2 $l2 \
    --save-folder ${expdir} \
    --checkpoint $checkpoint \
    --print-freq ${print_freq} \
    --visdom $visdom \
    --visdom_epoch $visdom_epoch \
    --visdom-id "$visdom_id"
fi
