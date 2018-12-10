#!/bin/bash

# Created on 2018/12/10
# Author: Kaituo XU

stage=1

sample_rate=8000

# Directory path of wsj0 including tr, cv and tt
data=/home/work_nfs/ktxu/data/wsj-mix/2speakers/wav8k/min/

. ./path.sh
. utils/parse_options.sh || exit 1;

if [ $stage -le 1 ]; then
  echo "Generating json files including wav path and duration"
  [ ! -d data ] && mkdir data
  preprocess.py --in-dir $data --out-dir data --sample-rate $sample_rate
fi
