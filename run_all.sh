#!/usr/bin/env bash
# All runs in the post: 4 sampling schemes x 3 seeds at width 256, plus widths 128 and 512,
# batch sizes 16, 64 and 128, and beta2 0.95, 0.9, 0.8 and 0.7, all with shuffling.
# Runs are spread over $NGPU GPUs (default 1) and launched in parallel; each needs about 1 GB of memory.
set -euo pipefail
NGPU=${NGPU:-1}
mkdir -p runs
i=0
launch() {  # name, extra args
  CUDA_VISIBLE_DEVICES=$((i % NGPU)) python train.py --out runs/$1.npz "${@:2}" > runs/$1.log 2>&1 &
  i=$((i + 1))
}
for s in 0 1 2; do
  for m in shuffle noshuffle reverse replace; do launch ${m}_d256_s$s --mode $m --seed $s; done
  for d in 128 512; do launch shuffle_d${d}_s$s --mode shuffle --d $d --seed $s; done
  for b in 16 64 128; do launch shuffle_bs${b}_s$s --mode shuffle --bs $b --seed $s; done
  for b2 in 0.95 0.9 0.8 0.7; do launch shuffle_b2${b2}_s$s --mode shuffle --beta2 $b2 --seed $s; done
done
wait
echo "done: $(ls runs/*.npz | wc -l) runs"
