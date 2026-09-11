#!/bin/bash
# MPS latency sweep: 3 resolutions x (eager vs compile modes) + compiled precisions.
# usage: MODEL=lama-fourier ./run_mps_sweep.sh
MODEL=${MODEL:-lama-fourier}
cd /Users/zhukov/git/lama || exit 1
export TORCH_HOME=$(pwd) PYTHONPATH=$(pwd)
P=./venv-latest/bin/python; M=$(pwd)/$MODEL
echo "MODEL=$MODEL"
for ds in dataset16_lat80:1280x720 dataset16_lat80_1024x512:1024x512 dataset16_lat80_832x480:832x480; do
  d=${ds%%:*}; res=${ds##*:}; I=/Users/zhukov/$d
  echo "############ MPS $MODEL $res :: fp32 eager vs compile-default vs compile-reduce-overhead"
  env -u LAMA_FP16_FFT $P bench_latency_mps.py --indir $I --model $M --frames 16 --warmup 10 --iters 100 2>&1
  echo "############ MPS $MODEL $res compiled :: fp32 + fp16-amp (FFT in fp32)"
  env -u LAMA_FP16_FFT $P bench_latency_mps_fp16.py --indir $I --model $M --frames 16 --warmup 10 --iters 100 --compile --precisions fp32,fp16-amp 2>&1
  echo "############ MPS $MODEL $res compiled :: fp16-half, native half FFT (LAMA_FP16_FFT=1)"
  env LAMA_FP16_FFT=1 $P bench_latency_mps_fp16.py --indir $I --model $M --frames 16 --warmup 10 --iters 100 --compile --precisions fp16-half 2>&1
  echo "############ MPS $MODEL $res compiled :: fp16-half, fp32 FFT island"
  env -u LAMA_FP16_FFT $P bench_latency_mps_fp16.py --indir $I --model $M --frames 16 --warmup 10 --iters 100 --compile --precisions fp16-half 2>&1
done
echo BENCH_DONE_OK
