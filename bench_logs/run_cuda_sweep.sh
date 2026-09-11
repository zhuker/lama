#!/bin/bash
# CUDA latency sweep: 3 resolutions x eager/compile(reduce-overhead) x precisions.
# usage: MODEL=lama-fourier ITERS=200 COOLDOWN=0 TELEMETRY=0 ~/run_cuda_sweep.sh
MODEL=${MODEL:-lama-fourier}; ITERS=${ITERS:-200}; COOLDOWN=${COOLDOWN:-0}; TELEMETRY=${TELEMETRY:-0}
cd ~/git/lama || exit 1
export TORCH_HOME=$(pwd) PYTHONPATH=$(pwd)
if [ "$TELEMETRY" = 1 ]; then
  nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,utilization.gpu \
    --format=csv -l 5 > ~/gpu_telemetry_${MODEL}.csv 2>&1 &
  TELEM=$!; trap 'kill $TELEM 2>/dev/null' EXIT
fi
B="./venv-latest/bin/python bench_latency_cuda.py --model $(pwd)/$MODEL --frames 16 --warmup 20 --iters $ITERS"
gpu() { nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader; }
run() {
  local label=$1; shift
  echo "############ $(hostname) $MODEL $label   [gpu before: $(gpu)]"
  "$@" 2>&1 | grep -vE "FutureWarning|torch.jit.script|^\s*$"
  sync; [ "$COOLDOWN" -gt 0 ] && sleep "$COOLDOWN"
}
echo "MODEL=$MODEL ITERS=$ITERS COOLDOWN=$COOLDOWN TELEMETRY=$TELEMETRY"
for ds in dataset16_lat80:1280x720 dataset16_lat80_1024x512:1024x512 dataset16_lat80_832x480:832x480; do
  d=${ds%%:*}; res=${ds##*:}
  for m in eager compiled; do
    extra=""; [ "$m" = compiled ] && extra="--compile --compile-mode reduce-overhead"
    run "$res $m :: fp32 + fp16-amp (FFT in fp32)" env -u LAMA_FP16_FFT $B --indir ~/$d --precisions fp32,fp16-amp $extra
    run "$res $m :: fp16-half, native half FFT (LAMA_FP16_FFT=1)" env LAMA_FP16_FFT=1 $B --indir ~/$d --precisions fp16-half $extra
    run "$res $m :: fp16-half, fp32 FFT island" env -u LAMA_FP16_FFT $B --indir ~/$d --precisions fp16-half $extra
  done
done
echo BENCH_DONE_OK
