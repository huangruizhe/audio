#!/usr/bin/env bash
#$ -wd /exp/rhuang/meta/audio/examples/asr/librispeech_conformer_ctc2
#$ -V
#$ -N train
#$ -j y -o log/log-$JOB_NAME-$JOB_ID.out
#$ -M ruizhe@jhu.edu
#$ -m e
#$ -l mem_free=20G,h_rt=600:00:00,gpu=1
#$ -q gpu.q@@v100

# #$ -q gpu.q@@v100
# #$ -q gpu.q@@rtx

# #$ -l ram_free=300G,mem_free=300G,gpu=0,hostname=b*

# hostname=b19
# hostname=!c04*&!b*&!octopod*
# hostname
# nvidia-smi

# conda activate aligner5
export PATH="/home/hltcoe/rhuang/mambaforge/envs/aligner5/bin/":$PATH
module load cuda11.7/toolkit
module load cudnn/8.5.0.96_cuda11.x
module load nccl/2.13.4-1_cuda11.7
module load gcc/7.2.0
module load intel/mkl/64/2019/5.281

which python
nvcc --version
nvidia-smi

# k2
K2_ROOT=/exp/rhuang/meta/k2/
export PYTHONPATH=$K2_ROOT/k2/python:$PYTHONPATH # for `import k2`
export PYTHONPATH=$K2_ROOT/temp.linux-x86_64-cpython-310/lib:$PYTHONPATH # for `import _k2`
export PYTHONPATH=/exp/rhuang/meta/icefall:$PYTHONPATH

# # torchaudio recipe
# cd /exp/rhuang/meta/audio
# cd examples/asr/librispeech_conformer_ctc

# To verify SGE_HGR_gpu and CUDA_VISIBLE_DEVICES match for GPU jobs.
env | grep SGE_HGR_gpu
env | grep CUDA_VISIBLE_DEVICES
echo "hostname: `hostname`"
echo "current path:" `pwd`

export PYTHONPATH=/exp/rhuang/meta/audio/examples/asr/librispeech_conformer_ctc2:$PYTHONPATH

# train
# exp_dir=./experiments/char_k2_p0.0_0.2_0.7
# exp_dir=./experiments/char_k2_p0.0_0.0_0.0_lstm
# exp_dir=./experiments/char_k2_p0.0_0.1_0.5_lstm
# exp_dir=./experiments/char_k2_p0.2_0.0_0.0_lstm
# exp_dir=./experiments/char_pytorch_p0.0_0.0_0.0_lstm
# exp_dir=./experiments/char_k2_p0.0_0.0_0.0_stride1
# exp_dir=./experiments/char_k2_p0.2_0.0_0.0_stride1
# exp_dir=./experiments/char_k2_p0.0_0.0_0.0_stride4
# exp_dir=./experiments/char_k2_p0.2_0.0_0.0_stride4
# exp_dir=./experiments/phone_pytorch_p0.0_0.0_0.0
# exp_dir=./experiments/phone_k2_p0.0_0.0_0.0

echo
echo "exp_dir:" $exp_dir
echo

python train.py \
  --exp-dir $exp_dir \
  --librispeech-path /exp/rhuang/librispeech/download2 \
  --global-stats-path ./global_stats.json \
  --sp-model-path ./spm_unigram_1023.model \
  --epochs 10 \
  --nodes 1 \
  --gpus 1 \
  --train-config $exp_dir/train_config.yaml # --checkpoint-path experiments/char_k2_p0.0_0.0_0.0_stride4/checkpoints/epoch=4-step=32684.ckpt
