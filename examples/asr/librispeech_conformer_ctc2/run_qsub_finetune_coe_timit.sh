#!/usr/bin/env bash
#$ -wd /exp/rhuang/meta/audio/examples/asr/librispeech_conformer_ctc2
#$ -V
#$ -N finetune_timit
#$ -j y -o log/log-$JOB_NAME-$JOB_ID.out
#$ -M ruizhe@jhu.edu
#$ -m e
#$ -l mem_free=20G,h_rt=600:00:00,gpu=4
#$ -q gpu.q@@rtx

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
date

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
# exp_dir=./experiments/char_k2_p0.0_0.0_0.0_lstm
# exp_dir=./experiments/char_pytorch_p0.0_0.0_0.0_lstm
# exp_dir=./experiments/char_k2_p0.0_0.1_0.5_lstm

# epoch=1

# echo
# echo "exp_dir:" $exp_dir
# echo

# # align
# buckeye=/exp/rhuang/buckeye/datasets/Buckeye_Corpus2/temp3/
# python train.py \
#   --exp-dir $exp_dir \
#   --buckeye-path $buckeye \
#   --global-stats-path ./global_stats.json \
#   --sp-model-path ./spm_unigram_1023.model \
#   --epochs 20 \
#   --nodes 1 \
#   --gpus 1 \
#   --train-config $exp_dir/train_config.yaml \
#   --mode align \
#   --checkpoint-path $exp_dir/checkpoints/epoch=$epoch-*.ckpt

# # evaluate
# ali_pattern="ali_epoch=$epoch-*.pkl"

# echo ""
# echo "Word-level evaluation:"
# python /exp/rhuang/meta/audio/examples/asr/librispeech_conformer_ctc2/buckeye_evaluate.py \
#   --ref-dir $buckeye \
#   --ali-dir $exp_dir/ali/ \
#   --ali-pattern $ali_pattern \
#   --out-dir $exp_dir/temptemp \
#   --word-level

# echo ""
# echo "Phone-level evaluation:"
# python /exp/rhuang/meta/audio/examples/asr/librispeech_conformer_ctc2/buckeye_evaluate.py \
#   --ref-dir $buckeye \
#   --ali-dir $exp_dir/ali/ \
#   --ali-pattern $ali_pattern \
#   --out-dir $exp_dir/temptemp



# buckeye=/exp/rhuang/buckeye/datasets/Buckeye_Corpus2/temp3/
# buckeye=/exp/rhuang/timit/data/TEST/
buckeye=/exp/rhuang/timit/data/

# exp_dir=/exp/rhuang/meta/audio_ruizhe/zhaoheng/temp/dnn_phone_k2_p0.0_inter0.0_intra0.0_loop0.0_stride2
exp_dir=/exp/rhuang/meta/audio_ruizhe/zhaoheng/temp/dnn_phone_k2_p0.3_inter0.0_intra0.0_loop0.0_stride2

ngps=4
config=$exp_dir/train_config.yaml
# config=/exp/rhuang/meta/audio_ruizhe/zhaoheng/exp_0905_4/dnn_phone_k2_p0.0_inter0.0_intra0.0_loop0.0_stride2/train_config.yaml
# config=experiments/test//train_config.yaml
# config=experiments/test//train_config_stride1_dilation.yaml

# cp /exp/rhuang/meta/audio_ruizhe/zhaoheng/exp_0905_4/dnn_phone_k2_p0.3_inter0.0_intra0.0_loop0.0_stride2/train_config.yaml ./experiments/test/.
# mkdir $exp_dir
# cp experiments/test//train_config.yaml $exp_dir/.
# ls $exp_dir/train_config.yaml

echo
echo "exp_dir:" $exp_dir
echo

# finetune on buckeye
python train.py \
  --exp-dir $exp_dir/fttm \
  --buckeye-path $buckeye \
  --global-stats-path ./global_stats.json \
  --sp-model-path ./spm_unigram_1023.model \
  --epochs 20 --nodes 1 --gpus $ngps \
  --train-config $config \
  --mode train \
  --checkpoint-path $exp_dir/checkpoints/epoch=6-*.ckpt --dataset-name "TIMIT"

# --checkpoint-path /exp/rhuang/meta/audio_ruizhe/zhaoheng/temp/checkpoints_0.3/epoch=6-step=22881.ckpt
# --checkpoint-path /exp/rhuang/meta/audio_ruizhe/zhaoheng/temp/checkpoints_0.3/epoch=9-step=32687.ckpt

# # finetune on librispeech
# python train.py \
#   --exp-dir $exp_dir \
#   --librispeech-path /exp/rhuang/librispeech/download2 \
#   --global-stats-path ./global_stats.json \
#   --sp-model-path ./spm_unigram_1023.model \
#   --epochs 12 --nodes 1 --gpus $ngps \
#   --train-config $config \
#   --mode train \
#   --checkpoint-path $exp_dir/checkpoints/epoch=6-*.ckpt

# align
epoch=19
python train.py \
  --exp-dir $exp_dir/fttm \
  --buckeye-path $buckeye \
  --global-stats-path ./global_stats.json \
  --sp-model-path ./spm_unigram_1023.model \
  --epochs 40 \
  --nodes 1 \
  --gpus $ngps \
  --train-config $config \
  --mode align \
  --checkpoint-path $exp_dir/fttm/checkpoints/epoch=$epoch-*.ckpt --dataset-name "TIMIT"

# --checkpoint-path experiments/test/checkpoints/epoch=19-*.ckpt

# evaluate
ali_pattern="ali_epoch=$epoch-*.pkl"
# ali_pattern="ali_epoch=19-step=34417*.pkl"
# ali_pattern="ali_epoch=19-step=25130*.pkl"

echo ""
echo "Word-level evaluation:"
python /exp/rhuang/meta/audio/examples/asr/librispeech_conformer_ctc2/timit_evaluate.py \
  --ref-dir $buckeye \
  --ali-dir $exp_dir//fttm/ali_timit/ \
  --ali-pattern $ali_pattern \
  --out-dir $exp_dir/temptemp \
  --word-level

echo ""
echo "Phone-level evaluation:"
python /exp/rhuang/meta/audio/examples/asr/librispeech_conformer_ctc2/timit_evaluate.py \
  --ref-dir $buckeye \
  --ali-dir $exp_dir/fttm/ali_timit/ \
  --ali-pattern $ali_pattern \
  --out-dir $exp_dir/temptemp
