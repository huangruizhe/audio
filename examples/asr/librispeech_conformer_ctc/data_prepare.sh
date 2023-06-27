cd /fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc
# conda activate aligner2
conda activate aligner

K2_ROOT=/fsx/users/huangruizhe/k2
export PYTHONPATH=$K2_ROOT/k2/python:$PYTHONPATH
export PYTHONPATH=$K2_ROOT/build_release/lib:$PYTHONPATH
export PYTHONPATH=/fsx/users/huangruizhe/icefall_align2:$PYTHONPATH

mkdir lm

# ls /data/home/huangruizhe/.cache/torch/hub/torchaudio/decoder-assets/librispeech-4-gram/
# tokens.txt
# lexicon.txt
# lm.bin

python -c '''
import sentencepiece as spm
sp_model = spm.SentencePieceProcessor(model_file="./spm_unigram_1023.model")
with open("lm/tokens.txt", "w") as fout:
  for i in range(sp_model.vocab_size()):
    print(sp_model.id_to_piece(i), file=fout)
'''

cp /fsx/users/huangruizhe/icefall_align2/egs/librispeech/ASR/data/lang_bpe_500/words.txt lm/.
ln -s /fsx/users/huangruizhe/icefall_align2/egs/librispeech/ASR/data/lang_bpe_500/transcript_words.txt lm/.

# https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/prepare.sh#L192
# Make ``lang_bpe`` as in icefall
# Be careful: (1) the index of <blk> symbol is not 0 in torchaudio; (2) lowercase 
cd /fsx/users/huangruizhe/icefall_align2/egs/librispeech/ASR

dl_dir=$PWD/download
lang_dir=data/lang_bpe_torchaudio
mkdir -p $lang_dir
cat data/lang_phone/words.txt | tr '[:upper:]' '[:lower:]' > $lang_dir/words.txt

# if [ ! -f $lang_dir/transcript_words.txt ]; then
#   echo "Generate data for BPE training"
#   files=$(
#     find "$dl_dir/LibriSpeech/train-clean-100" -name "*.trans.txt"
#     find "$dl_dir/LibriSpeech/train-clean-360" -name "*.trans.txt"
#     find "$dl_dir/LibriSpeech/train-other-500" -name "*.trans.txt"
#   )
#   for f in ${files[@]}; do
#     cat $f | cut -d " " -f 2-
#   done > $lang_dir/transcript_words.txt
# fi
# wc $lang_dir/transcript_words.txt
# ln -s $(realpath data/lang_bpe_500/transcript_words.txt) $lang_dir/transcript_words.txt
cat data/lang_bpe_500/transcript_words.txt | tr '[:upper:]' '[:lower:]' > $lang_dir/transcript_words.txt

ln -s /fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_rnnt/spm_unigram_1023.model $lang_dir/bpe.model

# There will be errors, just ignore it. No need to resolve it
if [ ! -f $lang_dir/L_disambig.pt ]; then
  ./local/prepare_lang_bpe.py --lang-dir $lang_dir

  echo "Validating $lang_dir/lexicon.txt"
  ./local/validate_bpe_lexicon.py \
    --lexicon $lang_dir/lexicon.txt \
    --bpe-model $lang_dir/bpe.model
fi

# Manually modify data/lang_bpe_torchaudio/lexicon.txt
# to remove the 4 lines containing < symbols

# LM comes with librispeech:
ls /fsx/users/huangruizhe/icefall_align2/egs/librispeech/ASR/download/lm

# KenLM

# Installation
# pip install https://github.com/kpu/kenlm/archive/master.zip
git clone git@github.com:kpu/kenlm.git
mkdir -p build
cd build
cmake ..
make -j 4

export PATH=$PATH:/fsx/users/huangruizhe/kenlm/build/bin

# Usage:

# For training a ngram model with Kneser-Ney smoothing:
order=4
lang_dir=data/lang_bpe_torchaudio
text=/fsx/users/huangruizhe/icefall_align2/egs/librispeech/ASR/data/lang_bpe_torchaudio/transcript_words.txt
arpa=$lang_dir/kn.$order.arpa
lmplz -S 20% -o $order --discount_fallback <$text >$arpa

# When n is large, use the following.
# Otherwise: No space left on device in /tmp/M8SSql
lmplz -S 20% -o $order --discount_fallback --temp_prefix "/export/fs05/rhuang/kenlm_tmp" <$text >$arpa

# --limit_vocab_file
# https://github.com/kpu/kenlm/issues/177

build_binary $arpa ${arpa%.arpa}.binary

# query $arpa <$val_text
# query $arpa <$test_text
query ${arpa%.arpa}.binary <$val | tail -n4
query ${arpa%.arpa}.binary <$test | tail -n4

ls /fsx/users/huangruizhe/icefall_align2/egs/librispeech/ASR/data/lang_bpe_torchaudio/kn.4.bin

