import logging
import pathlib
from argparse import ArgumentParser

import sentencepiece as spm
import math

import torch
import torchaudio
from lightning import ConformerCTCModule
from transforms import get_data_module

from config import load_config, update_config, save_config
import logging
from tokenizer_char import CharTokenizer
from tokenizer_char_boundary import CharTokenizerBoundary
import werpy
import itertools

logging.getLogger("lightning.pytorch").setLevel(logging.INFO)

logger = logging.getLogger()


def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())


def filter_repeat_letters(text):
    return ''.join(c[0] for c in itertools.groupby(text))


def run_eval(args, config):
    if config["model_unit"] == "bpe":
       sp_model = spm.SentencePieceProcessor(model_file=str(args.sp_model_path))
    elif config["model_unit"] == "char":
        sp_model = CharTokenizer()
    elif config["model_unit"] == "char_boundary":
        sp_model = CharTokenizerBoundary()

    # https://pytorch.org/audio/main/generated/torchaudio.models.decoder.ctc_decoder.html
    inference_args = {
        "inference_type": "greedy",
    }
    # inference_args = {
    #     "inference_type": "4gram",
    #     "nbest": 3,
    #     "beam_size": 1500,
    #     "beam_size_token": None,
    #     "beam_threshold": 50,
    #     "lm_weight": 3.23,
    #     "word_score": -0.26,
    #     # "unk_score": -math.inf,
    #     # "sil_score": 0,
    #     "lexicon": "/fsx/users/huangruizhe/icefall_align2/egs/librispeech/ASR/data/lang_bpe_torchaudio/lexicon.txt",
    #     "tokens": "/fsx/users/huangruizhe/icefall_align2/egs/librispeech/ASR/data/lang_bpe_torchaudio/tokens.txt",
    #     "lm": "/fsx/users/huangruizhe/icefall_align2/egs/librispeech/ASR/data/lang_bpe_torchaudio/kn.4.bin",
    #     "sil_token": "q",
    #     "blank_token": "q",
    # }

    model = ConformerCTCModule.load_from_checkpoint(args.checkpoint_path, sp_model=sp_model, inference_args=inference_args, config=config).eval()
    data_module = get_data_module(str(args.librispeech_path), str(args.global_stats_path), sp_model, config)

    if args.use_cuda:
        model = model.to(device="cuda")

    total_edit_distance = 0
    total_length = 0
    dataloader = data_module.test_dataloader()
    with torch.no_grad():
        for idx, (batch, sample) in enumerate(dataloader):
            # WER:
            actual = sample[0][2]
            actual = filter_repeat_letters(actual)
            predicted = model(batch)
            predicted = predicted.replace("<B>", "").replace("-", "").strip()
            predicted = filter_repeat_letters(predicted)
            # total_edit_distance += compute_word_level_distance(actual, predicted)
            total_edit_distance += werpy.summary(actual, predicted).iloc[:, :-3]
            # print(f"[{idx}][predicted]\t{predicted}")
            # print(f"[{idx}][actual   ]\t{actual}")

            # # CER:
            # actual = " ".join(list(sample[0][2].replace(" ", "")))
            # actual = filter_repeat_letters(actual)
            # predicted = model(batch)
            # predicted = " ".join(list(predicted.replace(" ", "").replace("<B>", "").replace("-", "")))
            # predicted = filter_repeat_letters(predicted)
            # # total_edit_distance += compute_word_level_distance(actual, predicted)
            # total_edit_distance += werpy.summary(actual, predicted).iloc[:, :-3]
            # # print(f"[{idx}][predicted]\t{predicted}")
            # # print(f"[{idx}][actual   ]\t{actual}")

            total_length += len(actual.split())
            if idx % 100 == 0:
                if type(total_edit_distance) is int:
                    logger.warning(f"Processed elem {idx}; WER: {total_edit_distance / total_length}")
                else:
                    logger.warning(f"Processed elem {idx}; WER: {total_edit_distance.iloc[0]['ld'] / total_edit_distance.iloc[0]['m']}")

    logger.warning(args.checkpoint_path)
    if type(total_edit_distance) is int:
        logger.warning(f"Final WER: {total_edit_distance / total_length}")
    else:
        total_edit_distance.at[0, 'wer'] = total_edit_distance.iloc[0]['ld'] / total_edit_distance.iloc[0]['m']
        logger.warning(f"Processed elem {idx}; WER: \n{total_edit_distance}")


def cli_main():
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        type=pathlib.Path,
        help="Path to checkpoint to use for evaluation.",
        required=True,
    )
    parser.add_argument(
        "--global-stats-path",
        default=pathlib.Path("global_stats.json"),
        type=pathlib.Path,
        help="Path to JSON file containing feature means and stddevs.",
    )
    parser.add_argument(
        "--librispeech-path",
        type=pathlib.Path,
        help="Path to LibriSpeech datasets.",
        required=True,
    )
    parser.add_argument(
        "--sp-model-path",
        type=pathlib.Path,
        help="Path to SentencePiece model.",
        required=True,
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        default=False,
        help="Run using CUDA.",
    )
    parser.add_argument(
        "--train-config",
        default=None,
        type=pathlib.Path,
        help="Path to config file.",
    )
    args = parser.parse_args()

    config = load_config(args.train_config)

    run_eval(args, config)


if __name__ == "__main__":
    cli_main()
