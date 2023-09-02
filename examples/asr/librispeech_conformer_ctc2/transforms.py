import json
import math
from functools import partial
from typing import List

import sentencepiece as spm
import torch
import torchaudio
from data_module import LibriSpeechDataModule
from lightning import Batch

import torchaudio.transforms as T
from additive_noise import AddNoise
from torchaudio.prototype.datasets import Musan


_decibel = 2 * 20 * math.log10(torch.iinfo(torch.int16).max)
_gain = pow(10, 0.05 * _decibel)

_spectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, n_mels=80, hop_length=160)
_speed_perturb_transform = torchaudio.transforms.SpeedPerturbation(orig_freq=16000, factors=[0.9, 1.0, 1.1])

# # TODO: fbank: 
# # https://github.com/lhotse-speech/lhotse/issues/14
# # https://github.com/lhotse-speech/lhotse/blob/master/test/features/test_kaldi_features.py
# from lhotse import TorchaudioFbank
# from lhotse.features.kaldi.extractors import Fbank
# fbank = Fbank()
# fbank_ta = TorchaudioFbank()

musan_path = "/checkpoints/huangruizhe/datasets/musan/"   # pytorch cluster
# musan_path = "/private/home/huangruizhe/datasets/musan/"  # fair cluster
# subsets = ["noise"]  # ["music", "noise", "speech"]
# musan = Musan(musan_path, subsets)
# _additive_noise_transform = AddNoise(musan, snr=(10, 20), p=0.5)
_additive_noise_transform = None

def _piecewise_linear_log(x):
    x = x * _gain
    x[x > math.e] = torch.log(x[x > math.e])
    x[x <= math.e] = x[x <= math.e] / math.e
    return x


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class GlobalStatsNormalization(torch.nn.Module):
    def __init__(self, global_stats_path):
        super().__init__()

        with open(global_stats_path) as f:
            blob = json.loads(f.read())

        self.mean = torch.tensor(blob["mean"])
        self.invstddev = torch.tensor(blob["invstddev"])

    def forward(self, input):
        return (input - self.mean) * self.invstddev


def _extract_labels(sp_model, samples: List):
    targets = [sp_model.encode(sample[2].lower()) for sample in samples]
    lengths = torch.tensor([len(elem) for elem in targets]).to(dtype=torch.int32)
    targets = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(elem) for elem in targets],
        batch_first=True,
        padding_value=1.0,
    ).to(dtype=torch.int32)
    return targets, lengths


def _extract_features(data_pipeline, samples: List, speed_perturbation=False, musan_noise=False):
    if speed_perturbation:
        samples = [_speed_perturb_transform(sample[0].squeeze()) for sample in samples]

    if musan_noise:
        total_length = sum([sample[0].size(-1) for sample in samples])
        _additive_noise_transform.fetch_noise_batch(total_length)
        samples = [_additive_noise_transform(sample[0].squeeze()) for sample in samples]

    mel_features = [_spectrogram_transform(sample[0].squeeze()).transpose(1, 0) for sample in samples]
    features = torch.nn.utils.rnn.pad_sequence(mel_features, batch_first=True)
    features = data_pipeline(features)
    lengths = torch.tensor([elem.shape[0] for elem in mel_features], dtype=torch.int32)
    return features, lengths


# def _extract_features_train(data_pipeline, samples: List):
#     samples = [_speed_perturb_transform(sample[0].squeeze()) for sample in samples]
    
#     total_length = sum([sample[0].size(-1) for sample in samples])
#     _additive_noise_transform.fetch_noise_batch(total_length)
#     samples = [_additive_noise_transform(sample[0].squeeze()) for sample in samples]
    
#     mel_features = [_spectrogram_transform(sample[0].squeeze()).transpose(1, 0) for sample in samples]
#     features = torch.nn.utils.rnn.pad_sequence(mel_features, batch_first=True)
#     features = data_pipeline(features)
#     lengths = torch.tensor([elem.shape[0] for elem in mel_features], dtype=torch.int32)
#     return features, lengths


class TrainTransform:
    def __init__(self, global_stats_path: str, sp_model, config: dict):
        self.sp_model = sp_model

        self.config = config
        if config["specaug_conf"]["new_spec_aug_api"]:
            spec_aug_transform = T.SpecAugment(
                n_time_masks=config["specaug_conf"]["n_time_masks"],
                time_mask_param=config["specaug_conf"]["time_mask_param"],
                p=config["specaug_conf"]["p"],
                n_freq_masks=config["specaug_conf"]["n_freq_masks"],
                freq_mask_param=config["specaug_conf"]["freq_mask_param"],
                iid_masks=config["specaug_conf"]["iid_masks"],
                zero_masking=config["specaug_conf"]["zero_masking"],
            )
            self.train_data_pipeline = torch.nn.Sequential(
                FunctionalModule(_piecewise_linear_log),
                GlobalStatsNormalization(global_stats_path),
                FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
                spec_aug_transform,
                FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
            )
        else:
            layers = []
            layers.append(FunctionalModule(_piecewise_linear_log))
            layers.append(GlobalStatsNormalization(global_stats_path))
            layers.append(FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)))
            for _ in range(config["specaug_conf"]["n_freq_masks"]):
                layers.append(
                    torchaudio.transforms.FrequencyMasking(
                        config["specaug_conf"]["freq_mask_param"]
                    )
                )
            for _ in range(config["specaug_conf"]["n_time_masks"]):
                layers.append(
                    torchaudio.transforms.TimeMasking(
                        config["specaug_conf"]["time_mask_param"], 
                        p=config["specaug_conf"]["p"]
                    )
                )
            layers.append(FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)))
            self.train_data_pipeline = torch.nn.Sequential(
                *layers,
            )

    def __call__(self, samples: List):
        # if self.config["speed_perturbation"] is None or self.config["speed_perturbation"] is False:
        #     features, feature_lengths = _extract_features(self.train_data_pipeline, samples)
        # else:
        #     features, feature_lengths = _extract_features_train(self.train_data_pipeline, samples)
        features, feature_lengths = _extract_features(
            self.train_data_pipeline, 
            samples,
            speed_perturbation=self.config["speed_perturbation"],
            musan_noise=self.config["musan_noise"],
        )

        targets, target_lengths = _extract_labels(self.sp_model, samples)
        return Batch(features, feature_lengths, targets, target_lengths, samples)


class ValTransform:
    def __init__(self, global_stats_path: str, sp_model):
        self.sp_model = sp_model
        self.valid_data_pipeline = torch.nn.Sequential(
            FunctionalModule(_piecewise_linear_log),
            GlobalStatsNormalization(global_stats_path),
        )

    def __call__(self, samples: List):
        features, feature_lengths = _extract_features(self.valid_data_pipeline, samples)
        targets, target_lengths = _extract_labels(self.sp_model, samples)
        return Batch(features, feature_lengths, targets, target_lengths, samples)


class TestTransform:
    def __init__(self, global_stats_path: str, sp_model):
        self.val_transforms = ValTransform(global_stats_path, sp_model)

    def __call__(self, sample):
        return self.val_transforms([sample]), [sample]


def get_data_module(librispeech_path, global_stats_path, sp_model, config):
    if config["musan_noise"]:
        subsets = config["musan_noise"]["subsets"]
        musan = Musan(musan_path, subsets)
        global _additive_noise_transform
        _additive_noise_transform = AddNoise(musan, snr=tuple(config["musan_noise"]["snr"]), p=config["musan_noise"]["p"])

    train_transform = TrainTransform(global_stats_path=global_stats_path, sp_model=sp_model, config=config)
    val_transform = ValTransform(global_stats_path=global_stats_path, sp_model=sp_model)
    test_transform = TestTransform(global_stats_path=global_stats_path, sp_model=sp_model)
    return LibriSpeechDataModule(
        librispeech_path=librispeech_path,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        batch_size=config["optim_config"]["batch_size"],
        max_tokens=config["optim_config"]["max_tokens"],
        train_num_buckets=config["optim_config"]["train_num_buckets"],
        full_libri=config["training_config"]["full_libri"],
    )
