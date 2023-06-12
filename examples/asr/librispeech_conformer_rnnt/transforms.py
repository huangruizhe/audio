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


_decibel = 2 * 20 * math.log10(torch.iinfo(torch.int16).max)
_gain = pow(10, 0.05 * _decibel)

_spectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, n_mels=80, hop_length=160)
_speed_perturb_transform = torchaudio.transforms.SpeedPerturbation(orig_freq=16000, factors=[0.9, 1.0, 1.1])


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


def _extract_features(data_pipeline, samples: List):
    mel_features = [_spectrogram_transform(sample[0].squeeze()).transpose(1, 0) for sample in samples]
    features = torch.nn.utils.rnn.pad_sequence(mel_features, batch_first=True)
    features = data_pipeline(features)
    lengths = torch.tensor([elem.shape[0] for elem in mel_features], dtype=torch.int32)
    return features, lengths


def _extract_features_train(data_pipeline, samples: List):
    samples = [_speed_perturb_transform(sample[0].squeeze()) for sample in samples]
    mel_features = [_spectrogram_transform(sample[0].squeeze()).transpose(1, 0) for sample in samples]
    features = torch.nn.utils.rnn.pad_sequence(mel_features, batch_first=True)
    features = data_pipeline(features)
    lengths = torch.tensor([elem.shape[0] for elem in mel_features], dtype=torch.int32)
    return features, lengths


class TrainTransform:
    def __init__(self, global_stats_path: str, sp_model_path: str, config: dict):
        self.sp_model = spm.SentencePieceProcessor(model_file=sp_model_path)

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
        if self.config["speed_perturbation"] is None or self.config["speed_perturbation"] is False:
            features, feature_lengths = _extract_features(self.train_data_pipeline, samples)
        else:
            features, feature_lengths = _extract_features_train(self.train_data_pipeline, samples)
        targets, target_lengths = _extract_labels(self.sp_model, samples)
        return Batch(features, feature_lengths, targets, target_lengths)


class ValTransform:
    def __init__(self, global_stats_path: str, sp_model_path: str):
        self.sp_model = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.valid_data_pipeline = torch.nn.Sequential(
            FunctionalModule(_piecewise_linear_log),
            GlobalStatsNormalization(global_stats_path),
        )

    def __call__(self, samples: List):
        features, feature_lengths = _extract_features(self.valid_data_pipeline, samples)
        targets, target_lengths = _extract_labels(self.sp_model, samples)
        return Batch(features, feature_lengths, targets, target_lengths)


class TestTransform:
    def __init__(self, global_stats_path: str, sp_model_path: str):
        self.val_transforms = ValTransform(global_stats_path, sp_model_path)

    def __call__(self, sample):
        return self.val_transforms([sample]), [sample]


def get_data_module(librispeech_path, global_stats_path, sp_model_path, config):
    train_transform = TrainTransform(global_stats_path=global_stats_path, sp_model_path=sp_model_path, config=config)
    val_transform = ValTransform(global_stats_path=global_stats_path, sp_model_path=sp_model_path)
    test_transform = TestTransform(global_stats_path=global_stats_path, sp_model_path=sp_model_path)
    return LibriSpeechDataModule(
        librispeech_path=librispeech_path,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        batch_size=config["optim_config"]["batch_size"],
        max_tokens=config["optim_config"]["max_tokens"],
        train_num_buckets=config["optim_config"]["train_num_buckets"],
    )
