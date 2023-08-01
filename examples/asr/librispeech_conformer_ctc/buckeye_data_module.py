import os
import random

import torch
import torchaudio
from pytorch_lightning import LightningDataModule
from buckeye_dataset import BUCKEYE


def _batch_by_token_count(idx_target_lengths, max_tokens, batch_size=None):
    batches = []
    current_batch = []
    current_token_count = 0
    for idx, target_length in idx_target_lengths:
        if current_token_count + target_length > max_tokens or (batch_size and len(current_batch) == batch_size):
            batches.append(current_batch)
            current_batch = [idx]
            current_token_count = target_length
        else:
            current_batch.append(idx)
            current_token_count += target_length

    if current_batch:
        batches.append(current_batch)

    return batches


def get_sample_lengths(buckeye_dataset):
    sample_lengths = []

    for i in range(len(buckeye_dataset)):
        waveform, sample_rate, text, speaker_id, utter_id, wav_file = \
            buckeye_dataset[i]
        sample_lengths.append(len(text))

    return sample_lengths


class CustomBucketDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        lengths,
        max_tokens,
        num_buckets,
        shuffle=False,
        batch_size=None,
    ):
        super().__init__()

        assert len(dataset) == len(lengths)

        self.dataset = dataset

        max_length = max(lengths)
        min_length = min(lengths)

        assert max_tokens >= max_length

        buckets = torch.linspace(min_length, max_length, num_buckets)
        lengths = torch.tensor(lengths)
        bucket_assignments = torch.bucketize(lengths, buckets)

        idx_length_buckets = [(idx, length, bucket_assignments[idx]) for idx, length in enumerate(lengths)]
        if shuffle:
            idx_length_buckets = random.sample(idx_length_buckets, len(idx_length_buckets))
        else:
            idx_length_buckets = sorted(idx_length_buckets, key=lambda x: x[1], reverse=True)

        sorted_idx_length_buckets = sorted(idx_length_buckets, key=lambda x: x[2])
        self.batches = _batch_by_token_count(
            [(idx, length) for idx, length, _ in sorted_idx_length_buckets],
            max_tokens,
            batch_size=batch_size,
        )

    def __getitem__(self, idx):
        return [self.dataset[subidx] for subidx in self.batches[idx]]

    def __len__(self):
        return len(self.batches)


class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform_fn):
        self.dataset = dataset
        self.transform_fn = transform_fn

    def __getitem__(self, idx):
        return self.transform_fn(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)


class BuckeyeDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        buckeye_path,
        transform,
        max_tokens=700,
        batch_size=2,
        num_buckets=50,
        train_shuffle=True,
        num_workers=10,
    ):
        super().__init__()
        self.buckeye_path = buckeye_path
        self.dataset_lengths = None
        self.transform = transform
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.num_buckets = num_buckets
        self.train_shuffle = train_shuffle
        self.num_workers = num_workers

    def train_dataloader(self):
        datasets = [BUCKEYE(self.buckeye_path)]

        if not self.dataset_lengths:
            self.dataset_lengths = [get_sample_lengths(dataset) for dataset in datasets]

        dataset = torch.utils.data.ConcatDataset(
            [
                CustomBucketDataset(
                    dataset,
                    lengths,
                    self.max_tokens,
                    self.num_buckets,
                    batch_size=self.batch_size,
                    shuffle=True,
                )
                for dataset, lengths in zip(datasets, self.dataset_lengths)
            ]
        )
        dataset = TransformDataset(dataset, self.transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=None,
            shuffle=self.train_shuffle,
        )
        return dataloader

    def val_dataloader(self):
        return self.train_dataloader()
    
    # def val_dataloader(self):
    #     datasets = [
    #         self.librispeech_cls(self.librispeech_path, url="dev-clean"),
    #         self.librispeech_cls(self.librispeech_path, url="dev-other"),
    #     ]

    #     if not self.val_dataset_lengths:
    #         self.val_dataset_lengths = [get_sample_lengths(dataset) for dataset in datasets]

    #     dataset = torch.utils.data.ConcatDataset(
    #         [
    #             CustomBucketDataset(
    #                 dataset,
    #                 lengths,
    #                 self.max_tokens,
    #                 1,
    #                 batch_size=self.batch_size,
    #             )
    #             for dataset, lengths in zip(datasets, self.val_dataset_lengths)
    #         ]
    #     )
    #     dataset = TransformDataset(dataset, self.val_transform)
    #     dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=self.num_workers)
    #     return dataloader

    # def test_dataloader(self, test_part="test-other"):
    #     dataset = self.librispeech_cls(self.librispeech_path, url=test_part)
    #     dataset = TransformDataset(dataset, self.test_transform)
    #     dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)
    #     return dataloader
