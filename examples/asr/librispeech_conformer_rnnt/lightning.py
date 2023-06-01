import logging
import math
from collections import namedtuple
from typing import List, Tuple

import sentencepiece as spm
import torch
import torchaudio
from pytorch_lightning import LightningModule
from torchaudio.models import Hypothesis, RNNTBeamSearch
from torchaudio.prototype.models import conformer_rnnt_base, conformer_rnnt_model


logger = logging.getLogger("lightning.pytorch.core")

_expected_spm_vocab_size = 1023

Batch = namedtuple("Batch", ["features", "feature_lengths", "targets", "target_lengths"])


class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    r"""Learning rate scheduler that performs linear warmup and exponential annealing.

    Args:
        optimizer (torch.optim.Optimizer): optimizer to use.
        warmup_steps (int): number of scheduler steps for which to warm up learning rate.
        force_anneal_step (int): scheduler step at which annealing of learning rate begins.
        anneal_factor (float): factor to scale base learning rate by at each annealing step.
        last_epoch (int, optional): The index of last epoch. (Default: -1)
        verbose (bool, optional): If ``True``, prints a message to stdout for
            each update. (Default: ``False``)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        force_anneal_step: int,
        anneal_factor: float,
        last_epoch=-1,
        verbose=False,
    ):
        self.warmup_steps = warmup_steps
        self.force_anneal_step = force_anneal_step
        self.anneal_factor = anneal_factor
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self._step_count < self.force_anneal_step:
            return [(min(1.0, self._step_count / self.warmup_steps)) * base_lr for base_lr in self.base_lrs]
        else:
            scaling_factor = self.anneal_factor ** (self._step_count - self.force_anneal_step)
            return [scaling_factor * base_lr for base_lr in self.base_lrs]

            # scaling_factor = self.anneal_factor ** ((self._step_count - self.force_anneal_step) * 0.3)
            # return [scaling_factor * base_lr for base_lr in self.base_lrs]

            # scaling_factor = 1.0 if self._step_count < 250 else 0.6
            # return [2e-4 * scaling_factor for base_lr in self.base_lrs]


class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    r"""
    https://nn.labml.ai/optimizers/noam.html
    https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/optimizer.py
    https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/transducer_stateless/transformer.py
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        model_size: int,
        last_epoch=-1,
        verbose=False,
    ):
        self.step_per_epoch = 3000
        self.warmup_steps = warmup_steps * self.step_per_epoch
        self.model_size = model_size
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        scaling_factor = self.model_size ** (-0.5) \
            * min((self._step_count * self.step_per_epoch) ** (-0.5), (self._step_count * self.step_per_epoch) * self.warmup_steps ** (-1.5))
        return [scaling_factor * base_lr for base_lr in self.base_lrs]


def post_process_hypos(
    hypos: List[Hypothesis], sp_model: spm.SentencePieceProcessor
) -> List[Tuple[str, float, List[int], List[int]]]:
    tokens_idx = 0
    score_idx = 3
    post_process_remove_list = [
        sp_model.unk_id(),
        sp_model.eos_id(),
        sp_model.pad_id(),
    ]
    filtered_hypo_tokens = [
        [token_index for token_index in h[tokens_idx][1:] if token_index not in post_process_remove_list] for h in hypos
    ]
    hypos_str = [sp_model.decode(s) for s in filtered_hypo_tokens]
    hypos_ids = [h[tokens_idx][1:] for h in hypos]
    hypos_score = [[math.exp(h[score_idx])] for h in hypos]

    nbest_batch = list(zip(hypos_str, hypos_score, hypos_ids))

    return nbest_batch


def get_conformer_rnnt(config):
    return conformer_rnnt_model(
        input_dim=config["rnnt_config"]["input_dim"],
        encoding_dim=config["rnnt_config"]["encoding_dim"],
        time_reduction_stride=config["rnnt_config"]["time_reduction_stride"],
        conformer_input_dim=config["rnnt_config"]["conformer_input_dim"],
        conformer_ffn_dim=config["rnnt_config"]["conformer_ffn_dim"],
        conformer_num_layers=config["rnnt_config"]["conformer_num_layers"],
        conformer_num_heads=config["rnnt_config"]["conformer_num_heads"],
        conformer_depthwise_conv_kernel_size=config["rnnt_config"]["conformer_depthwise_conv_kernel_size"],
        conformer_dropout=config["rnnt_config"]["conformer_dropout"],
        num_symbols=config["rnnt_config"]["num_symbols"],
        symbol_embedding_dim=config["rnnt_config"]["symbol_embedding_dim"],
        num_lstm_layers=config["rnnt_config"]["num_lstm_layers"],
        lstm_hidden_dim=config["rnnt_config"]["lstm_hidden_dim"],
        lstm_layer_norm=config["rnnt_config"]["lstm_layer_norm"],
        lstm_layer_norm_epsilon=config["rnnt_config"]["lstm_layer_norm_epsilon"],
        lstm_dropout=config["rnnt_config"]["lstm_dropout"],
        joiner_activation=config["rnnt_config"]["joiner_activation"],
    )


class ConformerRNNTModule(LightningModule):
    def __init__(self, sp_model, config):
        super().__init__()

        self.sp_model = sp_model
        spm_vocab_size = self.sp_model.get_piece_size()
        assert spm_vocab_size == _expected_spm_vocab_size, (
            "The model returned by conformer_rnnt_base expects a SentencePiece model of "
            f"vocabulary size {_expected_spm_vocab_size}, but the given SentencePiece model has a vocabulary size "
            f"of {spm_vocab_size}. Please provide a correctly configured SentencePiece model."
        )
        assert spm_vocab_size == config["spm_vocab_size"]
        self.blank_idx = spm_vocab_size

        self.config = config

        # ``conformer_rnnt_base`` hardcodes a specific Conformer RNN-T configuration.
        # For greater customizability, please refer to ``conformer_rnnt_model``.
        self.model = get_conformer_rnnt(config)
        self.loss = torchaudio.transforms.RNNTLoss(reduction=config["optim_config"]["reduction"])

        # Default:
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config["optim_config"]["lr"], 
            betas=(0.9, 0.98), 
            eps=1e-9, 
            weight_decay=config["optim_config"]["weight_decay"]
        )
        self.warmup_lr_scheduler = WarmupLR(
            self.optimizer, 
            config["optim_config"]["warmup_steps"], 
            config["optim_config"]["force_anneal_step"], 
            config["optim_config"]["anneal_factor"]
        )

        # # Noam:
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5.0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0)
        # self.warmup_lr_scheduler = NoamLR(self.optimizer, warmup_steps=30, model_size=512)

        self._total_loss = 0
        self._total_frames = 0

        self._total_val_loss = 0
        self._total_val_frames = 0

    def _step(self, batch, _, step_type):
        if batch is None:
            return None

        prepended_targets = batch.targets.new_empty([batch.targets.size(0), batch.targets.size(1) + 1])
        prepended_targets[:, 1:] = batch.targets
        prepended_targets[:, 0] = self.blank_idx
        prepended_target_lengths = batch.target_lengths + 1
        output, src_lengths, _, _ = self.model(
            batch.features,
            batch.feature_lengths,
            prepended_targets,
            prepended_target_lengths,
        )
        loss = self.loss(output, batch.targets, src_lengths, batch.target_lengths)
        self.log(f"Losses/{step_type}_loss", loss, on_step=True, on_epoch=True, sync_dist=True)

        subsampling_factor = self.config["rnnt_config"]["time_reduction_stride"]
        num_frames = (batch.feature_lengths // subsampling_factor).sum().item()
        # https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/conformer_ctc2/train.py#L699
        reset_interval = 200
        self._total_loss = (self._total_loss * (1 - 1 / reset_interval)) + loss.item()
        self._total_frames = (self._total_frames * (1 - 1 / reset_interval)) + num_frames
        self.log(f"Losses_normalized/{step_type}_loss", self._total_loss / self._total_frames, on_step=True, on_epoch=True, sync_dist=True)

        if step_type == "val":
            self._total_val_loss += loss.item()
            self._total_val_frames += num_frames

        return loss

    def configure_optimizers(self):
        return (
            [self.optimizer],
            [{"scheduler": self.warmup_lr_scheduler, "interval": "epoch"}],
        )

    def forward(self, batch: Batch):
        decoder = RNNTBeamSearch(self.model, self.blank_idx)
        hypotheses = decoder(batch.features.to(self.device), batch.feature_lengths.to(self.device), 20)
        return post_process_hypos(hypotheses, self.sp_model)[0][0]

    def training_step(self, batch: Batch, batch_idx):
        """Custom training step.

        By default, DDP does the following on each train step:
        - For each GPU, compute loss and gradient on shard of training data.
        - Sync and average gradients across all GPUs. The final gradient
          is (sum of gradients across all GPUs) / N, where N is the world
          size (total number of GPUs).
        - Update parameters on each GPU.

        Here, we do the following:
        - For k-th GPU, compute loss and scale it by (N / B_total), where B_total is
          the sum of batch sizes across all GPUs. Compute gradient from scaled loss.
        - Sync and average gradients across all GPUs. The final gradient
          is (sum of gradients across all GPUs) / B_total.
        - Update parameters on each GPU.

        Doing so allows us to account for the variability in batch sizes that
        variable-length sequential data yield.
        """
        loss = self._step(batch, batch_idx, "train")
        batch_size = batch.features.size(0)
        batch_sizes = self.all_gather(batch_size)
        self.log("Gathered batch size", batch_sizes.sum(), on_step=True, on_epoch=True)
        loss *= batch_sizes.size(0) / batch_sizes.sum()  # world size / batch size
        return loss

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def on_validation_epoch_end(self):
        # This still has some issue: some utterance may be double-counted
        # https://lightning.ai/docs/pytorch/stable/common/evaluation_intermediate.html#validation

        _total_val_loss = self.all_gather(self._total_val_loss)
        _total_val_frames = self.all_gather(self._total_val_frames)

        self.log(f"Losses_val/val_loss", _total_val_loss.sum() / _total_val_frames.sum(), on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log(f"Losses_val/val_frames", _total_val_frames.sum(), on_epoch=True, sync_dist=True, rank_zero_only=True)
        if self.global_rank == 0:
            logger.info(f"\nvalidation loss: avg={_total_val_loss.sum() / _total_val_frames.sum()} over {_total_val_frames.sum()} frames\n")
        self._total_val_loss = 0
        self._total_val_frames = 0

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")
