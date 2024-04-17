import logging
import math
from collections import namedtuple
from typing import List, Tuple

import torch
import torchaudio
from pytorch_lightning import LightningModule
from models import tdnn_blstm_ctc_model_base
from loss import CTCLossWithLabelPriors

logger = logging.getLogger()

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


class AcousticModelModule(LightningModule):
    def __init__(self, tokenizer):
        super().__init__()

        self.tokenizer = tokenizer
        output_vocab_size = len(tokenizer.token2id)
        self.blank_idx = tokenizer.blk_id

        # The acoustic model hardcodes a specific TDNN-FFN configuration.
        self.model = tdnn_blstm_ctc_model_base(output_vocab_size)
        self.loss = CTCLossWithLabelPriors(ctc_implementation="torch", blank=self.blank_idx)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=8e-4, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
        self.warmup_lr_scheduler = WarmupLR(self.optimizer, 40, 120, 0.96)

    def _step(self, batch, _, step_type):
        if batch is None:
            return None

        output, src_lengths = self.model(
            batch.features,
            batch.feature_lengths,
        )
        loss = self.loss(output, batch.targets, src_lengths, batch.target_lengths, step_type=step_type)
        self.log(f"Losses/{step_type}_loss", loss, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return (
            [self.optimizer],
            [{"scheduler": self.warmup_lr_scheduler, "interval": "epoch"}],
        )
    
    def decode(self, batch: Batch):
        if batch is None:
            return None

        with torch.inference_mode():
            output, src_lengths = self.model(
                batch.features,
                batch.feature_lengths,
            )
        emission = output.permute(1, 0, 2).cpu()  # (T, N, num_label) => (N, T, num_label)

        # A simplest greedy decoding
        indices = torch.argmax(emission, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [[i for i in utt.tolist() if i != self.blank_idx] for utt in indices]
        joined = ["".join(self.tokenizer.decode_flatten(utt)) for utt in indices]
        return joined

    def forward(self, batch: Batch):
        # TODO: return alignment results
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

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")
