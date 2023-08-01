from typing import List, Optional, Union, Tuple
import k2
import torch
from torch import Tensor, nn

from graph_compiler_bpe import BpeCtcTrainingGraphCompiler
from graph_compiler_char import CharCtcTrainingGraphCompiler
from graph_compiler_phone import PhonemeCtcTrainingGraphCompiler

import sys
sys.path.insert(0,'/fsx/users/huangruizhe/icefall_align2')
sys.path.insert(0,'/fsx/users/huangruizhe/icefall_align2/egs/librispeech/ASR/zipformer_mmi')
from icefall.decode import one_best_decoding
from icefall.utils import get_alignments


class MaximumLikelihoodLoss(nn.Module):
    """
    Computes maximum likelihood loss.

    TODO: more detailed description
    """

    def __init__(
        self,
        graph_compiler: BpeCtcTrainingGraphCompiler,
        padding_value=1.0,
        subsampling_factor: int = 4,
        ctc_beam_size: float = 10.0,
        reduction = "sum",
        use_double_scores = True,
    ):
        super().__init__()
        self.graph_compiler = graph_compiler
        # The default padding value is 1.0 as in
        # /fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc/transforms.py
        # Does the padding values matter?
        self.padding_value = padding_value

        self.subsampling_factor = subsampling_factor
        self.ctc_beam_size = ctc_beam_size
        self.reduction = reduction
        self.use_double_scores = use_double_scores

    def encode_supervisions(
        self, targets, target_lengths, input_lengths
    ) -> Tuple[torch.Tensor, Union[List[str], List[List[int]]]]:
        """
        Encodes Lhotse's ``batch["supervisions"]`` dict into
        a pair of torch Tensor, and a list of transcription strings or token indexes

        The supervision tensor has shape ``(batch_size, 3)``.
        Its second dimension contains information about sequence index [0],
        start frames [1] and num frames [2].

        The batch items might become re-ordered during this operation -- the
        returned tensor and list of strings are guaranteed to be consistent with
        each other.
        """
        batch_size = targets.size(0)
        supervision_segments = torch.stack(
            (
                torch.arange(batch_size),
                torch.zeros(batch_size),
                input_lengths.cpu(),
            ),
            1,
        ).to(torch.int32)

        indices = torch.argsort(supervision_segments[:, 2], descending=True)
        supervision_segments = supervision_segments[indices]
        # import pdb; pdb.set_trace()

        res = targets[indices].tolist()
        res_lengths = target_lengths[indices].tolist()
        res = [[i + 1 for i in l[:l_len]] for l, l_len in zip(res, res_lengths)]  # hard-coded for torchaudio

        return supervision_segments, res, indices

    def forward(self, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, samples = None) -> Tensor:
        # Be careful: the targets here are already padded! We need to remove paddings from it
        supervision_segments, texts, indices = self.encode_supervisions(targets, target_lengths, input_lengths)
        token_ids = texts

        # import pdb; pdb.set_trace()
        
        if type(self.graph_compiler) is BpeCtcTrainingGraphCompiler:
            decoding_graph = self.graph_compiler.compile(token_ids)
        elif type(self.graph_compiler) is CharCtcTrainingGraphCompiler:
            _samples = [samples[i] for i in indices.tolist()]
            decoding_graph = self.graph_compiler.compile(token_ids, _samples)
        elif type(self.graph_compiler) is PhonemeCtcTrainingGraphCompiler:
            _samples = [samples[i] for i in indices.tolist()]
            decoding_graph = self.graph_compiler.compile(token_ids, _samples)
        else:
            raise NotImplementedError

        log_probs = log_probs.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        log_probs = torch.roll(log_probs, 1, -1)  # Now blank symbol has the index of 0

        dense_fsa_vec = k2.DenseFsaVec(
            log_probs,
            supervision_segments,
            allow_truncate=self.subsampling_factor - 1,
        )

        loss = k2.ctc_loss(
            decoding_graph=decoding_graph,
            dense_fsa_vec=dense_fsa_vec,
            output_beam=self.ctc_beam_size,
            reduction=self.reduction,
            use_double_scores=self.use_double_scores,
        )
        return loss
    
    def align(self, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, samples = None) -> Tensor:
        # Be careful: the targets here are already padded! We need to remove paddings from it
        supervision_segments, texts, indices = self.encode_supervisions(targets, target_lengths, input_lengths)
        token_ids = texts

        # import pdb; pdb.set_trace()
        
        if type(self.graph_compiler) is BpeCtcTrainingGraphCompiler:
            decoding_graph = self.graph_compiler.compile(token_ids)
        elif type(self.graph_compiler) is CharCtcTrainingGraphCompiler:
            _samples = [samples[i] for i in indices.tolist()]
            decoding_graph = self.graph_compiler.compile(token_ids, _samples)
        elif type(self.graph_compiler) is PhonemeCtcTrainingGraphCompiler:
            _samples = [samples[i] for i in indices.tolist()]
            decoding_graph = self.graph_compiler.compile(token_ids, _samples)
        else:
            raise NotImplementedError

        log_probs = log_probs.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        log_probs = torch.roll(log_probs, 1, -1)  # Now blank symbol has the index of 0

        dense_fsa_vec = k2.DenseFsaVec(
            log_probs,
            supervision_segments,
            allow_truncate=self.subsampling_factor - 1,
        )

        output_beam = 10  # https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/conformer_ctc/ali.py
        lattice = k2.intersect_dense(
            decoding_graph,
            dense_fsa_vec,
            output_beam,
        )

        best_path = one_best_decoding(
            lattice=lattice,
            use_double_scores=True,
        )

        labels_ali_ = get_alignments(best_path, kind="labels")
        aux_labels_ali_ = get_alignments(best_path, kind="aux_labels")
        
        labels_ali = [None] * len(labels_ali_)
        aux_labels_ali = [None] * len(aux_labels_ali_)
        for i, ali_, aux_ali_ in zip(indices.tolist(), labels_ali_, aux_labels_ali_):
            labels_ali[i] = ali_
            aux_labels_ali[i] = aux_ali_

        return labels_ali, aux_labels_ali, log_probs