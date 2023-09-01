from typing import List, Optional, Union, Tuple
import k2
import torch
from torch import Tensor, nn

from graph_compiler_bpe import BpeCtcTrainingGraphCompiler
from graph_compiler_char import CharCtcTrainingGraphCompiler
from graph_compiler_phone import PhonemeCtcTrainingGraphCompiler

import sys
import math
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
        mode="train",
        device=None,
        prior_scaling_factor=0,
        frame_dropout_rate=0,
        torch_ctc_loss=None,
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

        # if mode == "pseudo" or mode == "align":
        #     ce_weight = torch.full((self.graph_compiler.sp.get_piece_size(),), 2.0).to(device)
        #     ce_weight[0] = 1.0
        #     self.ce_loss = nn.CrossEntropyLoss(weight=ce_weight, reduction=self.reduction)
        # else:
        #     self.ce_loss = None
        ce_weight = torch.full((self.graph_compiler.sp.get_piece_size(),), 2.0).to(device)
        ce_weight[0] = 1.0
        self.ce_loss = nn.CrossEntropyLoss(weight=ce_weight, reduction=self.reduction)

        self.bce_loss = nn.BCELoss(reduction=self.reduction)
        self.torch_ctc_loss = torch_ctc_loss

        self.log_priors = None
        self.log_priors_sum = None
        self.priors_T = 0

        self.prior_scaling_factor = prior_scaling_factor
        self.frame_dropout_rate = frame_dropout_rate

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

    def _disable_neighboring(self, mask):
        # mask is a 1D boolean list
        prev = False
        i = 0
        while i < len(mask):
            if mask[i] and prev:
                mask[i] = False
            prev = mask[i]
            i += 1
        return mask

    def frame_dropout(self, log_probs, input_lengths, drop_out_rate, frame_length_threshold=25):
        """
        Randomly throw away frames by the probability of drop_out_rate
        """
        log_probs = log_probs.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)

        new_log_probs = []
        new_lengths = input_lengths.clone()
        for i in range(log_probs.size(0)):
            len_orig = int(input_lengths[i].item())

            if len_orig <= frame_length_threshold:
                new_log_probs.append(log_probs[i][:len_orig])
                continue

            num_throw_away = int(len_orig * drop_out_rate)
            if num_throw_away == 0:
                new_log_probs.append(log_probs[i][:len_orig])
                continue

            mask = torch.rand(len_orig).le(drop_out_rate)
            mask = self._disable_neighboring(mask)
            num_throw_away = int(mask.sum().item())
            len_new = len_orig - num_throw_away
            
            # log_probs[i][:len_new] = log_probs[i][(~mask).nonzero().squeeze()]
            new_log_probs.append(log_probs[i][(~mask).nonzero().squeeze()])
            new_lengths[i] = len_new

        # log_probs = log_probs.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
        new_log_probs = torch.nn.utils.rnn.pad_sequence(new_log_probs)
        return new_log_probs, new_lengths

    def forward(self, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, samples = None, step_type="train") -> Tensor:
        # print(f"(before) input_lengths={input_lengths}, log_probs.shape={log_probs.shape}")
        # input_lengths_b = input_lengths
        if self.frame_dropout_rate > 0:
            frame_dropout_rate_ = 0
            if torch.rand(1) < 0.9:
                frame_dropout_rate_ = torch.rand(1) * self.frame_dropout_rate
            log_probs, input_lengths = self.frame_dropout(log_probs, input_lengths, drop_out_rate=frame_dropout_rate_, frame_length_threshold=25)
        # print(f"(after)  input_lengths={input_lengths}, log_probs.shape={log_probs.shape}, ratio={input_lengths/input_lengths_b}")

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
        
        # # Adding label priors (per batch)
        # # prior_scaling_factor = 0.3
        # if self.log_priors is not None:
        #     log_probs -= self.log_priors * self.prior_scaling_factor

        if True and step_type == "train":
            log_probs_flattened = []
            for lp, le in zip(log_probs, input_lengths):
                log_probs_flattened.append(lp[:int(le.item())])      
            log_probs_flattened = torch.cat(log_probs_flattened, 0)

            # Note, the log_probs here is already log_softmax'ed.
            T = log_probs_flattened.size(0)
            self.priors_T += T
            log_batch_priors_sum = torch.logsumexp(log_probs_flattened, dim=0, keepdim=True)
            log_batch_priors_sum = log_batch_priors_sum.detach()
            if self.log_priors_sum is None:
                self.log_priors_sum = log_batch_priors_sum
            else:
                _temp = torch.stack([self.log_priors_sum, log_batch_priors_sum], dim=-1)
                self.log_priors_sum = torch.logsumexp(_temp, dim=-1)

        # Adding label priors (per batch)
        # prior_scaling_factor = 0.3
        log_probs_original = log_probs
        if True and self.log_priors is not None and self.prior_scaling_factor > 0:
            log_probs = log_probs - self.log_priors * self.prior_scaling_factor

        dense_fsa_vec = k2.DenseFsaVec(
            log_probs,
            supervision_segments,
            allow_truncate=self.subsampling_factor - 1,
        )

        if True:
            loss = k2.ctc_loss(
                decoding_graph=decoding_graph,
                dense_fsa_vec=dense_fsa_vec,
                output_beam=self.ctc_beam_size,
                reduction=self.reduction,
                use_double_scores=self.use_double_scores,
                # delay_penalty=0.05,
            )
        
        if False:
            output_beam = self.ctc_beam_size  # https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/conformer_ctc/ali.py
            lattice = k2.intersect_dense(
                decoding_graph,
                dense_fsa_vec,
                output_beam,
            )

            best_path = one_best_decoding(
                lattice=lattice,
                use_double_scores=True,
            )

            forward_scores = best_path.get_tot_scores(use_double_scores=True, log_semiring=True)
            loss = -forward_scores.sum()

        if False and step_type == "train":
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

            labels_ali = []
            for ali_ in labels_ali_:
                labels_ali[0:0] = ali_
            labels_ali = torch.tensor(labels_ali)
            ids, cnts = labels_ali.unique(return_counts=True)

            batch_priors_sum = torch.zeros((1, log_probs.size(-1)))
            batch_priors_sum[0][ids] += cnts
            log_batch_priors_sum = batch_priors_sum.log()
            if self.log_priors_sum is None:
                self.log_priors_sum = log_batch_priors_sum
            else:
                _temp = torch.stack([self.log_priors_sum, log_batch_priors_sum], dim=-1)
                self.log_priors_sum = torch.logsumexp(_temp, dim=-1)
            
            T = len(labels_ali)
            self.priors_T += T
        
        if False:  # Now, let's get the pseudo labels
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
            labels_ali = [None] * len(labels_ali_)
            for i, ali_ in zip(indices.tolist(), labels_ali_):
                labels_ali[i] = ali_

            log_probs_flattened = []
            labels_flattened = []
            assert len(labels_ali) == len(log_probs)
            for la, p, le in zip(labels_ali, log_probs_original, input_lengths):
                assert le.item() == len(la)
                log_probs_flattened.append(p[:int(le.item())])
                labels_flattened.extend(la)
            log_probs_flattened = torch.cat(log_probs_flattened, 0)
            labels_flattened = torch.tensor(labels_flattened, dtype=torch.long).to(log_probs_flattened.device)
            
            ce_loss = self.ce_loss(
                log_probs_flattened, labels_flattened
            )

            ctc_loss = loss
            loss = 0.7 * ctc_loss + 0.3 * ce_loss
            # print(f"ctc_loss={ctc_loss}, ce_loss={ce_loss}, loss={loss}")
            
        return loss
    
    def loss_with_pseudo_labels(self, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, samples = None) -> Tensor:
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

        ctc_loss = k2.ctc_loss(
            decoding_graph=decoding_graph,
            dense_fsa_vec=dense_fsa_vec,
            output_beam=self.ctc_beam_size,
            reduction=self.reduction,
            use_double_scores=self.use_double_scores,
        )

        # Now, let's get the pseudo labels

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
        labels_ali = [None] * len(labels_ali_)
        for i, ali_ in zip(indices.tolist(), labels_ali_):
            labels_ali[i] = ali_

        log_probs_flattened = []
        labels_flattened = []
        assert len(labels_ali) == len(log_probs)
        for la, p, le in zip(labels_ali, log_probs, input_lengths):
            assert le.item() == len(la)
            log_probs_flattened.append(p[:int(le.item())])
            labels_flattened.extend(la)        
        log_probs_flattened = torch.cat(log_probs_flattened, 0)
        labels_flattened = torch.tensor(labels_flattened, dtype=torch.long).to(log_probs_flattened.device)
        
        ce_loss = self.ce_loss(
            log_probs_flattened, labels_flattened
        )

        loss = 0.5 * ctc_loss + 0.5 * ce_loss

        return loss

    def loss_with_vad(self, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, samples = None, frame_dur=0.04) -> Tensor:
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

        ctc_loss = k2.ctc_loss(
            decoding_graph=decoding_graph,
            dense_fsa_vec=dense_fsa_vec,
            output_beam=self.ctc_beam_size,
            reduction=self.reduction,
            use_double_scores=self.use_double_scores,
        )

        # Now, let's get the labels from vad

        boundaries_0 = []
        boundaries_1 = []
        for sample in samples:
            waveform = sample[0]
            sample_rate = sample[1]
            speech_segments, non_speech_segments = energy_VAD(
                waveform, 
                sample_rate, 
                activation_th=0.4, 
                deactivation_th=0.2, 
                close_th=0.10,
                len_th=0.10,
                time_resolution=frame_dur,
            )
            boundaries_0.append(non_speech_segments)
            boundaries_1.append(speech_segments)

        log_probs_flattened_0 = []
        for p, segs in zip(log_probs, boundaries_0):
            for s in segs:
                log_probs_flattened_0.append(p[s[0]: s[1]][:,0])
        log_probs_flattened_0 = torch.cat(log_probs_flattened_0, 0)
        log_probs_flattened_0.exp_()
        labels_flattened_0 = torch.ones_like(log_probs_flattened_0)

        log_probs_flattened_1 = []
        for p, segs in zip(log_probs, boundaries_1):
            for s in segs:
                log_probs_flattened_1.append(p[s[0]: s[1]][:,0])
        log_probs_flattened_1 = torch.cat(log_probs_flattened_1, 0)
        log_probs_flattened_1.exp_()
        labels_flattened_1 = torch.zeros_like(log_probs_flattened_1)

        log_probs_flattened = torch.cat([log_probs_flattened_0, log_probs_flattened_1], 0)
        labels_flattened = torch.cat([labels_flattened_0, labels_flattened_1], 0)
        
        bce_loss = self.bce_loss(log_probs_flattened, labels_flattened)
        # scaling_factor = input_lengths.sum() / log_probs_flattened_0.size(0)
        scaling_factor = 0.01
        bce_loss *= scaling_factor

        import pdb; pdb.set_trace()
        # log_probs_flattened.round().int().tolist()
        # (log_probs_flattened < 0.8).sum(), log_probs_flattened.size()
        # ctc_loss, bce_loss

        loss = 0.5 * ctc_loss + 0.5 * bce_loss

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
        
        # prior_scaling_factor = 0.3
        if True and self.log_priors is not None and self.prior_scaling_factor > 0:
            log_probs -= self.log_priors * self.prior_scaling_factor

        dense_fsa_vec = k2.DenseFsaVec(
            log_probs,
            supervision_segments,
            allow_truncate=self.subsampling_factor - 1,
        )

        output_beam = self.ctc_beam_size  # https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/conformer_ctc/ali.py
        # output_beam = 20
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

    def align_reverse(self, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, samples = None) -> Tensor:
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
        
        # prior_scaling_factor = 0.3
        if self.log_priors is not None:
            log_probs -= self.log_priors * self.prior_scaling_factor

        #####
        new_log_probs = []
        for i in range(log_probs.size(0)):
            len_orig = int(input_lengths[i].item())
            new_log_probs.append(log_probs[i][:len_orig].flip(dims=[0,]))
        new_log_probs = torch.nn.utils.rnn.pad_sequence(new_log_probs, batch_first=True)
        decoding_graph = k2.reverse(decoding_graph)
        #####

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

        #####
        best_path = k2.reverse(best_path)
        #####

        labels_ali_ = get_alignments(best_path, kind="labels")
        aux_labels_ali_ = get_alignments(best_path, kind="aux_labels")
        
        labels_ali = [None] * len(labels_ali_)
        aux_labels_ali = [None] * len(aux_labels_ali_)
        for i, ali_, aux_ali_ in zip(indices.tolist(), labels_ali_, aux_labels_ali_):
            labels_ali[i] = ali_
            aux_labels_ali[i] = aux_ali_

        return labels_ali, aux_labels_ali, log_probs


from speechbrain.pretrained import VAD
vad = None
def energy_VAD(
    waveform,
    sample_rate,
    activation_th=0.5,
    deactivation_th=0.0,
    close_th=0.250,
    len_th=0.250,
    eps=1e-6,
    time_resolution=0.01,
):
    global vad
    if vad is None:
        vad = VAD(hparams={"time_resolution": time_resolution, "sample_rate": sample_rate, "device": torch.device("cpu")})
    chunk_len = int(time_resolution * sample_rate)

    # Create chunks
    segment_chunks = vad.create_chunks(
        waveform, chunk_size=chunk_len, chunk_stride=chunk_len
    )

    # Energy computation within each chunk
    energy_chunks = segment_chunks.abs().sum(-1) + eps
    energy_chunks = energy_chunks.log()

    # Energy normalization
    energy_chunks = (
        (energy_chunks - energy_chunks.mean())
        / (2 * energy_chunks.std())
    ) + 0.5
    energy_chunks = energy_chunks.unsqueeze(0).unsqueeze(2)

    # Apply threshold based on the energy value
    energy_vad = vad.apply_threshold(
        energy_chunks,
        activation_th=activation_th,
        deactivation_th=deactivation_th,
    )

    # Get the boundaries
    energy_boundaries = vad.get_boundaries(
        energy_vad, output_value="seconds"
    )

    # Get the final boundaries in the original signal
    boundaries = []
    for j in range(energy_boundaries.shape[0]):
        start_en = energy_boundaries[j, 0]
        end_end = energy_boundaries[j, 1]
        boundaries.append([start_en, end_end])

    # Convert boundaries to tensor
    boundaries = torch.FloatTensor(boundaries)

    # Merge short segments
    boundaries = vad.merge_close_segments(boundaries, close_th=close_th)

    # Remove short segments
    boundaries = vad.remove_short_segments(boundaries, len_th=len_th)

    # # Double check speech segments
    # if double_check:
    #     boundaries = self.double_check_speech_segments(
    #         boundaries, audio_file, speech_th=speech_th
    #     )
    boundaries /= vad.time_resolution
    boundaries = boundaries.round().int()

    num_frames = energy_chunks.size(1)
    speech_segments = boundaries.tolist()
    # non_speech_segments = torch.cat([torch.tensor([0]), speech_segments[0], torch.tensor([num_frames])])
    # non_speech_segments.unsqueeze_(0)
    non_speech_segments = [[None, 0]] + boundaries.tolist() + [[num_frames, None]]
    non_speech_segments = [[x[1], y[0]] for x, y in zip(non_speech_segments[:-1], non_speech_segments[1:])]
    # non_speech_segments = torch.tensor(non_speech_segments, dtype=int)

    # Tensor containing the start second (or sample) of speech segments
    # in even positions and their corresponding end in odd positions
    # (e.g, [1.0, 1.5, 5,.0 6.0] means that we have two speech segment;
    #     one from 1.0 to 1.5 seconds and another from 5.0 to 6.0 seconds).
    return speech_segments, non_speech_segments


