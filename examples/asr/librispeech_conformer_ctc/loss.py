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
        mode="train",
        device=None,
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

        # Adding label priors (per batch)
        T = log_probs.size(1)
        batch_priors = torch.logsumexp(log_probs, dim=[0, 1], keepdim=True) / T
        batch_priors = batch_priors.detach()
        log_probs -= batch_priors

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
            # delay_penalty=0.1,
        )
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


