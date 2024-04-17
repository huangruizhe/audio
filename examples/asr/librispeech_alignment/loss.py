import torch
from torch import Tensor, nn
import k2
from typing import List, Optional, Union, Tuple


class CTCLossWithLabelPriors(nn.Module):
    def __init__(self, ctc_implementation='k2', blank=0, reduction='sum'):
        super().__init__()
        self.ctc_implementation = ctc_implementation
        if ctc_implementation == 'k2':
            pass
        elif ctc_implementation == 'torch':
            self.ctc_loss = torch.nn.CTCLoss(blank=blank, reduction=reduction)
        elif ctc_implementation == 'ctc_primer':
            pass
        else:
            raise ValueError(f"ctc_implementation={ctc_implementation} is not supported")
    
    def forward(self, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor) -> Tensor:
        '''
        log_probs: (T, N, C)
        '''
        if self.ctc_implementation == 'k2':
            return self.forward_k2(log_probs, targets, input_lengths, target_lengths)
        elif self.ctc_implementation == 'torch':
            return self.forward_torch(log_probs, targets, input_lengths, target_lengths)
        elif self.ctc_implementation == 'ctc_primer':
            return self.forward_ctc_primer(log_probs, targets, input_lengths, target_lengths)
        else:
            raise ValueError(f"ctc_implementation={self.ctc_implementation} is not supported")

    def forward_torch(self, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor) -> Tensor:
        # Note:
        # Pytorch's ctc implementation is not suitable to use with label priors.
        # It leads to incorrect gradients.
        # Check out this issue for more details: https://github.com/pytorch/pytorch/issues/122243

        loss = self.ctc_loss(log_probs, targets, input_lengths.int(), target_lengths.int())
        return loss

    def forward_ctc_primer(self, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor) -> Tensor:
        if self.frame_dropout_rate > 0:
            frame_dropout_rate_ = 0
            if torch.rand(1) < 0.9:
                frame_dropout_rate_ = torch.rand(1) * self.frame_dropout_rate
            log_probs, input_lengths = self.frame_dropout(log_probs, input_lengths, drop_out_rate=frame_dropout_rate_, frame_length_threshold=25)

        log_probs = log_probs.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)
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

        if True and self.log_priors is not None and self.prior_scaling_factor > 0:
            log_probs = log_probs - self.log_priors * self.prior_scaling_factor

        scratch_space['log_probs'] = log_probs
        scratch_space['input_lengths'] = input_lengths

        log_probs = log_probs.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)        
        loss = ctc_primer.ctc_loss(log_probs, targets.to(torch.int64), input_lengths.to(torch.int64), target_lengths.to(torch.int64), blank = 0, reduction = 'none').sum()
        return loss

    def forward_k2(self, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor) -> Tensor:
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
        
        _samples = [samples[i] for i in indices.tolist()]
        decoding_graph = self.graph_compiler.compile(token_ids, _samples)

        log_probs = log_probs.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        # log_probs = torch.roll(log_probs, 1, -1)  # Now blank symbol has the index of 0
        
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
        
        scratch_space['log_probs'] = log_probs
        scratch_space['input_lengths'] = input_lengths

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
        
        if False:  # Also checkout: /exp/rhuang/meta/k2/k2/python/k2/ctc_loss.py
            output_beam = self.ctc_beam_size  # https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/conformer_ctc/ali.py
            if output_beam == 0:
                output_beam = 10
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