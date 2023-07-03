# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pathlib import Path
from typing import List, Union
from collections import defaultdict

import k2
from tokenizer_char import CharTokenizer as spm
import torch


class CharCtcTrainingGraphCompiler(object):
    def __init__(
        self,
        bpe_model,
        device: Union[str, torch.device] = "cpu",
        topo_type = "ctc",
        padding_value = 1.0,
    ) -> None:
        """
        Args:
          lang_dir:
            This directory is expected to contain the following files:

                - bpe.model
                - words.txt
          device:
            It indicates CPU or CUDA.
          sos_token:
            The word piece that represents sos.
          eos_token:
            The word piece that represents eos.
        """
        # lang_dir = Path(lang_dir)
        # model_file = lang_dir / "bpe.model"
        sp = bpe_model
        self.sp = sp
        # self.word_table = k2.SymbolTable.from_file(lang_dir / "words.txt")
        self.device = device

        self.start_tokens = {token_id for token_id in range(sp.vocab_size()) if sp.id_to_piece(token_id).startswith("▁")}
        self.remove_intra_word_blk_flag = True
        # print(f"self.remove_intra_word_blk_flag={self.remove_intra_word_blk_flag}")

        if topo_type == "hmm":
            self.max_token_id = sp.vocab_size() + 1  # hard-coded for torch audio
            self.topo = CharCtcTrainingGraphCompiler.hmm_topo(self.max_token_id, device, sil_id=0)
        else:
            self.max_token_id = None
            self.topo = None

        self.topo_type = topo_type
        self.padding_value = padding_value

    # This works!
    def compile_ctc(
        self,
        piece_ids: List[List[int]],
        modified: bool = False,
    ) -> k2.Fsa:
        """Build a ctc graph from a list-of-list piece IDs.

        Args:
          piece_ids:
           IDs. It is a list-of-list integer 
          modified:
           See :func:`k2.ctc_graph` for its meaning.
        Return:
          Return an FsaVec, which is the result of composing a
          CTC topology with linear FSAs constructed from the given
          piece IDs.
        """
        graph = k2.ctc_graph(piece_ids, modified=modified, device=self.device)

        # graph = self.remove_intra_word_blk(graph, self.start_tokens, flag=self.remove_intra_word_blk_flag)
        return graph

    # @staticmethod
    # def hmm_topo(
    #     max_token: int,
    #     device = None,
    #     sil_id: int = 0,
    # ) -> k2.Fsa:
    #     '''
    #     HMM topo
    #     '''
    #     print(f"Creating Char HMM topo for {max_token} tokens")
    #     num_tokens = max_token
    #     # assert (
    #     #     sil_id <= max_token
    #     # ), f"sil_id={sil_id} should be less or equal to max_token={max_token}"

    #     # ref: https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/local/prepare_lang.py#L248

    #     start_state = 0
    #     loop_state = 1
    #     blk_state = 2
    #     next_available_state = 3
    #     arcs = []

    #     blk = sil_id
    #     arcs.append([start_state, start_state, blk, blk, 0])

    #     for i in range(1, max_token + 1):
    #         arcs.append([start_state, loop_state, i, i, 0])

    #     arcs.append([loop_state, blk_state, blk, blk, 0])
    #     arcs.append([blk_state, blk_state, blk, blk, 0])

    #     for i in range(1, max_token + 1):
    #         cur_state = next_available_state  # state_id
    #         next_available_state += 1

    #         arcs.append([loop_state, loop_state, i, i, 0])
    #         arcs.append([loop_state, cur_state, i, i, 0])
    #         arcs.append([cur_state, cur_state, i, blk, 0])
    #         arcs.append([cur_state, loop_state, i, blk, 0])
            
    #         arcs.append([start_state, cur_state, i, i, 0])

    #         if i in start_tokens:
    #             arcs.append([blk_state, loop_state, i, i, 0])
    #             arcs.append([blk_state, cur_state, i, i, 0])

    #     final_state = next_available_state
    #     next_available_state += 1
    #     arcs.append([start_state, final_state, -1, -1, 0])
    #     arcs.append([loop_state, final_state, -1, -1, 0])
    #     arcs.append([blk_state, final_state, -1, -1, 0])    
    #     arcs.append([final_state])

    #     arcs = sorted(arcs, key=lambda arc: arc[0])
    #     arcs = [[str(i) for i in arc] for arc in arcs]
    #     arcs = [" ".join(arc) for arc in arcs]
    #     arcs = "\n".join(arcs)

    #     fst = k2.Fsa.from_str(arcs, acceptor=False)
    #     # fst = k2.remove_epsilon(fst)  # Credit: Matthew W
    #     # fst = k2.expand_ragged_attributes(fst)
    #     fst = k2.arc_sort(fst)
        
    #     if device is not None:
    #         fst = fst.to(device)

    #     return fst

    @staticmethod
    def hmm_topo(
        max_token: int,
        device = None,
        sil_id: int = 0,
    ) -> k2.Fsa:
        '''
        HMM topo
        '''
        print(f"Creating Char HMM topo for {max_token} tokens")

        return k2.ctc_topo(max_token, modified=True, device=device)

    def _get_transcript_fsa(self, target, blk_id=0):
        start_state = 0
        next_available_state = 1
        arcs = []

        prev_state = start_state
        blk_state = next_available_state
        next_available_state += 1
        next_state = next_available_state
        next_available_state += 1
        arcs.append([prev_state, blk_state, blk_id, 0])
        arcs.append([blk_state, blk_state, blk_id, 0])
        for word in target:
            for i, token in enumerate(word):
                if i == 0:  # The first token of a word
                    arcs.append([blk_state, next_state, token, 0])
                arcs.append([prev_state, next_state, token, 0])
                prev_state = next_state
                next_state = next_available_state
                next_available_state += 1
            blk_state = next_available_state
            next_available_state += 1
            arcs.append([prev_state, blk_state, blk_id, 0])
            arcs.append([blk_state, blk_state, blk_id, 0])
        end_state = next_available_state
        arcs.append([blk_state, end_state, -1, 0])
        arcs.append([prev_state, end_state, -1, 0])
        arcs.append([end_state])        

        arcs = sorted(arcs, key=lambda arc: arc[0])
        arcs = [[str(i) for i in arc] for arc in arcs]
        arcs = [" ".join(arc) for arc in arcs]
        arcs = "\n".join(arcs)

        fsa = k2.Fsa.from_str(arcs)
        fsa = k2.arc_sort(fsa)
        if self.device is not None:
            fsa = fsa.to(self.device)

        return fsa
    
    def get_transcript_fsa(self, samples):
        targets = [self.sp.encode_keep_boundary(sample[2]) for sample in samples]
        targets = [[[i + 1 for i in w if i != self.padding_value] for w in t] for t in targets]  # hard-coded for torchaudio
        transcript_fsa = []
        for target in targets:
            fsa = self._get_transcript_fsa(target)
            transcript_fsa.append(fsa)
            
        transcript_fsa = k2.create_fsa_vec(transcript_fsa)
        return transcript_fsa

    def _get_decoding_graph(self, target, blk_id=0):
        start_state = 0
        next_available_state = 1
        arcs = []

        prev_state = start_state
        blk_state = next_available_state
        next_available_state += 1
        next_state = next_available_state
        next_available_state += 1
        arcs.append([prev_state, blk_state, blk_id, 0])
        arcs.append([blk_state, blk_state, blk_id, 0])
        for word in target:
            for i, token in enumerate(word):
                if i == 0:  # The first token of a word
                    arcs.append([blk_state, next_state, token, 0])
                arcs.append([prev_state, next_state, token, 0])
                arcs.append([next_state, next_state, token, 0])
                prev_state = next_state
                next_state = next_available_state
                next_available_state += 1
            blk_state = next_available_state
            next_available_state += 1
            arcs.append([prev_state, blk_state, blk_id, 0])
            arcs.append([blk_state, blk_state, blk_id, 0])
        end_state = next_available_state
        arcs.append([blk_state, end_state, -1, 0])
        arcs.append([prev_state, end_state, -1, 0])
        arcs.append([end_state])        

        arcs = sorted(arcs, key=lambda arc: arc[0])
        arcs = [[str(i) for i in arc] for arc in arcs]
        arcs = [" ".join(arc) for arc in arcs]
        arcs = "\n".join(arcs)

        fsa = k2.Fsa.from_str(arcs)
        fsa = k2.arc_sort(fsa)
        if self.device is not None:
            fsa = fsa.to(self.device)

        return fsa

    def get_decoding_graph(self, samples):
        targets = [self.sp.encode_keep_boundary(sample[2]) for sample in samples]
        targets = [[[i + 1 for i in w if i != self.padding_value] for w in t] for t in targets]  # hard-coded for torchaudio
        decoding_graphs = []
        for target in targets:
            fsa = self._get_decoding_graph(target)
            decoding_graphs.append(fsa)
            
        decoding_graphs = k2.create_fsa_vec(decoding_graphs)
        return decoding_graphs

    def compile_hmm(
        self,
        samples,
        modified: bool = False,
    ) -> k2.Fsa:
        # # Method 1: through WFST composition
        # transcript_fsa = self.get_transcript_fsa(samples)
        # transcript_fsa = k2.arc_sort(transcript_fsa)

        # decoding_graphs = k2.compose(
        #     self.topo, transcript_fsa, treat_epsilons_specially=True
        # )

        # # Method 2: draw it directly
        decoding_graphs = self.get_decoding_graph(samples)

        decoding_graphs = k2.connect(decoding_graphs)
        decoding_graphs = decoding_graphs.to(self.device)
        
        return decoding_graphs

    def compile(
        self,
        piece_ids: List[List[int]],
        samples,
        modified: bool = False,
    ) -> k2.Fsa:
        if self.topo_type == "ctc":
            return self.compile_ctc(piece_ids, modified)
        elif self.topo_type == "hmm":
            return self.compile_hmm(samples, modified)
        else:
            raise NotImplementedError

