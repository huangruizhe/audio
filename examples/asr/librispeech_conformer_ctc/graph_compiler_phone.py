from pathlib import Path
from typing import List, Union
from collections import defaultdict
from lexicon import read_lexicon, Trie, fstr

import k2
import torch


class PhonemeCtcTrainingGraphCompiler(object):
    def __init__(
        self,
        bpe_model,
        device: Union[str, torch.device] = "cpu",
        topo_type = "ctc",
        index_offset=1,  # for torchaudio's non-zero blank id
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

        self.max_token_id = None
        self.topo = None

        self.topo_type = topo_type

        self.lexicon_fst = self.make_lexicon_fst(index_offset=index_offset, topo_type=topo_type)

    def make_lexicon_fst(self, index_offset=1, topo_type="ctc"):
        _lexicon = dict()
        for w, plist in self.sp.lexicon.items():
            trie = Trie()
            for prob, tokens in plist:
                trie.insert(tokens, weight=prob)
            
            res, next_index, last_index = trie.to_k2_str_topo(token2id=self.sp.token2id, index_offset=index_offset, topo_type=topo_type, blank_id=0)
            _lexicon[w] = (res, next_index)
        return _lexicon

    def _get_decoding_graph(self, sentence):
        next_index = 0
        fsa_str = ""
        sentence = sentence.strip().lower().split()
        
        for word in sentence:
            res, _next_index = self.lexicon_fst[word]
            fsa_str += "\n"
            fsa_str += self.sp.fstr("\n".join(res), x = next_index)
            next_index += _next_index

        blank_id = 0
        fsa_str += f"\n{next_index} {next_index + 1} {blank_id} {blank_id} 0"
        fsa_str += f"\n{next_index} {next_index + 2} -1 -1 0"
        fsa_str += f"\n{next_index + 1} {next_index + 1} {blank_id} {blank_id} 0"
        fsa_str += f"\n{next_index + 1} {next_index + 2} -1 -1 0"
        fsa_str += f"\n{next_index + 2}"
        fsa_str = fsa_str.strip()

        fsa = k2.Fsa.from_str(fsa_str, acceptor=False)
        # fsa.labels_sym = k2.SymbolTable.from_str("\n".join([f"{k} {v}" for k, v in self.sp.token2id.items()]))
        # fsa.aux_labels_sym = fsa.labels_sym
        return fsa

    def compile(
        self,
        piece_ids: List[List[int]],
        samples,
    ) -> k2.Fsa:
        
        targets = [sample[2] for sample in samples]
        decoding_graphs = []
        for target in targets:
            fsa = self._get_decoding_graph(target)
            decoding_graphs.append(fsa)
            
        decoding_graphs = k2.create_fsa_vec(decoding_graphs)
        decoding_graphs = k2.connect(decoding_graphs)
        decoding_graphs = decoding_graphs.to(self.device)
        return decoding_graphs

