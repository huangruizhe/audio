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
        sil_penalty_intra_word=0,
        sil_penalty_inter_word=0,
        aux_offset=0,
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
        self.index_offset = index_offset
        self.sil_penalty_intra_word = sil_penalty_intra_word
        self.sil_penalty_inter_word = sil_penalty_inter_word

        self.lexicon_fst = self.make_lexicon_fst(
            index_offset=index_offset, 
            topo_type=topo_type, 
            sil_penalty_intra_word=sil_penalty_intra_word,
            sil_penalty_inter_word=sil_penalty_inter_word,
            aux_offset=aux_offset,
        )

    def make_lexicon_fst(self, lexicon=None, index_offset=1, topo_type="ctc", sil_penalty_intra_word=0, sil_penalty_inter_word=0, aux_offset=0):
        lexicon_fst = dict()
        if lexicon is None:
            lexicon = self.sp.lexicon
        for w, plist in lexicon.items():
            trie = Trie()
            for prob, tokens in plist:
                trie.insert(tokens, weight=prob)
            
            res, next_index, last_index = trie.to_k2_str_topo(
                token2id=self.sp.token2id, 
                index_offset=index_offset, 
                topo_type=topo_type, 
                blank_id=0,
                sil_penalty_intra_word=sil_penalty_intra_word,
                sil_penalty_inter_word=sil_penalty_inter_word,
                aux_offset=aux_offset,
            )
            lexicon_fst[w] = (res, next_index)
        return lexicon_fst

    def get_fst(self, word):
        if word in self.lexicon_fst:
            return self.lexicon_fst[word]
        else:  # support new words
            print(f"Adding new word to the lexicon: {word}")
            lexicon_ = self.sp.add_new_word(word)
            lexicon_fst_ = self.make_lexicon_fst(
                lexicon=lexicon_,
                index_offset=self.index_offset, 
                topo_type=self.topo_type, 
                sil_penalty_intra_word=self.sil_penalty_intra_word,
                sil_penalty_inter_word=self.sil_penalty_inter_word,
            )
            self.lexicon_fst.update(lexicon_fst_)
            return self.lexicon_fst[word]

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
        fsa_str += f"\n{next_index} {next_index + 1} {blank_id} {blank_id} 0"  # TODO: {-sil_penalty_inter_word}
        fsa_str += f"\n{next_index} {next_index + 2} -1 -1 0"
        fsa_str += f"\n{next_index + 1} {next_index + 1} {blank_id} {blank_id} 0"  # TODO: {-sil_penalty_inter_word}
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

