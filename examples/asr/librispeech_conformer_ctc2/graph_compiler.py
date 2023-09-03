from pathlib import Path
from typing import List, Union
from collections import defaultdict
import string
import math
import k2
import torch


class TrieNode:
    def __init__(self, token):
        self.token = token
        self.is_end = False
        self.state_id = None
        self.weight = -1e9
        self.children = {}
        self.mandatory_blk = False  # True if a mandatory blank is needed between this token and its parent


class Trie(object):
    # https://albertauyeung.github.io/2020/06/15/python-trie.html/

    def __init__(self):
        self.root = TrieNode("")
        self.is_linear = False
    
    def insert(self, word_tokens, weight=1.0, prev_token=None):
        """Insert a word into the trie"""
        node = self.root
        
        # Loop through each token in the word
        # Check if there is no child containing the token, create a new child for the current node
        # prev_token = None
        for token in word_tokens:
            if token in node.children:
                node = node.children[token]
                node.weight = weight + node.weight
            else:
                # If a character is not found,
                # create a new node in the trie
                new_node = TrieNode(token)
                node.children[token] = new_node
                if new_node.token == node.token:
                    new_node.mandatory_blk = True
                node = new_node
                node.weight = weight
            
            if token == prev_token:
                node.mandatory_blk = True
            prev_token = token
        
        # Mark the end of a word
        node.is_end = True
    
    def print(self, node=None, last=True, header=''):
        elbow = "└──"
        pipe = "│  "
        tee = "├──"
        blank = "   "
        if node is None:
            node = self.root
        print(header + (elbow if last else tee) + f"{node.token}:{node.weight}")
        if len(node.children) > 0:
            for i, (label, c) in enumerate(node.children.items()):
                self.print(node=c, header=header + (blank if last else pipe), last=i == len(node.children) - 1)
    
    def to_k2_str_topo(
            self,
            node=None,
            start_index=0,
            last_index=-1,
            token2id=None,
            index_offset=0,
            topo_type="ctc",
            sil_penalty_intra_word=0,
            sil_penalty_inter_word=0,
            self_loop_bonus=0,
            blank_id=0,
            aux_offset=0,
        ):
        
        if node is None:
            node = self.root

            cnt_non_leaves = 1  # count the number of non-leaf nodes
            cnt_leaves = 0  # count the number of leaves
            leaves_labels = []
            temp_list = list(self.root.children.values())
            while len(temp_list) > 0:
                n = temp_list.pop()
                if len(n.children) > 0:
                    cnt_non_leaves += 1
                    temp_list.extend(n.children.values())
                else:
                    cnt_leaves += 1
                    leaves_labels.append(n.token)
            
            last_index = start_index + cnt_non_leaves * 2 + cnt_leaves
            leaves_labels_0 = leaves_labels[0]
            self.is_linear = all(leaves_labels_0 == x for x in leaves_labels)

        res = []
        
        next_index = start_index + 1  # next_index is the next availabe state id
        blank_state_index = next_index
        next_index += 1

        penalty_threshold = 1e3
        has_blank_state = False

        # Step1: the start state can go to the blank state
        if node == self.root:  # inter-word blank at the beginning of each word/trie
            if sil_penalty_inter_word < penalty_threshold:
                res.append((start_index, f"{{x + {start_index}}} {{x + {blank_state_index}}} {blank_id} {blank_id} {-sil_penalty_inter_word}"))
                res.append((blank_state_index, f"{{x + {blank_state_index}}} {{x + {blank_state_index}}} {blank_id} {blank_id} {-sil_penalty_inter_word}"))
                has_blank_state = True
        else:  # intra-word blank
            if sil_penalty_intra_word < penalty_threshold:
                res.append((start_index, f"{{x + {start_index}}} {{x + {blank_state_index}}} {blank_id} {blank_id} {-sil_penalty_intra_word}"))
                res.append((blank_state_index, f"{{x + {blank_state_index}}} {{x + {blank_state_index}}} {blank_id} {blank_id} {-sil_penalty_intra_word}"))
                has_blank_state = True

        for i, (label, c) in enumerate(node.children.items()):
            token = token2id[c.token] + index_offset
            weight = math.log(c.weight)
            is_not_leaf = (len(c.children) > 0)
            my_aux_offset = aux_offset if node == self.root else 0

            if is_not_leaf:
                # Step2: the start state or the blank state can go to the next state; the next state has self-loop
                if has_blank_state:
                    res.append((blank_state_index, f"{{x + {blank_state_index}}} {{x + {next_index}}} {token} {token + my_aux_offset} {weight}"))
                if not c.mandatory_blk or not has_blank_state:
                    res.append((start_index, f"{{x + {start_index}}} {{x + {next_index}}} {token} {token + my_aux_offset} {weight}"))
                res.append((next_index, f"{{x + {next_index}}} {{x + {next_index}}} {token} {token} {self_loop_bonus}"))
                
                # Step3-1: recursion
                _res, _next_index = self.to_k2_str_topo(
                    node=c,
                    start_index=next_index,
                    last_index=last_index,
                    token2id=token2id,
                    index_offset=index_offset,
                    topo_type=topo_type,
                    sil_penalty_intra_word=sil_penalty_intra_word,
                    sil_penalty_inter_word=sil_penalty_inter_word,
                    self_loop_bonus=self_loop_bonus,
                    blank_id=blank_id,
                    aux_offset=aux_offset,
                )
                next_index = _next_index
                res.extend(_res)
            else:
                if self.is_linear:
                    # Step3-2-1: no recursion
                    if has_blank_state:
                        res.append((blank_state_index, f"{{x + {blank_state_index}}} {{x + {last_index}}} {token} {token + my_aux_offset} {weight}"))
                    if not c.mandatory_blk or not has_blank_state:
                        res.append((start_index, f"{{x + {start_index}}} {{x + {last_index}}} {token} {token + my_aux_offset} {weight}"))
                    res.append((last_index, f"{{x + {last_index}}} {{x + {last_index}}} {token} {token} {self_loop_bonus}"))
                    next_index += 1
                else:
                    # Step2: the start state or the blank state can go to the next state; the next state has self-loop
                    if has_blank_state:
                        res.append((blank_state_index, f"{{x + {blank_state_index}}} {{x + {next_index}}} {token} {token + my_aux_offset} {weight}"))
                    if not c.mandatory_blk or not has_blank_state:
                        res.append((start_index, f"{{x + {start_index}}} {{x + {next_index}}} {token} {token + my_aux_offset} {weight}"))
                    res.append((next_index, f"{{x + {next_index}}} {{x + {next_index}}} {token} {token} {self_loop_bonus}"))

                    # Step3-2-2: no recursion
                    if has_blank_state:
                        res.append((blank_state_index, f"{{x + {blank_state_index}}} {{x + {last_index}}} {token} {token + my_aux_offset} {weight}"))
                    if not c.mandatory_blk or not has_blank_state:
                        res.append((start_index, f"{{x + {start_index}}} {{x + {last_index}}} {token} {token + my_aux_offset} {weight}"))
                    res.append((next_index, f"{{x + {next_index}}} {{x + {last_index}}} {token} {token} {self_loop_bonus}"))
                    next_index += 1
            
        if node == self.root:
            # res.sort()
            res = sorted(set(res))
            res = [r[1] for r in res]
            # assert next_index == last_index, f"{next_index} vs. {last_index}"
        
        if node == self.root:
            return res, last_index
        else:
            return res, next_index


class DecodingGraphCompiler(object):
    def __init__(
        self,
        tokenizer,
        lexicon,
        device: Union[str, torch.device] = "cpu",
        topo_type = "ctc",
        index_offset=1,  # for torchaudio's non-zero blank id
        sil_penalty_intra_word=0,
        sil_penalty_inter_word=0,
        aux_offset=0,
        modeling_unit="phoneme",
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

        sp = tokenizer
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
            try:
                res, _next_index = self.lexicon_fst[word]
            except:
                word_ = word.translate(str.maketrans('', '', string.punctuation))
                if word_ in self.lexicon_fst:
                    res, _next_index = self.lexicon_fst[word_]
                else:
                    assert word in self.lexicon_fst, f"{word} does not have lexicon entry"
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


def fstr(template, x):
    # https://stackoverflow.com/questions/42497625/how-to-postpone-defer-the-evaluation-of-f-strings
    return eval(f"f'''{template}'''")


def test3():
    import logging
    logging.getLogger("graphviz").setLevel(logging.WARNING)

    from lexicon import Lexicon

    phone_set = ['ə', 'ɛ', 'd', 'ɪ', 'ɾ', 't', 'm', 'n', 'ɫ', 'i', 'ɫ̩', 'a', 'ɚ', 'ʔ', 'ɹ', 's', 'z', 'ɔ', 'ɐ', 'v', 'spn', 'ej', 'e', 'ɑ', 'ɑː', 'ɒ', 'dʲ', 'iː', 'dʒ', 'vʲ', 'ɒː', 'bʲ', 'tʃ', 'æ', 'b', 'ow', 'aj', 'cʰ', 'p', 'kʰ', 'pʰ', 'k', 'j', 'ʊ', 'ɡ', 'ʎ', 'l', 'w', 'f', 'h', 'ʉː', 'ʉ', 'uː', 'u', 'ɛː', 'ɲ', 'pʲ', 'o', 'əw', 'θ', 'tʲ', 'ʃ', 'c', 'tʰ', 'n̩', 'ŋ', 'ʒ', 'tʷ', 'mʲ', 'ç', 'ɝ', 'ɔj', 'aw', 'ɟ', 'fʲ', 'aː', 'ɜː', 'vʷ', 'kʷ', 'ɜ', 'cʷ', 'ɾʲ', 'ɡb', 'ð', 'ɾ̃', 'kp', 'ɡʷ', 'ɟʷ', 'd̪', 't̪', 'pʷ', 'm̩', 'fʷ']
    token2id = {p: i + 1 for i, p in enumerate(phone_set)}
    token2id["-"] = 0

    lexicon = Lexicon(
        files=[
            "/exp/rhuang/meta/audio_ruizhe/librispeech_conformer_ctc/librispeech_english_us_mfa.prob.dict",
            "/exp/rhuang/meta/audio_ruizhe/librispeech_conformer_ctc/librispeech_english_us_mfa.new_words.dict",
            "/exp/rhuang/buckeye/datasets/Buckeye_Corpus2/buckeye_words.dict",
        ]
    )
    lexicon = lexicon.lexicon

    aux_offset = 1000000
    for k, v in list(token2id.items()):
        token2id[f"▁{k}"] = v + aux_offset

    sil_penalty_intra_word = 0.5
    sil_penalty_inter_word = 0.1
    self_loop_bonus = 0
    topo_type = "ctc"

    text = "pen pineapple apple pen"
    # text = "THAT THE HEBREWS WERE RESTIVE UNDER THIS TYRANNY WAS NATURAL INEVITABLE"
    # text = "boy they'd pay for my high school my college and everything yknow even living expenses yknow but because i have a bachelors"
    
    _lexicon = dict()
    for w in text.strip().lower().split():
        trie = Trie()
        for prob, tokens in lexicon[w]:
            trie.insert(tokens, weight=prob)
        
        res, last_index = trie.to_k2_str_topo(
            token2id=token2id, 
            index_offset=0, 
            topo_type=topo_type, 
            sil_penalty_intra_word=sil_penalty_intra_word, 
            sil_penalty_inter_word=sil_penalty_inter_word, 
            self_loop_bonus=self_loop_bonus,
            blank_id=token2id["-"], 
            aux_offset=aux_offset
        )
        _lexicon[w] = (res, last_index)

    fsa_str = ""
    next_index = 0

    for w in text.strip().lower().split():
        res, _next_index = _lexicon[w]
        fsa_str += "\n"
        fsa_str += fstr("\n".join(res), x = next_index)
        # print(w)
        # print(fstr("\n".join(res), x = next_index))
        next_index += _next_index

    blank_id = token2id["-"]
    fsa_str += f"\n{next_index} {next_index + 1} {blank_id} {blank_id} {-sil_penalty_inter_word}"
    fsa_str += f"\n{next_index} {next_index + 2} -1 -1 0"
    fsa_str += f"\n{next_index + 1} {next_index + 1} {blank_id} {blank_id} {-sil_penalty_inter_word}"
    fsa_str += f"\n{next_index + 1} {next_index + 2} -1 -1 0"
    fsa_str += f"\n{next_index + 2}"
    # print(res)

    import k2
    fsa = k2.Fsa.from_str(fsa_str.strip(), acceptor=False)
    fsa.labels_sym = k2.SymbolTable.from_str("\n".join([f"{k} {v}" for k, v in token2id.items()]))
    fsa.aux_labels_sym = fsa.labels_sym

    fsa.draw('fsa_symbols.svg', title='An FSA with symbol table')


if __name__ == "__main__":
    # test1()
    # test2()
    test3()

    # TODO:
    # 1. compose the `pronunciation graph` in test1 with `k2.ctc_topo`, we should get the results in test3