from pathlib import Path
from typing import List, Union
from collections import defaultdict
import string
import math
import k2
import torch
import logging


logging.basicConfig(
    format = "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    level = logging.INFO
)


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
                # if new_node.token == node.token:
                #     new_node.mandatory_blk = True
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
            modified_ctc=True,
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
                if not c.mandatory_blk or modified_ctc or not has_blank_state:
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
                    modified_ctc=modified_ctc,
                )
                next_index = _next_index
                res.extend(_res)
            else:
                if self.is_linear:
                    # Step3-2-1: no recursion, go to the last state immediately
                    if has_blank_state:
                        res.append((blank_state_index, f"{{x + {blank_state_index}}} {{x + {last_index}}} {token} {token + my_aux_offset} {weight}"))
                    if not c.mandatory_blk or modified_ctc or not has_blank_state:
                        res.append((start_index, f"{{x + {start_index}}} {{x + {last_index}}} {token} {token + my_aux_offset} {weight}"))
                    res.append((last_index, f"{{x + {last_index}}} {{x + {last_index}}} {token} {token} {self_loop_bonus}"))
                    next_index += 1
                else:
                    # Step2: the start state or the blank state can go to the next state; the next state has self-loop
                    if has_blank_state:
                        res.append((blank_state_index, f"{{x + {blank_state_index}}} {{x + {next_index}}} {token} {token + my_aux_offset} {weight}"))
                    if not c.mandatory_blk or modified_ctc or not has_blank_state:
                        res.append((start_index, f"{{x + {start_index}}} {{x + {next_index}}} {token} {token + my_aux_offset} {weight}"))
                    res.append((next_index, f"{{x + {next_index}}} {{x + {next_index}}} {token} {token} {self_loop_bonus}"))

                    # Step3-2-2: no recursion
                    if has_blank_state:
                        res.append((blank_state_index, f"{{x + {blank_state_index}}} {{x + {last_index}}} {token} {token + my_aux_offset} {weight}"))
                    if not c.mandatory_blk or modified_ctc or not has_blank_state:
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


def fstr(template, x):
    # https://stackoverflow.com/questions/42497625/how-to-postpone-defer-the-evaluation-of-f-strings
    return eval(f"f'''{template}'''")


class DecodingGraphCompiler(object):
    def __init__(
        self,
        tokenizer,
        lexicon,
        device: Union[str, torch.device] = "cpu",
        topo_type = "ctc",
        index_offset=0,
        sil_penalty_intra_word=0,
        sil_penalty_inter_word=0,
        self_loop_bonus=0,
        aux_offset=0,
        modeling_unit="phoneme",
        modified_ctc=True,  # modified CTC does not require a mandatory blank between repeated tokens
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

        self.tokenizer = tokenizer
        self.lexicon = lexicon
        # self.word_table = k2.SymbolTable.from_file(lang_dir / "words.txt")
        self.device = device

        self.modeling_unit = modeling_unit
        self.topo_type = topo_type
        self.index_offset = index_offset
        self.aux_offset = aux_offset
        self.sil_penalty_intra_word = sil_penalty_intra_word
        self.sil_penalty_inter_word = sil_penalty_inter_word
        self.self_loop_bonus = self_loop_bonus
        self.modified_ctc = modified_ctc

        self.lexicon_fst = self.make_lexicon_fst()

    def make_lexicon_fst(self, lexicon=None):
        lexicon_fst = dict()
        if lexicon is None:
            lexicon = self.tokenizer.lexicon.lexicon

        assert self.tokenizer.blank_id == 0

        for w, plist in lexicon.items():
            trie = Trie()
            for prob, tokens in plist:
                trie.insert(tokens, weight=prob)
            
            res, last_index = trie.to_k2_str_topo(
                token2id=self.tokenizer.token2id, 
                index_offset=self.index_offset, 
                aux_offset=self.aux_offset,
                topo_type=self.topo_type, 
                blank_id=self.tokenizer.blank_id,
                sil_penalty_intra_word=self.sil_penalty_intra_word,
                sil_penalty_inter_word=self.sil_penalty_inter_word,
                self_loop_bonus=self.self_loop_bonus,
                modified_ctc=self.modified_ctc,
            )
            lexicon_fst[w] = (res, last_index)
        return lexicon_fst

    def get_word_fst(self, word):  # this can support unseen words
        if word in self.lexicon_fst:
            return self.lexicon_fst[word]
        elif self.modeling_unit == "char" or self.modeling_unit == "bpe":  # support new words
            # print(f"Adding new word to the lexicon: {word}")
            tokens = self.tokenizer.encode(word, out_type=str)
            lexicon_ = {word: [(1.0, tokens)]}
            lexicon_fst_ = self.make_lexicon_fst(lexicon=lexicon_)
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
                    if self.modeling_unit == "char" or self.modeling_unit == "bpe":
                        res, _next_index = self.get_word_fst(word)
                    else:
                        assert word in self.lexicon_fst, f"{word} does not have lexicon entry"
            fsa_str += "\n"
            fsa_str += fstr("\n".join(res), x = next_index)
            next_index += _next_index

        blank_id = 0
        fsa_str += f"\n{next_index} {next_index + 1} {blank_id} {blank_id} {-self.sil_penalty_inter_word}"
        fsa_str += f"\n{next_index} {next_index + 2} -1 -1 0"
        fsa_str += f"\n{next_index + 1} {next_index + 1} {blank_id} {blank_id} {-self.sil_penalty_inter_word}"
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


def test3():
    logging.getLogger("graphviz").setLevel(logging.WARNING)

    from lexicon import Lexicon
    from tokenizer import Tokenizer

    phone_set = ['ə', 'ɛ', 'd', 'ɪ', 'ɾ', 't', 'm', 'n', 'ɫ', 'i', 'ɫ̩', 'a', 'ɚ', 'ʔ', 'ɹ', 's', 'z', 'ɔ', 'ɐ', 'v', 'spn', 'ej', 'e', 'ɑ', 'ɑː', 'ɒ', 'dʲ', 'iː', 'dʒ', 'vʲ', 'ɒː', 'bʲ', 'tʃ', 'æ', 'b', 'ow', 'aj', 'cʰ', 'p', 'kʰ', 'pʰ', 'k', 'j', 'ʊ', 'ɡ', 'ʎ', 'l', 'w', 'f', 'h', 'ʉː', 'ʉ', 'uː', 'u', 'ɛː', 'ɲ', 'pʲ', 'o', 'əw', 'θ', 'tʲ', 'ʃ', 'c', 'tʰ', 'n̩', 'ŋ', 'ʒ', 'tʷ', 'mʲ', 'ç', 'ɝ', 'ɔj', 'aw', 'ɟ', 'fʲ', 'aː', 'ɜː', 'vʷ', 'kʷ', 'ɜ', 'cʷ', 'ɾʲ', 'ɡb', 'ð', 'ɾ̃', 'kp', 'ɡʷ', 'ɟʷ', 'd̪', 't̪', 'pʷ', 'm̩', 'fʷ']
    token2id = {p: i + 1 for i, p in enumerate(phone_set)}
    token2id["-"] = 0

    aux_offset = 1000000
    # sil_penalty_intra_word = 0.5
    sil_penalty_intra_word = 100000
    sil_penalty_inter_word = 0.1
    self_loop_bonus = 0
    topo_type = "ctc"
    modeling_unit = "phoneme"

    lexicon = Lexicon(
        files=[
            "/exp/rhuang/meta/audio_ruizhe/librispeech_conformer_ctc/librispeech_english_us_mfa.prob.dict",
            "/exp/rhuang/meta/audio_ruizhe/librispeech_conformer_ctc/librispeech_english_us_mfa.new_words.dict",
            "/exp/rhuang/buckeye/datasets/Buckeye_Corpus2/buckeye_words.dict",
        ]
    )

    tokenizer = Tokenizer(
        has_boundary=False,
        modeling_unit=modeling_unit,
        lexicon=lexicon,
        token2id=token2id,
        blank_token='-',
        unk_token='spn',
        sp_model_path=None,
    )

    compiler = DecodingGraphCompiler(
        tokenizer=tokenizer,
        lexicon=lexicon,
        topo_type = "ctc",
        index_offset=0,
        sil_penalty_intra_word=sil_penalty_intra_word,
        sil_penalty_inter_word=sil_penalty_inter_word,
        self_loop_bonus=self_loop_bonus,
        aux_offset=aux_offset,
        modeling_unit=modeling_unit,
        modified_ctc=True,
    )

    lexicon = lexicon.lexicon

    for k, v in list(token2id.items()):
        token2id[f"▁{k}"] = v + aux_offset

    text = "pen pineapple apple pen"
    # text = "THAT THE HEBREWS WERE RESTIVE UNDER THIS TYRANNY WAS NATURAL INEVITABLE"
    # text = "boy they'd pay for my high school my college and everything yknow even living expenses yknow but because i have a bachelors"

    if False:
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
        
        fsa = k2.Fsa.from_str(fsa_str.strip(), acceptor=False)
    else:
        fsas = compiler.compile(None, [[None, None, text]])
        fsa = fsas[0]

    fsa.labels_sym = k2.SymbolTable.from_str("\n".join([f"{k} {v}" for k, v in token2id.items()]))
    fsa.aux_labels_sym = fsa.labels_sym

    fsa.draw('fsa_symbols.svg', title='An FSA with symbol table')


def test4():
    logging.getLogger("graphviz").setLevel(logging.WARNING)

    from lexicon import Lexicon
    from tokenizer import Tokenizer
    import sentencepiece as spm

    # phone_set = ['ə', 'ɛ', 'd', 'ɪ', 'ɾ', 't', 'm', 'n', 'ɫ', 'i', 'ɫ̩', 'a', 'ɚ', 'ʔ', 'ɹ', 's', 'z', 'ɔ', 'ɐ', 'v', 'spn', 'ej', 'e', 'ɑ', 'ɑː', 'ɒ', 'dʲ', 'iː', 'dʒ', 'vʲ', 'ɒː', 'bʲ', 'tʃ', 'æ', 'b', 'ow', 'aj', 'cʰ', 'p', 'kʰ', 'pʰ', 'k', 'j', 'ʊ', 'ɡ', 'ʎ', 'l', 'w', 'f', 'h', 'ʉː', 'ʉ', 'uː', 'u', 'ɛː', 'ɲ', 'pʲ', 'o', 'əw', 'θ', 'tʲ', 'ʃ', 'c', 'tʰ', 'n̩', 'ŋ', 'ʒ', 'tʷ', 'mʲ', 'ç', 'ɝ', 'ɔj', 'aw', 'ɟ', 'fʲ', 'aː', 'ɜː', 'vʷ', 'kʷ', 'ɜ', 'cʷ', 'ɾʲ', 'ɡb', 'ð', 'ɾ̃', 'kp', 'ɡʷ', 'ɟʷ', 'd̪', 't̪', 'pʷ', 'm̩', 'fʷ']
    # token2id = {p: i + 1 for i, p in enumerate(phone_set)}
    # token2id["-"] = 0
    # blank_token = "-"
    # unk_token = "spn"
    # modeling_unit = "phoneme"
    # sp_model_path = None

    # token2id = {'-': 0, '@': 1, 'e': 2, 't': 3, 'a': 4, 'o': 5, 'n': 6, 'i': 7, 'h': 8, 's': 9, 'r': 10, 'd': 11, 'l': 12, 'u': 13, 'm': 14, 'w': 15, 'c': 16, 'f': 17, 'g': 18, 'y': 19, 'p': 20, 'b': 21, 'v': 22, 'k': 23, "'": 24, 'x': 25, 'j': 26, 'q': 27, 'z': 28}
    # blank_token = '-'
    # unk_token = '@'
    # modeling_unit = "char"
    # sp_model_path = None

    sp_model_path = "/exp/rhuang/meta/audio/examples/asr/librispeech_conformer_ctc/spm_unigram_1023.model"
    sp_model = spm.SentencePieceProcessor(model_file=str(sp_model_path))
    token2id = {sp_model.id_to_piece(i): i + 1 for i in range(sp_model.vocab_size())}
    assert "-" not in token2id
    token2id["-"] = 0
    blank_token = '-'
    unk_token = '<unk>'
    modeling_unit = "bpe"

    

    aux_offset = 1000000
    sil_penalty_intra_word = 0.5
    # sil_penalty_intra_word = 100000
    sil_penalty_inter_word = 0.1
    self_loop_bonus = 0
    topo_type = "ctc"

    lexicon = Lexicon(
        files=[
            "/exp/rhuang/meta/audio_ruizhe/librispeech_conformer_ctc/librispeech_english_us_mfa.prob.dict",
            "/exp/rhuang/meta/audio_ruizhe/librispeech_conformer_ctc/librispeech_english_us_mfa.new_words.dict",
            "/exp/rhuang/buckeye/datasets/Buckeye_Corpus2/buckeye_words.dict",
        ],
        modeling_unit=modeling_unit,
    )

    tokenizer = Tokenizer(
        has_boundary=False,
        modeling_unit=modeling_unit,
        lexicon=lexicon,
        token2id=token2id,
        blank_token=blank_token,
        unk_token=unk_token,
        sp_model_path=sp_model_path,
    )

    if modeling_unit != "phoneme":
        lexicon.populate_lexicon_with_tokenizer(tokenizer)

    compiler = DecodingGraphCompiler(
        tokenizer=tokenizer,
        lexicon=lexicon,
        topo_type=topo_type,
        index_offset=0,
        sil_penalty_intra_word=sil_penalty_intra_word,
        sil_penalty_inter_word=sil_penalty_inter_word,
        self_loop_bonus=self_loop_bonus,
        aux_offset=aux_offset,
        modeling_unit=modeling_unit,
        modified_ctc=True,
    )

    lexicon = lexicon.lexicon

    for k, v in list(token2id.items()):
        # token2id[f"▁{k}"] = v + aux_offset
        token2id[f"_{k}"] = v + aux_offset

    # text = "pen pineapple apple pen"
    text = "pen pineappla apple pen"
    # text = "THAT THE HEBREWS WERE RESTIVE UNDER THIS TYRANNY WAS NATURAL INEVITABLE"
    # text = "boy they'd pay for my high school my college and everything yknow even living expenses yknow but because i have a bachelors"

    fsas = compiler.compile(None, [[None, None, text]])
    fsa = fsas[0]

    fsa.labels_sym = k2.SymbolTable.from_str("\n".join([f"{k} {v}" for k, v in token2id.items()]))
    fsa.aux_labels_sym = fsa.labels_sym

    fsa.draw('fsa_symbols.svg', title='An FSA with symbol table')


if __name__ == "__main__":
    # test3()
    test4()
