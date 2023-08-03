import logging
import re
import sys
from pathlib import Path
from fastnumbers import check_float
from collections import defaultdict
import math
import subprocess


def read_lexicon(filename: str, has_boundary=False, quiet=False):
    """Read a lexicon from `filename`.

    Each line in the lexicon contains "word p1 p2 p3 ...".
    That is, the first field is a word and the remaining
    fields are tokens. Fields are separated by space(s).

    Args:
      filename:
        Path to the lexicon.txt

    Returns:
      A dictionary, e.g., 
      {
        'w0': [(s00, ['p1', 'p2']), (s01, ['p1', 'p3'])], 
        'w1': [(s10, ['p3, 'p4'])],
        ...
      }
    """
    ans = defaultdict(list)
    token2id = dict()

    with open(filename, "r", encoding="utf-8") as fin:
        # whitespace = re.compile("[ \t]+")
        for line in fin:
            a = line.strip()
            # a = line.strip().split()
            if len(a) == 0:
                continue
            a = a.split("\t")

            if len(a) != 2 and len(a) != 6:
                logging.info(f"Found bad line {line} in lexicon file {filename}")
                logging.info("Every line is expected to contain at least 2 fields")
                sys.exit(1)
            word = a[0]
            if word == "<eps>":
                logging.info(f"Found bad line {line} in lexicon file {filename}")
                logging.info("<eps> should not be a valid word")
                sys.exit(1)

            # tokens = a[1:]
            # prob = 1.0  # probability
            # if check_float(tokens[0]):
            #     prob = float(tokens[0])
            #     tokens = tokens[1:]
            
            if len(a) == 2:
                tokens = a[1].split()
                prob = 1.0
            elif len(a) == 6:
                tokens = a[5].split()
                prob = float(a[1])
            
            if has_boundary:
                tokens[0] = f"▁{tokens[0]}"

            for t in tokens:
                if t not in token2id:
                    token2id[t] = len(token2id)

            ans[word].append([prob, tokens])

    # Normalization
    for word, pron_list in ans.items():
        total = sum([prob for prob, tokens in pron_list])
        for entry in pron_list:
            entry[0] /= total 

    if not quiet:
        total_entries = sum([len(v) for k, v in ans.items()])
        # logging.info(f"Number of words in dict: {len(ans)}")
        # logging.info(f"Number of tokens in dict: {len(token2id)}")
        # logging.info(f"Average number of pronunciations per word: {total_entries/len(ans):.2f}")
        print(f"Number of words in dict: {len(ans)}")
        print(f"Number of tokens in dict: {len(token2id)}")
        print(f"Average number of pronunciations per word: {total_entries/len(ans):.2f}")
        print(f"Tokens: {list(token2id.keys())}")

    return ans, token2id


def call_mfa_g2p_cmd(words, g2p_model="english_us_mfa"):
    # The input is a list of words

    # https://stackabuse.com/executing-shell-commands-with-python/
    mfa_g2p_call = subprocess.Popen(["mfa", "g2p", "-", g2p_model, "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, errors = mfa_g2p_call.communicate(input="\n".join(words))
    if mfa_g2p_call.wait() == 0:
        return output.strip().split("\n")
    else:
        print(errors)
        exit(1)


def call_mfa_g2p_python(words, g2p_model="english_us_mfa"):
    # https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/blob/main/montreal_forced_aligner/command_line/g2p.py
    pass


def call_mfa_g2p(words, g2p_model="english_us_mfa"):
    return call_mfa_g2p_python(words, g2p_model=g2p_model)


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
    
    def to_k2_str_pron(self, node=None, start_index=0, last_index=-1, token2id=None, index_offset=1):
        if node is None:
            node = self.root

            cnt = 0
            temp_list = list(self.root.children.values())
            while len(temp_list) > 0:
                n = temp_list.pop()
                if len(n.children) > 0:
                    cnt += 1
                    temp_list.extend(n.children.values())
            
            last_index = start_index + cnt + 1

        res = []

        next_index = start_index + 1
        for i, (label, c) in enumerate(node.children.items()):
            if token2id is None:
                token = c.token
            else:
                token = token2id[c.token] + index_offset
            # weight = math.log(c.weight)
            weight = c.weight

            if len(c.children) > 0:
                res.append((start_index, f"{{x + {start_index}}} {{x + {next_index}}} {token} {token} {weight}"))
                _res, _next_index, _last_index = self.to_k2_str_pron(node=c, start_index=next_index, last_index=last_index, token2id=token2id, index_offset=index_offset)
                next_index = _next_index
                res.extend(_res)
            else:
                res.append((start_index, f"{{x + {start_index}}} {{x + {last_index}}} {token} {token} {weight}"))
            
        if node == self.root:
            res.sort()
            res = [r[1] for r in res]
            assert next_index == last_index
            
        return res, next_index, last_index
    
    def to_k2_str_topo(self, node=None, start_index=0, last_index=-1, token2id=None, index_offset=1, topo_type="ctc", sil_penalty_intra_word=0, sil_penalty_inter_word=0, blank_id = 0, aux_offset=0, mandatory_blk=False):
        # TODO: simplify/unify making hmm and ctc decoding graphs
        if node is None:
            node = self.root

            cnt = 1  # count the number of non-leaf nodes
            # cnt_children = 0  # count the number of children
            cnt_leaves = 0  # count the number of leaves
            temp_list = list(self.root.children.values())
            while len(temp_list) > 0:
                n = temp_list.pop()
                # cnt_children += 1
                if len(n.children) > 0:
                    cnt += 1
                    temp_list.extend(n.children.values())
                else:
                    cnt_leaves += 1
            
            if topo_type == "hmm":
                last_index = start_index + cnt + 1 + cnt_leaves
            else:
                last_index = start_index + cnt + cnt + cnt_leaves

        res = []
        
        next_index = start_index + 1  # next_index is the next availabe state id
        if topo_type == "hmm" and node != self.root:
            pass
        else:
            blank_state_index = next_index
            next_index += 1

        for i, (label, c) in enumerate(node.children.items()):
            if token2id is None:
                token = c.token
            else:
                token = token2id[c.token] + index_offset
            weight = math.log(c.weight)

            if topo_type == "hmm" and node != self.root:
                # No intra-word blank/silence
                if len(c.children) > 0:
                    res.append((start_index, f"{{x + {start_index}}} {{x + {next_index}}} {token} {token} {weight}"))
                    res.append((next_index, f"{{x + {next_index}}} {{x + {next_index}}} {token} {token} 0"))
                    _res, _next_index, _last_index = self.to_k2_str_topo(node=c, start_index=next_index, last_index=last_index, token2id=token2id, index_offset=index_offset, topo_type=topo_type, sil_penalty_intra_word=sil_penalty_intra_word, sil_penalty_inter_word=sil_penalty_inter_word, blank_id=blank_id)
                    next_index = _next_index
                    res.extend(_res)
                else:
                    res.append((start_index, f"{{x + {start_index}}} {{x + {last_index}}} {token} {token} {weight}"))
                    # res.append((last_index, f"{{x + {last_index}}} {{x + {last_index}}} {token} {token} 0"))
                    res.append((start_index, f"{{x + {start_index}}} {{x + {next_index}}} {token} {token} {weight}"))
                    res.append((next_index, f"{{x + {next_index}}} {{x + {next_index}}} {token} {token} 0"))
                    res.append((next_index, f"{{x + {next_index}}} {{x + {last_index}}} {token} {token} 0"))
                    next_index += 1
            else:
                # There is intra-word blank/silence with some probability
                if len(c.children) > 0:
                    if i == 0:
                        if node == self.root:  # inter-word blank at the beginning of each word/trie
                            res.append((start_index, f"{{x + {start_index}}} {{x + {blank_state_index}}} {blank_id} {blank_id} {-sil_penalty_inter_word}"))
                            res.append((blank_state_index, f"{{x + {blank_state_index}}} {{x + {blank_state_index}}} {blank_id} {blank_id} {-sil_penalty_inter_word}"))
                        else:  # intra-word blank
                            res.append((start_index, f"{{x + {start_index}}} {{x + {blank_state_index}}} {blank_id} {blank_id} {-sil_penalty_intra_word}"))
                            res.append((blank_state_index, f"{{x + {blank_state_index}}} {{x + {blank_state_index}}} {blank_id} {blank_id} {-sil_penalty_intra_word}"))
                    _aux_offset = aux_offset if node == self.root else 0
                    res.append((blank_state_index, f"{{x + {blank_state_index}}} {{x + {next_index}}} {token} {token + _aux_offset} {weight}"))
                    if not c.mandatory_blk:
                        res.append((start_index, f"{{x + {start_index}}} {{x + {next_index}}} {token} {token + _aux_offset} {weight}"))
                    res.append((next_index, f"{{x + {next_index}}} {{x + {next_index}}} {token} {token} 0"))
                    _res, _next_index, _last_index = self.to_k2_str_topo(node=c, start_index=next_index, last_index=last_index, token2id=token2id, index_offset=index_offset, topo_type=topo_type, sil_penalty_intra_word=sil_penalty_intra_word, sil_penalty_inter_word=sil_penalty_inter_word, blank_id=blank_id)
                    next_index = _next_index
                    res.extend(_res)
                else:
                    if i == 0:
                        if node == self.root:  # inter-word blank at the beginning of each word/trie
                            res.append((start_index, f"{{x + {start_index}}} {{x + {blank_state_index}}} {blank_id} {blank_id} {-sil_penalty_inter_word}"))
                            res.append((blank_state_index, f"{{x + {blank_state_index}}} {{x + {blank_state_index}}} {blank_id} {blank_id} {-sil_penalty_inter_word}"))
                        else:  # intra-word blank
                            res.append((start_index, f"{{x + {start_index}}} {{x + {blank_state_index}}} {blank_id} {blank_id} {-sil_penalty_intra_word}"))
                            res.append((blank_state_index, f"{{x + {blank_state_index}}} {{x + {blank_state_index}}} {blank_id} {blank_id} {-sil_penalty_intra_word}"))
                    # res.append((blank_state_index, f"{{x + {blank_state_index}}} {{x + {last_index}}} {token} {token} {0}"))
                    # res.append((start_index, f"{{x + {start_index}}} {{x + {last_index}}} {token} {token} {weight}"))
                    # res.append((last_index, f"{{x + {last_index}}} {{x + {last_index}}} {token} {token} 0"))
                    _aux_offset = aux_offset if node == self.root else 0
                    res.append((blank_state_index, f"{{x + {blank_state_index}}} {{x + {last_index}}} {token} {token + _aux_offset} {weight}"))
                    res.append((blank_state_index, f"{{x + {blank_state_index}}} {{x + {next_index}}} {token} {token + _aux_offset} {weight}"))
                    if not c.mandatory_blk:
                        res.append((start_index, f"{{x + {start_index}}} {{x + {last_index}}} {token} {token + _aux_offset} {weight}"))
                        res.append((start_index, f"{{x + {start_index}}} {{x + {next_index}}} {token} {token + _aux_offset} {weight}"))
                    res.append((next_index, f"{{x + {next_index}}} {{x + {next_index}}} {token} {token} 0"))
                    res.append((next_index, f"{{x + {next_index}}} {{x + {last_index}}} {token} {token} 0"))
                    next_index += 1

            
        if node == self.root:
            # res.sort()
            res = sorted(set(res))
            res = [r[1] for r in res]
            assert next_index == last_index, f"{next_index} vs. {last_index}"
            
        return res, next_index, last_index


def fstr(template, x):
    # https://stackoverflow.com/questions/42497625/how-to-postpone-defer-the-evaluation-of-f-strings
    return eval(f"f'''{template}'''")


def test1():
    lexicon, token2id = read_lexicon("/fsx/users/huangruizhe/audio_ruizhe/librispeech_conformer_ctc/test.dict")
    # id2token = {v: k for k, v in token2id.items()}
    # print(lexicon)

    trie = Trie()
    trie.insert(['f', 'ɛ', 't', 'n', 'ɑː', 'z'])
    trie.insert(['f', 'ɛ', 't', 'n', 'ɑ', 'z'])
    trie.insert(['fʲ', 'i', 'tʰ', 'n', 'ə', 'z'])

    # trie.print()
    res, next_index, last_index = trie.to_k2_str_pron(token2id=token2id, index_offset=0)
    res = "\n".join(res)
    res = fstr(res, x=0)
    res += f"\n{last_index} {last_index + 1} -1 -1 0\n{last_index + 1}"
    # print(res)

    import k2
    fsa = k2.Fsa.from_str(res, acceptor=False)
    fsa.labels_sym = k2.SymbolTable.from_str("\n".join([f"{k} {v}" for k, v in token2id.items()]))
    fsa.aux_labels_sym = fsa.labels_sym

    fsa.draw('fsa_symbols.svg', title='An FSA with symbol table')


def test2():
    # get pronunciation graph

    has_boundary = False
    lexicon, token2id = read_lexicon(
        # "/fsx/users/huangruizhe/audio_ruizhe/librispeech_conformer_ctc/librispeech_english_us_mfa.dict",
        "/fsx/users/huangruizhe/audio_ruizhe/librispeech_conformer_ctc/librispeech_english_us_mfa.prob.dict",
        has_boundary=has_boundary,
    )

    try:
        lexicon_new_words, _ = read_lexicon(
            "/fsx/users/huangruizhe/audio_ruizhe/librispeech_conformer_ctc/librispeech_english_us_mfa.new_words.dict",
            has_boundary=has_boundary,
            quiet=True,
        )
        # lexicon.update(lexicon_new_words)
        for w in lexicon_new_words.keys():
            if w not in lexicon:
                lexicon[w] = lexicon_new_words[w]
    except:
        pass

    try:
        lexicon_new_words, _ = read_lexicon(
            "/fsx/users/huangruizhe/datasets/Buckeye_Corpus2/buckeye_words.dict",
            has_boundary=has_boundary,
            quiet=True,
        )
        # lexicon.update(lexicon_new_words)
        for w in lexicon_new_words.keys():
            if w not in lexicon:
                lexicon[w] = lexicon_new_words[w]
    except:
        pass

    print(token2id)

    _lexicon = dict()
    for w in ["pen", "pineapple", "apple", "pen"]:
        trie = Trie()
        for prob, tokens in lexicon[w]:
            trie.insert(tokens, weight=prob)
        
        res, next_index, last_index = trie.to_k2_str_pron(token2id=token2id, index_offset=0)
        _lexicon[w] = (res, next_index)

    fsa_str = ""
    next_index = 0

    res, _next_index = _lexicon["pen"]
    fsa_str += "\n"
    fsa_str += fstr("\n".join(res), x = next_index)
    next_index += _next_index
    
    res, _next_index = _lexicon["pineapple"]
    fsa_str += "\n"
    fsa_str += fstr("\n".join(res), x = next_index)
    next_index += _next_index

    res, _next_index = _lexicon["apple"]
    fsa_str += "\n"
    fsa_str += fstr("\n".join(res), x = next_index)
    next_index += _next_index

    res, _next_index = _lexicon["pen"]
    fsa_str += "\n"
    fsa_str += fstr("\n".join(res), x = next_index)
    next_index += _next_index

    fsa_str += f"\n{next_index} {next_index + 1} -1 -1 0\n{next_index + 1}"
    # print(res)

    import k2
    fsa = k2.Fsa.from_str(fsa_str.strip(), acceptor=False)
    fsa.labels_sym = k2.SymbolTable.from_str("\n".join([f"{k} {v}" for k, v in token2id.items()]))
    fsa.aux_labels_sym = fsa.labels_sym

    fsa.draw('fsa_symbols.svg', title='An FSA with symbol table')


def test3():
    has_boundary = False
    lexicon, token2id = read_lexicon(
        # "/fsx/users/huangruizhe/audio_ruizhe/librispeech_conformer_ctc/librispeech_english_us_mfa.dict",
        "/fsx/users/huangruizhe/audio_ruizhe/librispeech_conformer_ctc/librispeech_english_us_mfa.prob.dict",
        has_boundary=has_boundary,
    )

    try:
        lexicon_new_words, _ = read_lexicon(
            "/fsx/users/huangruizhe/audio_ruizhe/librispeech_conformer_ctc/librispeech_english_us_mfa.new_words.dict",
            has_boundary=has_boundary,
            quiet=True,
        )
        # lexicon.update(lexicon_new_words)
        for w in lexicon_new_words.keys():
            if w not in lexicon:
                lexicon[w] = lexicon_new_words[w]
    except:
        pass

    try:
        lexicon_new_words, _ = read_lexicon(
            "/fsx/users/huangruizhe/datasets/Buckeye_Corpus2/buckeye_words.dict",
            has_boundary=has_boundary,
            quiet=True,
        )
        # lexicon.update(lexicon_new_words)
        for w in lexicon_new_words.keys():
            if w not in lexicon:
                lexicon[w] = lexicon_new_words[w]
    except:
        pass
    token2id["-"] = len(token2id)

    aux_offset = 1000000
    for k, v in list(token2id.items()):
        token2id[f"▁{k}"] = v + aux_offset

    sil_penalty_intra_word = 0.1
    sil_penalty_inter_word = 0.5
    topo_type = "ctc"

    # text = "pen pineapple apple pen"
    # text = "THAT THE HEBREWS WERE RESTIVE UNDER THIS TYRANNY WAS NATURAL INEVITABLE"
    text = "boy they'd pay for my high school my college and everything yknow even living expenses yknow but because i have a bachelors"
    
    _lexicon = dict()
    for w in text.strip().lower().split():
        trie = Trie()
        for prob, tokens in lexicon[w]:
            trie.insert(tokens, weight=prob)
        
        res, next_index, last_index = trie.to_k2_str_topo(token2id=token2id, index_offset=0, topo_type=topo_type, sil_penalty_intra_word=sil_penalty_intra_word, sil_penalty_inter_word=sil_penalty_inter_word, blank_id=token2id["-"], aux_offset=aux_offset)
        _lexicon[w] = (res, next_index)

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