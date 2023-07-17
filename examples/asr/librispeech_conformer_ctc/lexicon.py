import logging
import re
import sys
from pathlib import Path
from fastnumbers import check_float
from collections import defaultdict
import math

def read_lexicon(filename: str, has_boundary=False):
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
        whitespace = re.compile("[ \t]+")
        for line in fin:
            a = whitespace.split(line.strip(" \t\r\n"))
            if len(a) == 0:
                continue

            if len(a) < 2:
                logging.info(f"Found bad line {line} in lexicon file {filename}")
                logging.info("Every line is expected to contain at least 2 fields")
                sys.exit(1)
            word = a[0]
            if word == "<eps>":
                logging.info(f"Found bad line {line} in lexicon file {filename}")
                logging.info("<eps> should not be a valid word")
                sys.exit(1)

            tokens = a[1:]

            prob = 1.0  # probability
            if check_float(tokens[0]):
                prob = float(tokens[0])
                tokens = tokens[1:]
            
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

    total_entries = sum([len(v) for k, v in ans.items()])
    logging.info(f"Number of words in dict: {len(ans)}")
    logging.info(f"Number of tokens in dict: {len(token2id)}")
    logging.info(f"Average number of pronunciations per word: {total_entries/len(ans):.2f}")
    # print(f"Number of words in dict: {len(ans)}")
    # print(f"Number of tokens in dict: {len(token2id)}")
    # print(f"Average number of pronunciations per word: {total_entries/len(ans):.2f}")
    # print(f"Tokens: {list(token2id.keys())}")

    return ans, token2id


class TrieNode:
    def __init__(self, token):
        self.token = token
        self.is_end = False
        self.state_id = None
        self.weight = -1e9
        self.children = {}


class Trie(object):
    # https://albertauyeung.github.io/2020/06/15/python-trie.html/

    def __init__(self):
        self.root = TrieNode("")
    
    def insert(self, word_tokens, weight=1.0):
        """Insert a word into the trie"""
        node = self.root
        
        # Loop through each token in the word
        # Check if there is no child containing the token, create a new child for the current node
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
    
    def to_k2_str_topo(self, node=None, start_index=0, last_index=-1, token2id=None, index_offset=1, topo_type="ctc", sil_log_scale=1.0, blank_id = 0):
        if node is None:
            node = self.root

            cnt = 1  # count the number of non-leaf nodes
            # cnt_children = 0  # count the number of children
            temp_list = list(self.root.children.values())
            while len(temp_list) > 0:
                n = temp_list.pop()
                # cnt_children += 1
                if len(n.children) > 0:
                    cnt += 1
                    temp_list.extend(n.children.values())
            
            if topo_type == "hmm":
                last_index = start_index + cnt + 1
            else:
                last_index = start_index + cnt + cnt

        res = []

        if topo_type == "hmm":
            assert sil_log_scale == 1.0
        
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
                    _res, _next_index, _last_index = self.to_k2_str_topo(node=c, start_index=next_index, last_index=last_index, token2id=token2id, index_offset=index_offset, topo_type=topo_type, sil_log_scale=sil_log_scale, blank_id=blank_id)
                    next_index = _next_index
                    res.extend(_res)
                else:
                    res.append((start_index, f"{{x + {start_index}}} {{x + {last_index}}} {token} {token} {weight}"))
            else:
                # There is intra-word blank/silence with some probability
                if len(c.children) > 0:
                    if i == 0:
                        res.append((start_index, f"{{x + {start_index}}} {{x + {blank_state_index}}} {blank_id} {blank_id} {sil_log_scale * weight}"))
                        res.append((blank_state_index, f"{{x + {blank_state_index}}} {{x + {blank_state_index}}} {blank_id} {blank_id} 0"))
                    res.append((blank_state_index, f"{{x + {blank_state_index}}} {{x + {next_index}}} {token} {token} 0"))
                    res.append((start_index, f"{{x + {start_index}}} {{x + {next_index}}} {token} {token} {weight}"))
                    res.append((next_index, f"{{x + {next_index}}} {{x + {next_index}}} {token} {token} 0"))
                    _res, _next_index, _last_index = self.to_k2_str_topo(node=c, start_index=next_index, last_index=last_index, token2id=token2id, index_offset=index_offset, topo_type=topo_type, sil_log_scale=sil_log_scale, blank_id=blank_id)
                    next_index = _next_index
                    res.extend(_res)
                else:
                    if i == 0:
                        res.append((start_index, f"{{x + {start_index}}} {{x + {blank_state_index}}} {blank_id} {blank_id} {sil_log_scale * weight}"))
                        res.append((blank_state_index, f"{{x + {blank_state_index}}} {{x + {blank_state_index}}} {blank_id} {blank_id} {0}"))
                    res.append((blank_state_index, f"{{x + {blank_state_index}}} {{x + {last_index}}} {token} {token} {0}"))
                    res.append((start_index, f"{{x + {start_index}}} {{x + {last_index}}} {token} {token} {weight}"))
                    res.append((last_index, f"{{x + {last_index}}} {{x + {last_index}}} {token} {token} 0"))
            
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
    lexicon, token2id = read_lexicon("/fsx/users/huangruizhe/mfa-models/dictionary/english/mfa/english_mfa.dict")
    
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
    lexicon, token2id = read_lexicon("/fsx/users/huangruizhe/mfa-models/dictionary/english/mfa/english_mfa.dict")
    token2id["-"] = len(token2id)
    
    _lexicon = dict()
    for w in ["pen", "pineapple", "apple", "pen"]:
        trie = Trie()
        for prob, tokens in lexicon[w]:
            trie.insert(tokens, weight=prob)
        
        res, next_index, last_index = trie.to_k2_str_topo(token2id=token2id, index_offset=0, topo_type="ctc", sil_log_scale=1.0, blank_id=token2id["-"])
        # res, next_index, last_index = trie.to_k2_str_topo(token2id=token2id, index_offset=0, topo_type="hmm", blank_id=token2id["-"])
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

    blank_id = token2id["-"]
    fsa_str += f"\n{next_index} {next_index + 1} {blank_id} {blank_id} 0"
    fsa_str += f"\n{next_index} {next_index + 2} -1 -1 0"
    fsa_str += f"\n{next_index + 1} {next_index + 1} {blank_id} {blank_id} 0"
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