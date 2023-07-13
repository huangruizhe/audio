from typing import Union, List

class CharTokenizer:
    def __init__(self):
        # https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html
        self.char2id = {'-': 0, '|': 1, 'E': 2, 'T': 3, 'A': 4, 'O': 5, 'N': 6, 'I': 7, 'H': 8, 'S': 9, 'R': 10, 'D': 11, 'L': 12, 'U': 13, 'M': 14, 'W': 15, 'C': 16, 'F': 17, 'G': 18, 'Y': 19, 'P': 20, 'B': 21, 'V': 22, 'K': 23, "'": 24, 'X': 25, 'J': 26, 'Q': 27, 'Z': 28}
 
        # Follow the convention in torchaudio: making blank id the largest id
        self.char2id = {k: v - 1 for k, v in self.char2id.items()}
        self.char2id["-"] = len(self.char2id) - 1
        assert self.char2id["-"] == 28
 
        self.id2char = {v: k for k, v in self.char2id.items()}

        self.blank_id = self.char2id["-"]
        self.unk_id = self.char2id["|"]

    def get_piece_size(self):
        return len(self.char2id)
    
    def tokens(self):
        return list(self.char2id.keys())
    
    def id_to_piece(self, id):
        return self.id2char[id]

    def vocab_size(self):
        return self.get_piece_size()
    
    def encode(self, sentence, out_type: type = int, insert_blank = False) -> List:
        sentence = sentence.strip().upper()
        sentence = sentence.split()

        token_ids = []
        for word in sentence:
            ids = [
                self.char2id[c] if c in self.char2id
                else self.unk_id
                for c in word
            ]
            if insert_blank:
                ids.append(self.blank_id)
            token_ids.extend(ids)
        
        if out_type == int:
            pass
        elif out_type == str:
            token_ids = [self.id2char[t] for t in token_ids]
        else:
            raise NotImplementedError

        return token_ids

    def encode_keep_boundary(self, sentence, out_type: type = int) -> List:
        sentence = sentence.strip().upper()
        sentence = sentence.split()

        token_ids = []
        for word in sentence:
            ids = [
                self.char2id[c] if c in self.char2id
                else self.unk_id
                for c in word
            ]
            token_ids.append(ids)
        
        return token_ids

