from typing import Union, List
import logging
from tqdm import tqdm
import sentencepiece as spm
from lexicon import Lexicon


class Tokenizer:
    def __init__(
            self,
            has_boundary=True,
            modeling_unit="phoneme",  # char, bpe, phoneme
            lexicon=None,
            token2id=None,
            blank_token=None,
            unk_token=None,
            sp_model_path=None,
        ):

        if modeling_unit == "phoneme":
            assert lexicon is not None
        assert token2id is not None
        if modeling_unit == "bpe":
            assert sp_model_path is not None

        self.modeling_unit = modeling_unit
        self.has_boundary = has_boundary
        self.lexicon = lexicon
        self.token2id = token2id.copy()
 
        # Unlike the convention in torchaudio, the blank id is always 0 here
        if blank_token is None:
            blank_token = "-"
            if modeling_unit == "char" or modeling_unit == "bpe":
                blank_token = "--"
        self.blank_token = blank_token
        self.token2id[blank_token] = 0
        self.blank_id = 0
 
        self.unk_token = unk_token
        self.unk_id = self.token2id[unk_token]

        if has_boundary:
            next_id = max(self.token2id.values()) + 1
            for t in token2id.keys():
                if t != blank_token and t != unk_token:
                    self.token2id[f"▁{t}"] = next_id
                    next_id += 1

        self.id2token = {v: k for k, v in self.token2id.items()}

        self.sp_model = None
        if modeling_unit == "bpe":
            # sp_model_path = "/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_rnnt/spm_unigram_1023.model"
            # sp_model_path = "/exp/rhuang/meta/audio/examples/asr/librispeech_conformer_ctc/spm_unigram_1023.model"
            self.sp_model = spm.SentencePieceProcessor(model_file=str(sp_model_path))
    
    def fstr(self, template, x):
        return fstr(template, x)

    def get_piece_size(self):
        return len(self.token2id)
    
    def tokens(self):
        return list(self.token2id.keys())
    
    def id_to_piece(self, id):
        return self.id2token[id]

    def vocab_size(self):
        return self.get_piece_size()
    
    # def add_new_word(self, word):
    #     pass

    def encode_char(self, sentence, out_type: type = int) -> List:
        sentence = sentence.strip().lower()
        sentence = sentence.split()

        token_ids = []
        for word in sentence:
            ids = [
                self.token2id.get(c, self.unk_id)
                for c in word
            ]
            if self.has_boundary:
                c = f"▁{word[0]}"
                ids[0] = self.token2id.get(c, self.unk_id)
            token_ids.extend(ids)
        
        if out_type == int:
            pass
        elif out_type == str:
            token_ids = [self.id2token[t] for t in token_ids]
        else:
            raise NotImplementedError
        
        return token_ids
    
    def encode_bpe(self, sentence, out_type: type = int) -> List:
        sentence = sentence.strip().lower()

        _token_ids = self.sp_model.encode(sentence, out_type=int)
        tokens = [self.sp_model.id_to_piece(t) for t in _token_ids]
        token_ids = [self.token2id.get(t, self.unk_id) for t in tokens]

        # tokens = self.sp_model.encode(sentence, out_type=str)
        # token_ids = [self.token2id.get(t, self.unk_id) for t in tokens]

        if out_type == int:
            return token_ids
        elif out_type == str:
            return tokens
        else:
            raise NotImplementedError

    def encode_phoneme(self, sentence, out_type: type = int) -> List:
        sentence = sentence.strip().lower()
        sentence = sentence.split()

        token_ids = []
        for word in sentence:
            pron = self.lexicon.get_pron(word, limit=1)

            if pron is None:
                token_ids.append(self.unk_id)
                continue

            pron = pron[0][1]
            if self.has_boundary:
                pron[0] = f"▁{pron[0]}"
            pron_ids = [self.token2id.get(p, self.unk_id) for p in pron]
            token_ids.extend(pron_ids)
        
        if out_type == int:
            pass
        elif out_type == str:
            token_ids = [self.id2token[t] for t in token_ids]
        else:
            raise NotImplementedError
        
        return token_ids

    def encode(self, sentence, out_type: type = int) -> List:
        if self.modeling_unit == "char":
            return self.encode_char(sentence, out_type=out_type)
        elif self.modeling_unit == "bpe":
            return self.encode_bpe(sentence, out_type=out_type)
        elif self.modeling_unit == "phoneme":
            return self.encode_phoneme(sentence, out_type=out_type)
        else:
            raise NotImplementedError

def test1():
    # https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html
    char2id = {'-': 0, '@': 1, 'e': 2, 't': 3, 'a': 4, 'o': 5, 'n': 6, 'i': 7, 'h': 8, 's': 9, 'r': 10, 'd': 11, 'l': 12, 'u': 13, 'm': 14, 'w': 15, 'c': 16, 'f': 17, 'g': 18, 'y': 19, 'p': 20, 'b': 21, 'v': 22, 'k': 23, "'": 24, 'x': 25, 'j': 26, 'q': 27, 'z': 28}
    tokenizer = Tokenizer(
        has_boundary=False,
        modeling_unit="char",
        token2id=char2id,
        blank_token='-',
        unk_token='@',
        sp_model_path=None
    )
 
    while True:
        line = input("Type a sentence\n")
        out = tokenizer.encode(line, out_type=str)
        print(f'Input: {line}')
        print(f'Output: {out}')


def test2():
    sp_model_path = "/exp/rhuang/meta/audio/examples/asr/librispeech_conformer_ctc/spm_unigram_1023.model"
    sp_model = spm.SentencePieceProcessor(model_file=str(sp_model_path))
    token2id = {sp_model.id_to_piece(i): i + 1 for i in range(sp_model.vocab_size())}
    assert "-" not in token2id
    token2id["-"] = 0
    tokenizer = Tokenizer(
        has_boundary=False,
        modeling_unit="bpe",
        token2id=token2id,
        blank_token='-',
        unk_token='<unk>',
        sp_model_path=sp_model_path,
    )
 
    while True:
        line = input("Type a sentence\n")
        out = tokenizer.encode(line, out_type=str)
        print(f'Input: {line}')
        print(f'Output: {out}')


def test3():
    phone_set = ['ə', 'ɛ', 'd', 'ɪ', 'ɾ', 't', 'm', 'n', 'ɫ', 'i', 'ɫ̩', 'a', 'ɚ', 'ʔ', 'ɹ', 's', 'z', 'ɔ', 'ɐ', 'v', 'spn', 'ej', 'e', 'ɑ', 'ɑː', 'ɒ', 'dʲ', 'iː', 'dʒ', 'vʲ', 'ɒː', 'bʲ', 'tʃ', 'æ', 'b', 'ow', 'aj', 'cʰ', 'p', 'kʰ', 'pʰ', 'k', 'j', 'ʊ', 'ɡ', 'ʎ', 'l', 'w', 'f', 'h', 'ʉː', 'ʉ', 'uː', 'u', 'ɛː', 'ɲ', 'pʲ', 'o', 'əw', 'θ', 'tʲ', 'ʃ', 'c', 'tʰ', 'n̩', 'ŋ', 'ʒ', 'tʷ', 'mʲ', 'ç', 'ɝ', 'ɔj', 'aw', 'ɟ', 'fʲ', 'aː', 'ɜː', 'vʷ', 'kʷ', 'ɜ', 'cʷ', 'ɾʲ', 'ɡb', 'ð', 'ɾ̃', 'kp', 'ɡʷ', 'ɟʷ', 'd̪', 't̪', 'pʷ', 'm̩', 'fʷ']
    token2id = {p: i + 1 for i, p in enumerate(phone_set)}
    token2id["-"] = 0

    my_lexicon = Lexicon(
        files=[
            "/exp/rhuang/meta/audio_ruizhe/librispeech_conformer_ctc/librispeech_english_us_mfa.prob.dict",
            "/exp/rhuang/meta/audio_ruizhe/librispeech_conformer_ctc/librispeech_english_us_mfa.new_words.dict",
            "/exp/rhuang/buckeye/datasets/Buckeye_Corpus2/buckeye_words.dict",
        ]
    )

    tokenizer = Tokenizer(
        has_boundary=False,
        modeling_unit="phoneme",
        lexicon=my_lexicon,
        token2id=token2id,
        blank_token='-',
        unk_token='spn',
        sp_model_path=None
    )
 
    while True:
        line = input("Type a sentence\n")
        out = tokenizer.encode(line, out_type=str)
        print(f'Input: {line}')
        print(f'Output: {out}')


if __name__ == "__main__":
    logging.basicConfig(
        format = "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        level = 10
    )

    # test1()
    # test2()
    test3()

