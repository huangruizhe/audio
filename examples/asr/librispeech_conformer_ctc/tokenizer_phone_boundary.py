from typing import Union, List
from lexicon import read_lexicon, Trie, fstr
import logging
from tqdm import tqdm

class PhonemeTokenizerBoundary:
    def __init__(self, has_boundary=True, modeling_unit="phoneme"):
        if modeling_unit == "phoneme" or modeling_unit == "char" or modeling_unit == "bpe":
            lexicon, token2id = read_lexicon(
                # "/fsx/users/huangruizhe/audio_ruizhe/librispeech_conformer_ctc/librispeech_english_us_mfa.dict",
                "/fsx/users/huangruizhe/audio_ruizhe/librispeech_conformer_ctc/librispeech_english_us_mfa.prob.dict",
                has_boundary=has_boundary,
                modeling_unit=modeling_unit,
            )

            try:
                lexicon_new_words, _ = read_lexicon(
                    "/fsx/users/huangruizhe/audio_ruizhe/librispeech_conformer_ctc/librispeech_english_us_mfa.new_words.dict",
                    has_boundary=has_boundary,
                    quiet=True,
                    modeling_unit=modeling_unit,
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
                    modeling_unit=modeling_unit,
                )
                # lexicon.update(lexicon_new_words)
                for w in lexicon_new_words.keys():
                    if w not in lexicon:
                        lexicon[w] = lexicon_new_words[w]
            except:
                pass
        else:
            raise NotImplementedError

        self.lexicon = lexicon
        self.token2id = token2id
 
        # Follow the convention in torchaudio: making blank id the largest id
        self.token2id["-"] = len(self.token2id)
 
        self.id2token = {v: k for k, v in self.token2id.items()}

        self.blank_id = self.token2id["-"]

        if modeling_unit == "phoneme":
            if has_boundary:
                self.unk_id = self.token2id["▁spn"]
            else:
                self.unk_id = self.token2id["spn"]
        else:
            self.unk_id = None
    
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
    
    def add_new_word(self, word):
        pass
    
    def encode(self, sentence, out_type: type = int, insert_blank = False) -> List:
        sentence = sentence.strip().lower()
        sentence = sentence.split()

        token_ids = []
        if out_type == int:
            token_ids = [1]
        elif out_type == str:
            token_ids = [self.id2token[t] for t in [1]]
        else:
            raise NotImplementedError

        return token_ids


if __name__ == "__main__":
    logging.basicConfig(
        format = "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        level = 10
    )

    tokenizer = PhonemeTokenizerBoundary()
    print(tokenizer.encode("THE SCHOOL OF THE WILDERNESS"))
    print(tokenizer.encode("THE SCHOOL OF THE WILDERNESS", out_type=str))

