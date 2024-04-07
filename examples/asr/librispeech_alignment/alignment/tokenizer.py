from abc import ABC, abstractmethod
import sentencepiece as spm
import re


class TokenizerInterface(ABC):
    '''
    The tokenizer do the following:
    INPUT: a sentence
    OUTPUT: a list of lists, where each sublist is a list of alternative token ids for each word

    E.g.,
    INPUT: "this is a sentence"
    OUTPUT: 
           [[75], [47], [7], [629, 218]]
        or [[[75]], [[47]], [[7]], [[629, 218], [123, 456]]] in case a word has multiple pronunciations
    '''

    @abstractmethod
    def encode(self, sentence, out_type: type = int):
        raise NotImplementedError


class EnglishCharTokenizer(TokenizerInterface):
    def __init__(
        self,
        token2id,
        unk_id=0,
    ):
        super().__init__()
        self.token2id = token2id.copy()
        self.unk_id = unk_id  # <unk> token should already be in `token2id`
        self.id2token = {v: k for k, v in self.token2id.items()}
    
    def encode(self, sentence, out_type: type = int):
        # Assume that `sentence` is already in the correct casing
        token_ids = [[self.token2id.get(c, self.unk_id) for c in w] for w in sentence.split()]
        return token_ids

    def decode(self, token_ids):
        # For verification purposes
        return [[self.id2token[i] for i in w] for w in token_ids]


class EnglishBPETokenizer(TokenizerInterface):
    def __init__(
        self,
        sp_model,
        unk_id=0,
    ):
        super().__init__()
        self.unk_id = unk_id
        self.sp_model = sp_model
        self.start_token_ids = {i for i in range(sp_model.vocab_size()) if sp_model.id_to_piece(i).startswith("▁")}

    def get_word_boundaries(self, token_ids):
        rs = []
        word_start = 0
        for i in range(len(token_ids)):
            if token_ids[i] in self.start_token_ids:
                rs.append(token_ids[word_start: i])
                word_start = i
        rs.append(token_ids[word_start:])
        return rs[1:]

    def encode(self, sentence, out_type: type = int):
        sentence = sentence.strip()
        token_ids = self.sp_model.encode(sentence, out_type=out_type)
        token_ids = self.get_word_boundaries(token_ids)
        return token_ids

    def decode(self, token_ids):
        # For verification purposes
        return [[self.sp_model.id_to_piece(t) for t in w] for w in token_ids]


class EnglishPhonemeTokenizer(TokenizerInterface):
    '''
    Useful resources for G2P in python:
    # https://github.com/prosegrinder/python-cmudict
    # https://github.com/Kyubyong/g2p
    # https://github.com/lingjzhu/CharsiuG2P
    '''

    def __init__(
        self,
    ):
        # Importing them can be slow. So we only do it when needed
        try:
            import cmudict
            from g2p_en import G2p
        except ImportError:
            raise ImportError("Please install cmudict and g2p_en by, e.g., pip install cmudict g2p_en")

        super().__init__()
        self.cmu = cmudict.dict()
        self.g2p = G2p()

        self.token2id = {p: i for i, (p, _) in enumerate(cmudict.phones())}  # Note: we don't model stress level here
        self.unk = "<unk>"
        self.unk_id = len(self.token2id)
        self.token2id[self.unk] = self.unk_id

        self.id2token = {v: k for k, v in self.token2id.items()}
    
    def get_word_pron(self, word):
        if word in self.cmu:
            prons = self.cmu[word]
        else:
            pron = self.g2p(word)
            if len(pron) == 0:  # e.g., word = "你好"
                pron = [self.unk]
            prons = [pron]
        
        prons = [tuple(re.sub(r'\d', '', p) for p in pron) for pron in prons]  # remove stress level
        prons = list(set(prons))  # remove duplicates
        return prons

    def encode(self, sentence, out_type: type = int):
        sentence = sentence.strip().lower()

        # This will be a list of lists
        tokens = [
            self.get_word_pron(w) for w in sentence.split()
        ]
        token_ids = [
            [[self.token2id[p] for p in pron] for pron in prons] for prons in tokens
        ]

        # We just simply consider each pronunciation equally likely here
        # Not sure if the pronunciation probabilities can make any improvement
        if out_type == int:
            return token_ids
        elif out_type == str:
            return tokens
        else:
            raise NotImplementedError
    
    def decode(self, token_ids):
        # For verification purposes
        return [
            [[self.id2token[p] for p in pron] for pron in prons] for prons in token_ids
        ]


def test1():
    # https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html
    labels = ('-', '|', 'E', 'T', 'A', 'O', 'N', 'I', 'H', 'S', 'R', 'D', 'L', 'U', 'M', 'W', 'C', 'F', 'G', 'Y', 'P', 'B', 'V', 'K', "'", 'X', 'J', 'Q', 'Z')
    transcript = "|I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT|"
    transcript = transcript.replace("|", " | ").strip()

    tokenizer = EnglishCharTokenizer(
        token2id={c: i for i, c in enumerate(labels)},
        unk_id=1,
    )
    print(tokenizer.encode(transcript))
    print(tokenizer.decode(tokenizer.encode(transcript)))


def test2():
    # https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html
    labels = ('-', 'a', 'i', 'e', 'n', 'o', 'u', 't', 's', 'r', 'm', 'k', 'l', 'd', 'g', 'h', 'y', 'b', 'p', 'w', 'c', 'v', 'j', 'z', 'f', "'", 'q', 'x')
    transcript = "i had that curiosity beside me at this moment"

    tokenizer = EnglishCharTokenizer(
        token2id={c: i for i, c in enumerate(labels)},
        unk_id=1,
    )
    print(tokenizer.encode(transcript))
    print(tokenizer.decode(tokenizer.encode(transcript)))


def test3():
    sp_model_path = "/exp/rhuang/meta/audio/examples/asr/librispeech_conformer_ctc/spm_unigram_1023.model"
    sp_model = spm.SentencePieceProcessor(model_file=str(sp_model_path))
    tokenizer = EnglishBPETokenizer(
        unk_id=sp_model.piece_to_id("<unk>"),
        sp_model=sp_model,
    )
    # transcript = "i had that curiosity beside me at this moment"
    # transcript = "i had that curiosity beside me at this moment 你好"
    transcript = "i had that curiosity beside me at this mo1ment 你好"
    print(tokenizer.encode(transcript))
    print(tokenizer.decode(tokenizer.encode(transcript)))


def test4():
    tokenizer = EnglishPhonemeTokenizer()
 
    # transcript = "i had that curiosity beside me at this moment"
    # transcript = "i had that curiosity beside me at this moment 你好"
    transcript = "i had that curiosity beside me at this mo1ment 你好"
    print(tokenizer.encode(transcript))
    print(tokenizer.decode(tokenizer.encode(transcript)))


if __name__ == "__main__":

    # Char model:
    test1()
    # test2()

    # BPE model:
    # test3()

    # Phone model:
    # test4()

