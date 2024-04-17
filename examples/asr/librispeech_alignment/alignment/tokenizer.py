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
    
    Note:
    1. It should include two special tokens: <blk> and <unk>, where the token id for <blk> should be 0
    '''

    @abstractmethod
    def encode(self, sentence, out_type=int):
        raise NotImplementedError

    def encode_flatten(self, sentence, out_type=int):
        tokens = self.encode(sentence, out_type=out_type)

        # `tokens` can be either
        # - a list of lists of integers
        # - a list of lists of lists of integers
        # Here, we will remove the word boundaries.
        # If a word has multiple pronuncations, we will keep only the first one
        if isinstance(tokens[0][0], list):
            tokens = [t for w_prons in tokens for t in w_prons[0]]
        else:
            assert isinstance(tokens[0][0], int), tokens
            tokens = [t for w_prons in tokens for t in w_prons]

        return tokens


class EnglishTokenizer(TokenizerInterface):
    # Keep "'" as in Librispeech 
    # How about the dash symbol "-"? 
    # Ok, I don't find '-' symbol in Librispeech file. So we will not allow to use "-"
    punctuation = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'

    def _normalize(self, word):
        # Normalize a word
        # TODO: checkout how Gentle do this

        # Remove all punctuation
        word = word.translate(str.maketrans("", "", self.punctuation))
        # Convert all upper case to lower case
        word = word.lower()
        if len(word) == 0:
            return "*"
        return word
    
    def text_normalize(self, text: str) -> str:
        # We preserve the word index in the transcript (i.e, we don't remove words)
        # E.g., "hello 这 world" -> "hello * world"

        # [Ref] https://github.com/kaldi-asr/kaldi/blob/master/egs/librispeech/s5/local/lm/normalize_text.sh
        text = [self._normalize(w) for w in text.split()]
        text = " ".join(text)
        return text

    def encode(self, sentence, out_type=int):
        raise NotImplementedError


class EnglishCharTokenizer(EnglishTokenizer):
    def __init__(
        self,
        token2id,
        blk_token,
        unk_token,
    ):
        super().__init__()
        self.token2id = token2id.copy()
        self.unk_id = token2id[unk_token]  # <unk> token should already be in `token2id`
        self.blk_id = token2id[blk_token]
        assert self.blk_id == 0
        self.id2token = {v: k for k, v in self.token2id.items()}
    
    def encode(self, sentence, out_type=int):
        # Assume that `sentence` is already in the correct casing
        token_ids = [[self.token2id.get(c, self.unk_id) for c in w] for w in sentence.split()]
        return token_ids

    def decode(self, token_ids):
        # For verification purposes
        return [[self.id2token[i] for i in w] for w in token_ids]


class EnglishBPETokenizer(EnglishTokenizer):
    def __init__(
        self,
        sp_model,
        blk_token,
        unk_token,
    ):
        super().__init__()
        self.unk_id = sp_model.piece_to_id(unk_token)
        self.blk_id = sp_model.piece_to_id(blk_token)
        assert self.blk_id == 0
        self.sp_model = sp_model
        self.token2id = {sp_model.id_to_piece(i): i for i in range(sp_model.vocab_size())}
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

    def encode(self, sentence, out_type=int):
        sentence = sentence.strip()
        token_ids = self.sp_model.encode(sentence, out_type=out_type)
        token_ids = self.get_word_boundaries(token_ids)
        return token_ids

    def decode(self, token_ids):
        # For verification purposes
        return [[self.sp_model.id_to_piece(t) for t in w] for w in token_ids]


class EnglishPhonemeTokenizer(EnglishTokenizer):
    '''
    Useful resources for G2P in python:
    # https://github.com/prosegrinder/python-cmudict
    # https://github.com/Kyubyong/g2p
    # https://github.com/lingjzhu/CharsiuG2P
    '''

    def __init__(
        self,
        blk_token="<blk>",
        unk_token="<unk>",
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

        self.token2id = {p: i + 1 for i, (p, _) in enumerate(cmudict.phones())}  # Note: we don't model stress level here
        self.token2id[blk_token] = 0
        self.blk_id = 0
        self.unk_token = unk_token
        self.unk_id = len(self.token2id)
        self.token2id[unk_token] = self.unk_id

        self.id2token = {v: k for k, v in self.token2id.items()}
    
    def get_word_pron(self, word, num_prons=None):
        if word in self.cmu:
            prons = self.cmu[word]
            prons = prons[:num_prons]
        else:
            pron = self.g2p(word.replace("'", ""))
            if len(pron) == 0:  # e.g., word = "你好"
                pron = [self.unk_token]
            prons = [pron]
        
        prons = [tuple(re.sub(r'\d', '', p) for p in pron) for pron in prons]  # remove stress level
        prons = list(set(prons))  # remove duplicates
        return prons  # This is a list of pronuncations, where each pronuncation is a list of phonemes

    def encode(self, sentence, num_prons=None, out_type=int):
        sentence = sentence.strip().lower()

        # This will be a list of lists
        tokens = [
            self.get_word_pron(w, num_prons) for w in sentence.split()
        ]
        token_ids = [
            [[self.token2id[p] for p in pron] for pron in w_prons] for w_prons in tokens
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
        blk_token="-",
        unk_token="|",
    )
    print(tokenizer.encode(transcript))
    print(tokenizer.decode(tokenizer.encode(transcript)))


def test2():
    # https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html
    labels = ('-', 'a', 'i', 'e', 'n', 'o', 'u', 't', 's', 'r', 'm', 'k', 'l', 'd', 'g', 'h', 'y', 'b', 'p', 'w', 'c', 'v', 'j', 'z', 'f', "'", 'q', 'x', '*')
    transcript = "i had that curiosity beside me at this moment"

    tokenizer = EnglishCharTokenizer(
        token2id={c: i for i, c in enumerate(labels)},
        blk_token="-",
        unk_token="*",
    )
    print(tokenizer.encode(transcript))
    print(tokenizer.decode(tokenizer.encode(transcript)))


def test3():
    sp_model_path = "/exp/rhuang/meta/audio/examples/asr/librispeech_conformer_ctc/spm_unigram_1023.model"
    sp_model = spm.SentencePieceProcessor(model_file=str(sp_model_path))
    tokenizer = EnglishBPETokenizer(
        sp_model=sp_model,
        blk_token="<s>",  # You have to make sure these tokens are in the sp model 
        unk_token="<unk>",
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
    # transcript = "the meetin' hou-se"
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

