from dataclasses import dataclass
import itertools
import torch
from typing import List


@dataclass
class Frame:
    # This is the index of each token in the transcript,
    # i.e. the current frame aligns to the N-th character from the transcript.
    token_index: int
    time_index: int
    score: float


def get_frames(frame_alignment, frame_alignment_aux, frame_scores, blank_id=0, aux_offset=0):
    assert len(frame_alignment) == len(frame_scores)

    if isinstance(frame_alignment, torch.Tensor):
        frame_alignment = frame_alignment.tolist()
    if isinstance(frame_scores, torch.Tensor):
        frame_scores = frame_scores.tolist()

    strict_alignment = list(zip(frame_alignment, frame_alignment_aux))

    frames = []
    token_index = -1
    prev_token = None
    for i, (token_pair, score) in enumerate(zip(strict_alignment, frame_scores)):
        token, token_aux = token_pair

        if token == blank_id:
            prev_token = token
            continue

        # Two conditions to move on:
        # 1) new token
        # 2) not a new token, but token_aux tells us this is a word start
        if token != prev_token or token_aux >= aux_offset:
            token_index += 1

        frames.append(Frame(token_index, i, score))
        prev_token = token
    return frames

def hmm_k2_text_preprocess(text):
    return ''.join(c[0] for c in itertools.groupby(text))

def remove_consecutive_duplicates(lst, aux_offset=0):
    if isinstance(lst, torch.Tensor):
        lst = lst.tolist()

    last_seen = None
    res = []
    for x in lst:
        if x != last_seen or x >= aux_offset:
            res.append(x)
        last_seen = x
    return res

def ali_postprocessing_single(labels_ali, aux_labels_ali, sp_model, emission, aux_offset=100000, shift_label=False):
     # Hard-coded for torchaudio
    labels_ali = [x - 1 if x > 0 else sp_model.blank_id for x in labels_ali]
    aux_labels_ali = [x - 1 if x > 0 else sp_model.blank_id for x in aux_labels_ali]
    import pdb; pdb.set_trace()

    ############ get confidence scores and change frame rate ############
    # Ref: https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html
    assert emission.size(0) == len(labels_ali), f"{emission.size(0)} != {len(labels_ali)}"

    frame_alignment = torch.LongTensor(labels_ali)
    frame_alignment_aux = torch.LongTensor(aux_labels_ali)
    frame_scores = emission[torch.arange(len(frame_alignment)), frame_alignment]
    frame_scores = frame_scores.cpu()
    # frame_scores = frame_scores.exp()

    # frame_alignment = frame_alignment.repeat_interleave(subsampling_factor)
    # frame_scores = frame_scores.repeat_interleave(subsampling_factor)
    
    frames = get_frames(labels_ali, aux_labels_ali, frame_scores, blank_id=sp_model.blank_id, aux_offset=aux_offset)

    # self.scratch_space["decoding_graph"] = decoding_graph
    # self.scratch_space["emission"] = emission
    # self.scratch_space["frame_alignment_aux"] = torch.LongTensor(aux_labels_ali[0])

    # indices = torch.unique_consecutive(frame_alignment, dim=-1)
    # indices = [i for i in indices if i != sp_model.blank_id]
    # joined = "".join([sp_model.id_to_piece(i.item()) for i in indices])
    # # return joined.replace("▁", " ").strip().split()
    # tokens = [sp_model.id_to_piece(i.item()) for i in indices]
    # token_ids = [i.item() for i in indices]

    # indices_aux = torch.unique_consecutive(frame_alignment_aux, dim=-1).tolist()
    indices_aux = remove_consecutive_duplicates(frame_alignment_aux, aux_offset=aux_offset)
    token_ids_aux = []
    i_prev = -1
    for i in indices_aux:
        if i + aux_offset != i_prev:
            token_ids_aux.append(i if i < aux_offset else i - aux_offset)
        i_prev = i
    token_ids_aux = [i for i in token_ids_aux if i != sp_model.blank_id]
    token_ids = token_ids_aux
    tokens = [sp_model.id_to_piece(i) for i in token_ids]

    # if frames[-1].token_index >= len(tokens):  # or tuple(token_ids[:3]) == (34, 71, 83)
    #     import pdb; pdb.set_trace()
    assert frames[-1].token_index == len(tokens) - 1

    return tokens, token_ids, frame_alignment, frame_alignment_aux, frame_scores, frames


# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(frames, tokens, token_ids):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(frames):
        while i2 < len(frames) and frames[i1].token_index == frames[i2].token_index:
            i2 += 1
        score = sum(frames[k].score for k in range(i1, i2)) / (i2 - i1)

        # print(f"{i1}, {frames[i1]}")
        segments.append(
            Segment(
                tokens[frames[i1].token_index],
                frames[i1].time_index,
                frames[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


# Obtain word alignments from token alignments
def merge_words_bpe(tokens, segments):
    words = []

    word_start_idx = 0
    cur_idx = 1
    while cur_idx <= len(tokens):
        if cur_idx == len(tokens) or segments[cur_idx].label.startswith("▁"):
            segs = segments[word_start_idx: cur_idx]
            word = "".join([seg.label for seg in segs])
            score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
            words.append(Segment(word, segments[word_start_idx].start, segments[cur_idx - 1].end, score))
            word_start_idx = cur_idx
        cur_idx += 1
    return words


def merge_words_without_seperator(tokens, segments, text, separator=" "):
    words = []

    i1, i2, i3 = 0, 0, 0
    while i3 < len(tokens):
        if i3 - i1 > 0 and text[i2] == separator:
            # if i3 == len(tokens) - 1:
            #     i3 = len(tokens)
            segs = segments[i1: i3]
            word = "".join([seg.label for seg in segs])
            score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
            words.append(Segment(word, segments[i1].start, segments[i3 - 1].end, score))
            i1 = i3
            i3 -= 1
        i3 += 1
        i2 += 1
    
    # the last word
    if i3 - i1 > 0:
        segs = segments[i1: i3]
        word = "".join([seg.label for seg in segs])
        score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
        words.append(Segment(word, segments[i1].start, segments[i3 - 1].end, score))
        
    return words


# Obtain word alignments from token alignments
def merge_words_aux(tokens, segments, frame_alignment_aux, sp_model, aux_offset=100000):
    words = []

    indices_aux = remove_consecutive_duplicates(frame_alignment_aux, aux_offset=aux_offset)
    token_ids_aux = []
    i_prev = -1
    for i in indices_aux:
        if i + aux_offset != i_prev:
            token_ids_aux.append(i)
        i_prev = i
    token_ids_aux = [i for i in token_ids_aux if i != sp_model.blank_id]

    # if len(segments) != len(tokens):
    #     import pdb; pdb.set_trace()

    assert len(segments) == len(tokens), f"{len(segments)} vs. {len(tokens)}"
    assert len(segments) == len(token_ids_aux), f"{len(segments)} vs. {len(token_ids_aux)}"

    word_start_idx = 0
    cur_idx = 1
    while cur_idx <= len(tokens):
        if cur_idx == len(tokens) or token_ids_aux[cur_idx] >= aux_offset:
            segs = segments[word_start_idx: cur_idx]
            word = "".join([seg.label for seg in segs])
            score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
            words.append(Segment(word, segments[word_start_idx].start, segments[cur_idx - 1].end, score))
            word_start_idx = cur_idx
        cur_idx += 1
    return words


def frames_postprocessing_single(tokens, token_ids, frame_alignment, frame_alignment_aux, frame_scores, frames, token_type, utt_info, sp_model, frame_dur, aux_offset):
    segments = merge_repeats(frames, tokens, token_ids)
    # for seg in segments:
    #     print(seg)

    sample_rate, text, speaker_id, utter_id, wav_path = utt_info
    
    if token_type == "phoneme" or token_type == "bpe" or token_type == "char":
        word_segments = merge_words_aux(tokens, segments, frame_alignment_aux, sp_model, aux_offset=aux_offset)
    # elif token_type == "char":
    #     word_segments = merge_words_without_seperator(tokens, segments, text, separator=" ")
    # elif token_type == "bpe":
    #     word_segments = merge_words_bpe(tokens, segments)
    else:
        raise NotImplementedError
    
    time_start = -1
    time_end = -1

    words_text = [f"{x.label}" for x in word_segments]
    word_times_start = [x.start * frame_dur for x in word_segments]
    word_times_end = [x.end * frame_dur for x in word_segments]

    phones_text = [f"{x.label}" for x in segments]
    phones_beg_time = [x.start * frame_dur for x in segments]
    phones_end_time = [x.end * frame_dur for x in segments]

    # if len(words_text.split()) != len(text.split()):
    #     import pdb; pdb.set_trace()

    assert len(words_text) == len(text.split()), f"{len(words_text)} vs. {len(text.split())}\n{words_text}\n{text}"

    return utter_id, (time_start, time_end, words_text, word_times_start, word_times_end, phones_text, phones_beg_time, phones_end_time)


@dataclass
class TokenSpan:
    index: int  # index of token in transcript
    start: int  # start time (inclusive)
    end: int  # end time (exclusive)
    score: float

    def __len__(self) -> int:
        return self.end - self.start
    
def merge_tokens(tokens, tokens_aux, scores, blank=0) -> List[TokenSpan]:
    prev_token = blank
    i = start = -1
    spans = []
    for t, (token, token_aux) in enumerate(zip(tokens, tokens_aux)):
        if token != prev_token or token != token_aux:
            if prev_token != blank:
                spans.append(TokenSpan(i, start, t, scores[start:t].mean().item()))
            if token != blank:
                i += 1
                start = t
            prev_token = token
    if prev_token != blank:
        spans.append(TokenSpan(i, start, len(tokens), scores[start:].mean().item()))
    return spans


@dataclass
class WordSpan:
    token_spans: List[TokenSpan]
    score: float


# Obtain word alignments from token alignments
def merge_words(token_spans, tokens_aux, aux_offset=100000) -> List[WordSpan]:
    def _score(t_spans):
        try:
            return sum(s.score * len(s) for s in t_spans) / sum(len(s) for s in t_spans)
        except:
            return 0

    words = []
    i = 0

    for j, span in enumerate(token_spans):
        if j > i and tokens_aux[span.start] >= aux_offset:
            words.append(WordSpan(token_spans[i:j], _score(token_spans[i:j])))
            i = j
    if i < len(token_spans):
        words.append(WordSpan(token_spans[i:], _score(token_spans[i:])))
    return words


def frames_postprocessing_single_new(tokens, token_ids, frame_alignment, frame_alignment_aux, frame_scores, frames, token_type, utt_info, sp_model, frame_dur, aux_offset):
    alignment_scores = frame_scores.exp()
    aligned_tokens = frame_alignment
    aligned_tokens_aux = frame_alignment_aux

    token_spans = merge_tokens(aligned_tokens, aligned_tokens_aux, alignment_scores, sp_model.blank_id)
    word_spans = merge_words(token_spans, aligned_tokens_aux, aux_offset)

    return token_spans, word_spans




# Debug:
# !import code; code.interact(local=vars())
# aa = [i for i in frame_alignment.tolist() if i != sp_model.blank_id]
# bb = [i for i in frame_alignment_aux.tolist() if i != sp_model.blank_id]
# cc = list(zip(aa, bb))
# dd = cc[:1] + [x[1] for x in zip(cc, cc[1:]) if x[0]!=x[1]]
# uu = [x[0] for x in dd]
# [(x.label, y) for x, y in zip(segments, tokens)]

# Analysis:
# val, idx = torch.topk(log_prob, 5, dim=-1)
# idx.tolist()
# val
# for x, y, z in zip(idx.tolist(), val.tolist(), aux_ali):
#   print(f"{x}\n{y}\n{z}\n")



# !import code; code.interact(local=vars())
# aa = [i for i in frame_alignment.tolist() if i != sp_model.blank_id]
# bb = [i for i in frame_alignment_aux.tolist() if i != sp_model.blank_id]
# cc = list(zip(aa, bb))
# dd = cc[:1] + [x[1] for x in zip(cc, cc[1:]) if x[0]!=x[1]]
# uu = [x[0] for x in dd]
# [(x.label, y) for x, y in zip(segments, tokens)]

# import torch
# val, idx = torch.topk(log_prob, 5, dim=-1)
# idx.tolist()
# val
# for x, y, z in zip(idx.tolist(), val.tolist(), aux_ali):
#   print(f"{x}\n{y}\n{z}\n")

# fout = open("analysis4.txt", "w")
# print(utt_info, file=fout)
# print("", file=fout)
# for ii, (x, y, z) in enumerate(zip(idx.tolist(), val.tolist(), aux_ali)):
#   z_ = z - 1 if z > 0 else self.sp_model.blank_id
#   symbol = self.sp_model.id_to_piece(z_) if z_ < self.aux_offset else '_'+self.sp_model.id_to_piece(z_ - self.aux_offset)
#   print(f"{ii}: {ii*frame_dur}\n{x}\n{y}\n{z}: {symbol}\n", file=fout)
# print("\t".join(rs[2]), file=fout)
# print("\t".join(map(lambda x: f"{x:.2f}", rs[3])), file=fout)
# print("\t".join(map(lambda x: f"{x:.2f}", rs[4])), file=fout)
# print("", file=fout)
# print("\t".join(rs[5]), file=fout)
# print("\t".join(map(lambda x: f"{x:.2f}", rs[6])), file=fout)
# print("\t".join(map(lambda x: f"{x:.2f}", rs[7])), file=fout)
# print("", file=fout)
# print(f"aux_ali={aux_ali}", file=fout)
# fout.close()