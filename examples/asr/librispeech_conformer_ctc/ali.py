from dataclasses import dataclass
import itertools
import torch


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

def ali_postprocessing_single(labels_ali, aux_labels_ali, sp_model, emission, aux_offset=100000):
     # Hard-coded for torchaudio       
    labels_ali = [x - 1 if x > 0 else sp_model.blank_id for x in labels_ali]
    aux_labels_ali = [x - 1 if x > 0 else sp_model.blank_id for x in aux_labels_ali]
    # import pdb; pdb.set_trace()

    ############ get confidence scores and change frame rate ############
    # Ref: https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html
    assert emission.size(0) == len(labels_ali)

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
    indices_aux = [i for i in indices_aux if i != sp_model.blank_id]
    token_ids_aux = []
    i_prev = -1
    for i in indices_aux:
        if i + aux_offset != i_prev:
            token_ids_aux.append(i if i < aux_offset else i - aux_offset)
        i_prev = i
    token_ids = token_ids_aux
    tokens = [sp_model.id_to_piece(i) for i in token_ids]

    assert frames[-1].token_index == len(tokens) - 1
    # if frames[-1].token_index >= len(tokens):  # or tuple(token_ids[:3]) == (34, 71, 83)
    #     import pdb; pdb.set_trace()

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


# Obtain word alignments from token alignments
def merge_words_aux(tokens, segments, frame_alignment_aux, sp_model, aux_offset=100000):
    words = []

    indices_aux = remove_consecutive_duplicates(frame_alignment_aux, aux_offset=aux_offset)
    indices_aux = [i for i in indices_aux if i != sp_model.blank_id]
    token_ids_aux = []
    i_prev = -1
    for i in indices_aux:
        if i + aux_offset != i_prev:
            token_ids_aux.append(i)
        i_prev = i

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


def frames_postprocessing_single(tokens, token_ids, frame_alignment, frame_alignment_aux, frame_scores, frames, token_type, utt_info, sp_model, frame_dur):
    segments = merge_repeats(frames, tokens, token_ids)
    # for seg in segments:
    #     print(seg)
    
    if token_type == "phoneme":
        word_segments = merge_words_aux(tokens, segments, frame_alignment_aux, sp_model, aux_offset = 100000)
    else:
        word_segments = merge_words_bpe(tokens, segments)
    
    time_start = -1
    time_end = -1

    words_text = [f"{x.label}" for x in word_segments]
    word_times_start = [x.start * frame_dur for x in word_segments]
    word_times_end = [x.end * frame_dur for x in word_segments]

    phones_text = [f"{x.label}" for x in segments]
    phones_beg_time = [x.start * frame_dur for x in segments]
    phones_end_time = [x.end * frame_dur for x in segments]

    sample_rate, text, speaker_id, utter_id, wav_path = utt_info

    # if len(words_text.split()) != len(text.split()):
    #     import pdb; pdb.set_trace()

    assert len(words_text) == len(text.split())

    return utter_id, (time_start, time_end, words_text, word_times_start, word_times_end, phones_text, phones_beg_time, phones_end_time)
