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


def get_frames(frame_alignment, frame_scores, blank_id=0):
        assert len(frame_alignment) == len(frame_scores)

        frames = []
        token_index = -1
        prev_hyp = None
        for i in range(len(frame_alignment)):
            token = frame_alignment[i].item()
            if token != prev_hyp:
                if token == blank_id:
                    continue
                token_index += 1
            frames.append(Frame(token_index, i, frame_scores[i].exp().item()))
            prev_hyp = frame_alignment[i].item()
        return frames

def hmm_k2_text_preprocess(text):
    return ''.join(c[0] for c in itertools.groupby(text))


def ali_postprocessing_single(labels_ali, aux_labels_ali, sp_model, emission):
     # Hard-coded for torchaudio       
    labels_ali = [x - 1 if x > 0 else sp_model.blank_id for x in labels_ali]
    aux_labels_ali = [x - 1 if x > 0 else sp_model.blank_id for x in aux_labels_ali]
    # import pdb; pdb.set_trace()

    ############ get confidence scores and change frame rate ############
    # Ref: https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html
    assert emission.size(0) == len(labels_ali)

    frame_alignment = torch.LongTensor(labels_ali)
    frame_scores = emission[torch.arange(len(frame_alignment)), frame_alignment]
    frame_scores = frame_scores.cpu()
    # frame_scores = frame_scores.exp()

    # frame_alignment = frame_alignment.repeat_interleave(subsampling_factor)
    # frame_scores = frame_scores.repeat_interleave(subsampling_factor)
    
    frames = get_frames(frame_alignment, frame_scores, blank_id=sp_model.blank_id)

    # self.scratch_space["decoding_graph"] = decoding_graph
    # self.scratch_space["emission"] = emission
    # self.scratch_space["frame_alignment_aux"] = torch.LongTensor(aux_labels_ali[0])

    indices = torch.unique_consecutive(frame_alignment, dim=-1)
    indices = [i for i in indices if i != sp_model.blank_id]
    joined = "".join([sp_model.id_to_piece(i.item()) for i in indices])
    # return joined.replace("▁", " ").strip().split()
    tokens = [sp_model.id_to_piece(i.item()) for i in indices]
    token_ids = [i.item() for i in indices]

    return tokens, token_ids, frame_alignment, frame_scores, frames
