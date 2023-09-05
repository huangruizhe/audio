import torch
import torchaudio
import logging
from dataclasses import dataclass
import torchaudio.functional as F


logging.basicConfig(
    format = "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    level = 10
)

@dataclass
class Frame:
    # This is the index of each token in the transcript,
    # i.e. the current frame aligns to the N-th character from the transcript.
    token_index: int
    time_index: int
    score: float


class Aligner:
    def __init__(
        self,
    ) -> None:
        
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda", 0)
        self.device = device
        logging.info(f"Device: {device}")

        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        model = bundle.get_model().to(device)
        labels = bundle.get_labels()
        
        self.model = model
        self.labels = labels
        
        dictionary = {c: i for i, c in enumerate(self.labels)}
        self.dictionary = dictionary
        print(f"Model dictionary = {dictionary}")

    def align_ctc(self, wav_file, text):
        if "|" not in text:
            text = text.replace(" ", "|")

        ############ get emission ############
        with torch.inference_mode():
            waveform, _ = torchaudio.load(wav_file)
            emissions, _ = self.model(waveform.to(self.device))
            emissions = torch.log_softmax(emissions, dim=-1)

        emission = emissions[0].cpu().detach()

        ############ get alignment ############

        frames = []
        tokens = [self.dictionary[c] for c in text.replace(" ", "")]

        targets = torch.tensor(tokens, dtype=torch.int32)
        input_lengths = torch.tensor(emission.shape[0])
        target_lengths = torch.tensor(targets.shape[0])

        # This is the key step, where we call the forced alignment API functional.forced_align to compute alignments.
        frame_alignment, frame_scores = F.forced_align(emission, targets, input_lengths, target_lengths, 0)

        assert len(frame_alignment) == input_lengths.item()
        assert len(targets) == target_lengths.item()

        token_index = -1
        prev_hyp = 0
        for i in range(len(frame_alignment)):
            if frame_alignment[i].item() == 0:
                prev_hyp = 0
                continue

            if frame_alignment[i].item() != prev_hyp:
                token_index += 1
            frames.append(Frame(token_index, i, frame_scores[i].exp().item()))
            prev_hyp = frame_alignment[i].item()

        token_ids = tokens
        tokens = [c for c in text.replace(" ", "")]

        return tokens, token_ids, frame_alignment, frame_scores, frames

    def align_hmm(self, wav_file, text):
        pass

    def align(self, wav_file, text, topo_type="ctc"):
        if topo_type == "ctc":
            return self.align_ctc(wav_file, text)
        elif topo_type == "hmm":
            return self.align_hmm(wav_file, text)
        else:
            return None



