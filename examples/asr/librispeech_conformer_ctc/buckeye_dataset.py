import os
from pathlib import Path
from typing import Tuple, Union
import glob

from torch import Tensor
from torch.utils.data import Dataset
import torchaudio


class BUCKEYE(Dataset):
    
    def __init__(
        self,
        root: Union[str, Path],
    ) -> None:
        self.root = root
        self.wav_files = glob.glob(f"{root}/**/*.wav", recursive=False)

    def __getitem__(self, n: int): #  -> Tuple[Tensor, int, str, int, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform
            int:
                Sample rate
            str:
                Transcript
            int:
                Speaker ID
            int:
                Utterance ID
        """
        wav_path = self.wav_files[n]
        lab_path = wav_path[:-4] + ".lab"

        utter_id = Path(wav_path).stem
        speaker_id, segment_id = utter_id.split("-")

        with open(lab_path) as fin:
            text = fin.readlines()
        text = text[0].strip()

        waveform, sample_rate = torchaudio.load(wav_path)
        return waveform, sample_rate, text, speaker_id, utter_id, wav_path

    def __len__(self) -> int:
        return len(self.wav_files)