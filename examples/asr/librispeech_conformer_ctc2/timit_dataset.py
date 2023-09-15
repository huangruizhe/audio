import os
from pathlib import Path
from typing import Tuple, Union
import glob

from torch import Tensor
from torch.utils.data import Dataset
import torchaudio


class TIMIT(Dataset):
    
    def __init__(
        self,
        root: Union[str, Path],
    ) -> None:
        self.root = root
        self.wav_files = glob.glob(f"{root}/**/**/*.wav", recursive=False)
        if len(self.wav_files) == 0:
            self.wav_files = glob.glob(f"{root}/**/**/**/*.wav", recursive=False)

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
        wrd_path = wav_path[:-8] + ".WRD"

        pth = wav_path[:-8].split("/")
        utter_id = "_".join(pth[-4:])
        speaker_id, segment_id = pth[-2], pth[-1]

        with open(wrd_path) as fin:
            lines = fin.readlines()
        text = [line.strip().split()[2] for line in lines]
        text = " ".join([x for x in text if len(x) > 0])

        waveform, sample_rate = torchaudio.load(wav_path)
        return waveform, sample_rate, text, speaker_id, utter_id, wav_path

    def __len__(self) -> int:
        return len(self.wav_files)