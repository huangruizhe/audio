import sys
# sys.path.insert(0,'/fsx/users/huangruizhe/k2/k2/python')
# sys.path.insert(0,'/fsx/users/huangruizhe/k2/build_release/lib')
# sys.path.insert(0,'/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc')

# sys.path.insert(0,'/fsx/users/huangruizhe/icefall_align2')
# sys.path.insert(0,'/fsx/users/huangruizhe/icefall_align2/egs/librispeech/ASR/zipformer_mmi')

sys.path.insert(0,'/exp/rhuang/meta/k2/k2/python')
sys.path.insert(0,'/exp/rhuang/meta/k2/build_release/lib')
sys.path.insert(0,'/exp/rhuang/meta/audio/examples/asr/librispeech_conformer_ctc2')

sys.path.insert(0,'/exp/rhuang/meta/icefall')

import torch
import torchaudio
import logging
from dataclasses import dataclass

import sentencepiece as spm
import torchaudio.functional as F

from lightning import ConformerCTCModule
from transforms import TestTransform
from config import load_config, update_config, save_config
from lexicon import Lexicon
from tokenizer import Tokenizer
from loss import MaximumLikelihoodLoss

import k2
import itertools
from train import get_tokenizer
import ali as ali_lib


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
        checkpoint_path,
        sp_model_path,
        global_stats_path,
        config_path,
        device="cpu",
    ) -> None:
        
        if device == "cpu":
            device = torch.device("cpu")
        elif torch.cuda.is_available():
            device = torch.device("cuda", 0)
        self.device = device
        logging.info(f"Device: {device}")

        config = load_config(config_path)
        self.config = config

        tokenizer, lexicon = get_tokenizer(config)
        
        model = ConformerCTCModule.load_from_checkpoint(
            checkpoint_path, 
            tokenizer=tokenizer, 
            lexicon=lexicon, 
            config=config,
            strict=False,
            map_location=device,
        ).eval()
        model = model.to(device=self.device)
        model.initialize_loss_func()

        self.config["training_config"]["checkpoint_path"] = checkpoint_path

        self.tokenizer = tokenizer
        self.lexicon = lexicon
        self.model = model
        self.graph_compiler = None

        self.test_transform = TestTransform(global_stats_path=global_stats_path, sp_model=tokenizer)

        self.scratch_space = {}

    def get_log_prob(self, wav_file, text):
        waveform, sample_rate = torchaudio.load(wav_file)
        sample = waveform, sample_rate, text
        batch, _ = self.test_transform(sample)

        ############ get emission ############
        emissions, src_lengths = self.model(batch, emission_only=True)
        # emission = emissions.permute(1, 0, 2).cpu().detach()
        # emission = torch.roll(emission, 1, -1)  # Now blank symbol has the index of 0
        # # emission = emission.exp()
        emissions = emissions.cpu().detach()
        return emissions, src_lengths, sample

    def align(self, wav_file, text, enable_priors=True):
        emissions, src_lengths, sample = self.get_log_prob(wav_file, text)

        samples = [sample + (None, None, None)]
        labels_ali, aux_labels_ali, log_probs = \
            self.model.loss.align(
                emissions,
                torch.tensor([[1]]), 
                src_lengths, 
                torch.tensor([1]), 
                samples,
                enable_priors,
            )
        # log_probs_original = emissions.permute(1, 0, 2)

        # ret = list()
        # for i, (ali, aux_ali) in enumerate(zip(labels_ali, aux_labels_ali)):
        #     log_prob = log_probs[i][:src_lengths[i].int().item()]
        #     utt_info = samples[i][1:]
        #     model_unit = self.model.config["model_unit"]
        #     frame_dur = self.model.config["subsampling_factor"] * 0.01

        #     tokens, token_ids, frame_alignment, frame_alignment_aux, frame_scores, frames = \
        #         ali_lib.ali_postprocessing_single(
        #             ali, 
        #             aux_ali, 
        #             self.model.tokenizer, 
        #             log_prob, 
        #             aux_offset=self.model.aux_offset
        #         )
            
        #     utter_id, rs = \
        #         ali_lib.frames_postprocessing_single(
        #             tokens, 
        #             token_ids, 
        #             frame_alignment, 
        #             frame_alignment_aux, 
        #             frame_scores, 
        #             frames, 
        #             model_unit, 
        #             utt_info, 
        #             self.model.tokenizer, 
        #             frame_dur, 
        #             self.model.aux_offset
        #         )
            
        #     ret.append((utter_id, rs))

        i = 0
        ali = labels_ali[i]
        aux_ali = aux_labels_ali[i]

        log_prob = log_probs[i][:src_lengths[i].int().item()]
        utt_info = samples[i][1:]
        model_unit = self.model.config["model_unit"]
        frame_dur = self.model.config["subsampling_factor"] * 0.01

        tokens, token_ids, frame_alignment, frame_alignment_aux, frame_scores, frames = \
            ali_lib.ali_postprocessing_single(
                ali, 
                aux_ali, 
                self.model.tokenizer, 
                log_prob, 
                aux_offset=self.model.aux_offset
            )

        token_spans, word_spans = \
            ali_lib.frames_postprocessing_single_new(
                tokens, 
                token_ids, 
                frame_alignment, 
                frame_alignment_aux, 
                frame_scores, 
                frames, 
                model_unit, 
                utt_info, 
                self.model.tokenizer, 
                frame_dur, 
                self.model.aux_offset
            )

        return tokens, token_ids, frame_alignment, frame_scores, frames, token_spans, word_spans


