import sys
sys.path.insert(0,'/fsx/users/huangruizhe/k2/k2/python')
sys.path.insert(0,'/fsx/users/huangruizhe/k2/build_release/lib')
# sys.path.insert(0,'/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc')

sys.path.insert(0,'/fsx/users/huangruizhe/icefall_align2')
sys.path.insert(0,'/fsx/users/huangruizhe/icefall_align2/egs/librispeech/ASR/zipformer_mmi')

import torch
import torchaudio
import logging
from dataclasses import dataclass
import torchaudio.functional as F

import sentencepiece as spm

from lightning import ConformerCTCModule
from transforms import TestTransform
from config import load_config, update_config, save_config
from tokenizer_char import CharTokenizer
from tokenizer_char_boundary import CharTokenizerBoundary
from loss import MaximumLikelihoodLoss
from graph_compiler_char import CharCtcTrainingGraphCompiler
from tokenizer_phone_boundary import PhonemeTokenizerBoundary
from graph_compiler_phone import PhonemeCtcTrainingGraphCompiler

from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from icefall.mmi_graph_compiler import MmiTrainingGraphCompiler
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.lexicon import Lexicon, UniqLexicon
from icefall.utils import (
    AttributeDict,
    encode_supervisions,
    get_alignments,
    setup_logger,
)
from icefall.decode import one_best_decoding
import k2
import itertools


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
    ) -> None:
        
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda", 0)
        self.device = device
        logging.info(f"Device: {device}")

        config = load_config(config_path)
        self.config = config

        if config["model_unit"] == "bpe":
            sp_model = spm.SentencePieceProcessor(model_file=str(sp_model_path))
            blank_id = None
        elif config["model_unit"] == "char":
            sp_model = CharTokenizer()
            blank_id = None
        elif config["model_unit"] == "char_boundary":
            sp_model = CharTokenizerBoundary()
            blank_id = None
        elif config["model_unit"] == "phoneme":
            sp_model = PhonemeTokenizerBoundary(has_boundary=False)
            blank_id = sp_model.blank_id
        elif config["model_unit"] == "phoneme_boundary":
            sp_model = PhonemeTokenizerBoundary(has_boundary=True)
            blank_id = sp_model.blank_id
        else:
            raise NotImplementedError
        
        model = ConformerCTCModule.load_from_checkpoint(
            checkpoint_path, 
            sp_model=sp_model, 
            config=config,
            inference_args={"inference_type": "greedy"},
            map_location=self.device,
            blank_idx=blank_id,
        ).eval()
        model = model.to(device=self.device)

        self.sp_model = sp_model
        self.model = model
        self.graph_compiler = None
        self.ml_loss = None

        self.test_transform = TestTransform(global_stats_path=global_stats_path, sp_model=sp_model)

        self.scratch_space = {}

    def align(self, wav_file, text, blank_id=1023):
        waveform, sample_rate = torchaudio.load(wav_file)
        sample = waveform, sample_rate, text
        batch, _ = self.test_transform(sample)

        ############ get emission ############
        emissions = self.model(batch, emission_only=True)
        emission = emissions.permute(1, 0, 2)[0].cpu().detach()

        ############ get alignment ############

        frames = []
        text = text.lower()
        tokens = self.sp_model.encode(text)

        # import pdb; pdb.set_trace()

        targets = torch.tensor(tokens, dtype=torch.int32)
        input_lengths = torch.tensor(emission.shape[0])
        target_lengths = torch.tensor(targets.shape[0])

        # This is the key step, where we call the forced alignment API functional.forced_align to compute alignments.
        frame_alignment, frame_scores = F.forced_align(
            emission, 
            targets, 
            input_lengths, 
            target_lengths, 
            blank=blank_id,
        )

        assert len(frame_alignment) == input_lengths.item()
        assert len(targets) == target_lengths.item()

        token_index = -1
        prev_hyp = 0
        for i in range(len(frame_alignment)):
            if frame_alignment[i].item() == blank_id:
                prev_hyp = 0
                continue

            if frame_alignment[i].item() != prev_hyp:
                token_index += 1
            frames.append(Frame(token_index, i, frame_scores[i].exp().item()))
            prev_hyp = frame_alignment[i].item()

        token_ids = tokens
        tokens = self.sp_model.encode(text, out_type=str)

        return tokens, token_ids, frame_alignment, frame_scores, frames

    def get_frames(self, frame_alignment, frame_scores, blank_id=0):
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

    def hmm_k2_text_preprocess(self, text):
        return ''.join(c[0] for c in itertools.groupby(text))

    def align_k2(self, wav_file, text, topo_type="ctc", subsampling_factor=4, sil_penalty_inter_word=None, sil_penalty_intra_word=None, emission=None):
        if topo_type == "hmm" and self.config["model_unit"].startswith("char"):
            text = self.hmm_k2_text_preprocess(text)
        
        if sil_penalty_inter_word is None:
            sil_penalty_inter_word = float(self.config["sil_penalty_inter_word"])
        
        if sil_penalty_intra_word is None:
            sil_penalty_intra_word = float(self.config["sil_penalty_intra_word"])

        if emission is None:
            waveform, sample_rate = torchaudio.load(wav_file)
            sample = waveform, sample_rate, text
            batch, _ = self.test_transform(sample)

            ############ get emission ############
            emissions = self.model(batch, emission_only=True)
            emission = emissions.permute(1, 0, 2).cpu().detach()
            emission = torch.roll(emission, 1, -1)  # Now blank symbol has the index of 0
            emission = emission.exp()

        ############ get alignment ############
        sp_model = self.sp_model
        aux_offset = 0
        if self.graph_compiler is None:
            if self.config["model_unit"] == "char_boundary":
                sp_model = CharTokenizerBoundary()
                blank_id = None
                self.graph_compiler = CharCtcTrainingGraphCompiler(
                    bpe_model=sp_model,
                    device=self.device,  # torch.device("cuda", self.global_rank),
                    topo_type=topo_type,
                )
            elif self.config["model_unit"] == "phoneme" or self.config["model_unit"] == "phoneme_boundary":
                if self.config["model_unit"] == "phoneme":
                    has_boundary=False
                else:
                    has_boundary=True
                sp_model = PhonemeTokenizerBoundary(has_boundary=has_boundary)
                blank_id = sp_model.blank_id
                aux_offset = 100000
                self.graph_compiler = PhonemeCtcTrainingGraphCompiler(
                    bpe_model=self.sp_model,
                    device=self.device,  # torch.device("cuda", self.global_rank),
                    topo_type=topo_type,
                    index_offset=1,
                    sil_penalty_intra_word=sil_penalty_intra_word,
                    sil_penalty_inter_word=sil_penalty_inter_word,
                    aux_offset=aux_offset,
                )
        
        ml_loss = MaximumLikelihoodLoss(graph_compiler=self.graph_compiler)

        supervision_segments, token_ids, indices = ml_loss.encode_supervisions(batch.targets, batch.target_lengths, torch.tensor([emission.size(1)]))  # batch.feature_lengths
        tokens = [[sp_model.id_to_piece(tid - 1) if tid > 0 else sp_model.id_to_piece(sp_model.blank_id) for tid in token_ids[0]]]
        decoding_graph = self.graph_compiler.compile(token_ids, [sample])

        # import pdb; pdb.set_trace()

        dense_fsa_vec = k2.DenseFsaVec(
            emission,
            supervision_segments,
            allow_truncate=subsampling_factor - 1,
        )

        output_beam = 10  # https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/conformer_ctc/ali.py
        lattice = k2.intersect_dense(
            decoding_graph,
            dense_fsa_vec,
            output_beam,
        )

        best_path = one_best_decoding(
            lattice=lattice,
            use_double_scores=True,
        )

        labels_ali = get_alignments(best_path, kind="labels")
        aux_labels_ali = get_alignments(best_path, kind="aux_labels")

        # Hard-coded for torchaudio       
        labels_ali = [[x - 1 if x > 0 else sp_model.blank_id for x in l] for l in labels_ali]
        aux_labels_ali = [[x - 1 if x > 0 else sp_model.blank_id for x in l] for l in aux_labels_ali]
        # import pdb; pdb.set_trace()

        ############ get confidence scores and change frame rate ############
        # Ref: https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html
        assert emission.size(1) == len(labels_ali[0])

        frame_alignment = torch.LongTensor(labels_ali[0])
        frame_scores = emission[0][torch.arange(len(frame_alignment)), frame_alignment]
        # frame_scores = frame_scores.exp()

        # frame_alignment = frame_alignment.repeat_interleave(subsampling_factor)
        # frame_scores = frame_scores.repeat_interleave(subsampling_factor)
        
        frames = self.get_frames(frame_alignment, frame_scores, blank_id=sp_model.blank_id)

        self.scratch_space["decoding_graph"] = decoding_graph
        self.scratch_space["emission"] = emission
        self.scratch_space["frame_alignment_aux"] = torch.LongTensor(aux_labels_ali[0])

        indices = torch.unique_consecutive(frame_alignment, dim=-1)
        indices = [i for i in indices if i != self.sp_model.blank_id]
        joined = "".join([self.sp_model.id_to_piece(i.item()) for i in indices])
        # return joined.replace("▁", " ").strip().split()
        tokens = [self.sp_model.id_to_piece(i.item()) for i in indices]
        token_ids = [i.item() for i in indices]

        return tokens, token_ids, frame_alignment, frame_scores, frames

    def align_k2_batch(self, wav_file, texts, topo_type="ctc", subsampling_factor=4, sil_penalty_inter_word=None, sil_penalty_intra_word=None, emissions=None, batch=None, samples=None):
        if topo_type == "hmm" and self.config["model_unit"].startswith("char"):
            texts = [self.hmm_k2_text_preprocess(text) for text in texts]
        
        if sil_penalty_inter_word is None:
            sil_penalty_inter_word = float(self.config["sil_penalty_inter_word"])
        
        if sil_penalty_intra_word is None:
            sil_penalty_intra_word = float(self.config["sil_penalty_intra_word"])

        if emissions is None:
            waveform, sample_rate = torchaudio.load(wav_file)
            sample = waveform, sample_rate, text
            batch, _ = self.test_transform(sample)

            ############ get emission ############
            emissions = self.model(batch, emission_only=True)
        emission = emissions.permute(1, 0, 2).cpu().detach()
        emission = torch.roll(emission, 1, -1)  # Now blank symbol has the index of 0
        emission = emission.exp()

        ############ get alignment ############
        sp_model = self.sp_model
        aux_offset = 0
        if self.graph_compiler is None:
            if self.config["model_unit"] == "char_boundary":
                sp_model = CharTokenizerBoundary()
                blank_id = None
                self.graph_compiler = CharCtcTrainingGraphCompiler(
                    bpe_model=sp_model,
                    device=self.device,  # torch.device("cuda", self.global_rank),
                    topo_type=topo_type,
                )
            elif self.config["model_unit"] == "phoneme" or self.config["model_unit"] == "phoneme_boundary":
                if self.config["model_unit"] == "phoneme":
                    has_boundary=False
                else:
                    has_boundary=True
                sp_model = PhonemeTokenizerBoundary(has_boundary=has_boundary)
                blank_id = sp_model.blank_id
                aux_offset = 100000
                self.graph_compiler = PhonemeCtcTrainingGraphCompiler(
                    bpe_model=self.sp_model,
                    device=self.device,  # torch.device("cuda", self.global_rank),
                    topo_type=topo_type,
                    index_offset=1,
                    sil_penalty_intra_word=sil_penalty_intra_word,
                    sil_penalty_inter_word=sil_penalty_inter_word,
                    aux_offset=aux_offset,
                )
        
        if self.ml_loss is None:
            self.ml_loss = MaximumLikelihoodLoss(graph_compiler=self.graph_compiler)

        supervision_segments, token_ids, indices = self.ml_loss.encode_supervisions(batch.targets, batch.target_lengths, torch.tensor([emission.size(1)]))  # batch.feature_lengths
        tokens = [[sp_model.id_to_piece(tid - 1) if tid > 0 else sp_model.id_to_piece(sp_model.blank_id) for tid in token_ids[0]]]
        decoding_graph = self.graph_compiler.compile(token_ids, samples)

        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()

        dense_fsa_vec = k2.DenseFsaVec(
            emission,
            supervision_segments,
            allow_truncate=subsampling_factor - 1,
        )

        output_beam = 10  # https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/conformer_ctc/ali.py
        lattice = k2.intersect_dense(
            decoding_graph,
            dense_fsa_vec,
            output_beam,
        )

        best_path = one_best_decoding(
            lattice=lattice,
            use_double_scores=True,
        )

        labels_ali = get_alignments(best_path, kind="labels")
        aux_labels_ali = get_alignments(best_path, kind="aux_labels")

        # Hard-coded for torchaudio       
        labels_ali = [[x - 1 if x > 0 else sp_model.blank_id for x in l] for l in labels_ali]
        aux_labels_ali = [[x - 1 if x > 0 else sp_model.blank_id for x in l] for l in aux_labels_ali]
        # import pdb; pdb.set_trace()

        ############ get confidence scores and change frame rate ############
        # Ref: https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html
        assert emission.size(1) == len(labels_ali[0])

        frame_alignment = torch.LongTensor(labels_ali[0])
        frame_scores = emission[0][torch.arange(len(frame_alignment)), frame_alignment]
        # frame_scores = frame_scores.exp()

        # frame_alignment = frame_alignment.repeat_interleave(subsampling_factor)
        # frame_scores = frame_scores.repeat_interleave(subsampling_factor)
        
        frames = self.get_frames(frame_alignment, frame_scores, blank_id=sp_model.blank_id)

        self.scratch_space["decoding_graph"] = decoding_graph
        self.scratch_space["emission"] = emission
        self.scratch_space["frame_alignment_aux"] = torch.LongTensor(aux_labels_ali[0])

        indices = torch.unique_consecutive(frame_alignment, dim=-1)
        indices = [i for i in indices if i != self.sp_model.blank_id]
        joined = "".join([self.sp_model.id_to_piece(i.item()) for i in indices])
        # return joined.replace("▁", " ").strip().split()
        tokens = [self.sp_model.id_to_piece(i.item()) for i in indices]
        token_ids = [i.item() for i in indices]

        return tokens, token_ids, frame_alignment, frame_scores, frames


if __name__ == "__main__":
    # cd /fsx/users/huangruizhe/audio_ruizhe/librispeech_conformer_ctc
    
    base_path = "/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc/"
    # aligner = Aligner(
    #     checkpoint_path = f"{base_path}/experiments/ctc1/checkpoints/epoch=198-step=887506.ckpt",
    #     sp_model_path = f"{base_path}/spm_unigram_1023.model",
    #     global_stats_path = f"{base_path}/global_stats.json",
    #     config_path = f"{base_path}/experiments/hmm_char_stride4/checkpoints/epoch=109-step=488659.ckpt"
    # )

    # aligner_torchaudio_ctc = Aligner(
    #     checkpoint_path = f"{base_path}/experiments/ctc_char_stride4/checkpoints/epoch=199-step=888447.ckpt",
    #     sp_model_path = None,        
    #     global_stats_path = f"{base_path}/global_stats.json",
    #     config_path = f"{base_path}/experiments/ctc_char_stride4/train_config.yaml",
    # )

    aligner_torchaudio_ctc = Aligner(
        checkpoint_path = f"{base_path}/experiments/conformer_4_ctc_aug/checkpoints/epoch=195-step=875339.ckpt",
        sp_model_path = None,
        config_path = f"{base_path}/experiments/conformer_4_ctc_aug/train_config.yaml",
        global_stats_path = f"{base_path}/global_stats.json",
    )

    # aligner_torchaudio_hmm = Aligner(
    #     checkpoint_path = f"{base_path}/experiments/hmm_char_stride4/checkpoints/epoch=189-step=844033.ckpt",
    #     sp_model_path = None,
    #     global_stats_path = f"{base_path}/global_stats.json",
    #     config_path = f"{base_path}/experiments/hmm_char_stride4/train_config.yaml",
    # )

    base_path = "/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc/"

    # wav_file = "/fsx/users/huangruizhe/downloads/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
    # text = "I HAD THAT CURIOSITY BESIDE ME AT THIS MOMENT"
    # wav_file = "/fsx/users/huangruizhe/audio_ruizhe/librispeech_conformer_ctc/samples/1688-142285-0000.wav"
    # text = "THERE'S IRON THEY SAY IN ALL OUR BLOOD AND A GRAIN OR TWO PERHAPS IS GOOD BUT HIS HE MAKES ME HARSHLY FEEL HAS GOT A LITTLE TOO MUCH OF STEEL ANON"    
    wav_file = "/fsx/users/huangruizhe/audio_ruizhe/librispeech_conformer_ctc/samples/2609-156975-0012.wav"
    text = "THAT THE HEBREWS WERE RESTIVE UNDER THIS TYRANNY WAS NATURAL INEVITABLE"

    aligner_torchaudio_ctc.align(wav_file, text, blank_id=aligner_torchaudio_ctc.sp_model.blank_id)
    # aligner_torchaudio_ctc.align_k2(wav_file, text, topo_type="ctc")
    # aligner_torchaudio_hmm.align_k2(wav_file, text, topo_type="hmm")

    print("here")