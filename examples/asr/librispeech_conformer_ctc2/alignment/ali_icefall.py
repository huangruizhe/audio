import sys
sys.path.insert(0,'/fsx/users/huangruizhe/k2/k2/python')
sys.path.insert(0,'/fsx/users/huangruizhe/k2/build_release/lib')
sys.path.insert(0,'/fsx/users/huangruizhe/icefall_align2')
sys.path.insert(0,'/fsx/users/huangruizhe/icefall_align2/egs/librispeech/ASR/zipformer_mmi')

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
import sentencepiece as spm
from pathlib import Path
import logging
import argparse
from dataclasses import dataclass


from asr_datamodule import LibriSpeechAsrDataModule
from decode import get_parser, get_decoding_params
from train import add_model_arguments, get_ctc_model, get_params
import matplotlib.pyplot as plt

logging.basicConfig(
    format = "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    level = 10
)

import torch
import k2

from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy
from lhotse import Recording, SupervisionSegment, MonoCut
from torch.utils.data import DataLoader
from lhotse.dataset import (  # noqa F401 for PrecomputedFeatures
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    SimpleCutSampler,
    BucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SingleCutSampler,
    SpecAugment,
)
from lhotse.dataset.input_strategies import (  # noqa F401 For AudioSamples
    AudioSamples,
    OnTheFlyFeatures,
)
from lhotse.dataset.vis import plot_batch


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
        topo_type="hmm",
        lang_dir="/fsx/users/huangruizhe/icefall_align2/egs/librispeech/ASR/data/lang_bpe_500",
    ) -> None:
        self.checkpoint_path = checkpoint_path

        parser = get_parser()
        LibriSpeechAsrDataModule.add_arguments(parser)
        args = parser.parse_args("")

        params = get_params()
        params.update(get_decoding_params())
        params.update(vars(args))

        params.lang_dir = Path(lang_dir)
        params.topo_type = topo_type
        params.bpe_model = params.lang_dir / "bpe.model"

        self.params = params

        lexicon = Lexicon(params.lang_dir)
        max_token_id = max(lexicon.tokens)
        num_classes = max_token_id + 1  # +1 for the blank
        params.vocab_size = num_classes
        params.blank_id = 0

        logging.info(f"params: {params}")

        bpe_model = spm.SentencePieceProcessor()
        bpe_model.load(str(params.bpe_model))
        self.bpe_model = bpe_model
        self.lexicon = lexicon

        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda", 0)
        self.device = device
        logging.info(f"Device: {device}")

        self.ctc_graph_compiler = BpeCtcTrainingGraphCompiler(
            params.lang_dir,
            device=device,
            sos_token="<sos/eos>",
            eos_token="<sos/eos>",
            topo_type=params.topo_type,
        )
        self.mmi_graph_compiler = MmiTrainingGraphCompiler(
            params.lang_dir,
            uniq_filename="lexicon.txt",
            device=device,
            oov="<UNK>",
            sos_id=1,
            eos_id=1,
            topo_type=params.topo_type,
            bpe_model=bpe_model,
        )

        model = get_ctc_model(params)
        load_checkpoint(checkpoint_path, model, strict=True)
        model.to(device)
        model.eval()
        self.model = model

        HP = self.mmi_graph_compiler.ctc_topo_P
        HP.scores *= params.hp_scale
        if not hasattr(HP, "lm_scores"):
            HP.lm_scores = HP.scores.clone()
        logging.info(f"HP.num_arcs = {HP.num_arcs}")
        self.HP = HP  # TODO: try using H only

        num_param = sum([p.numel() for p in model.parameters()])
        logging.info(f"Number of model parameters: {num_param}")


    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        # logging.debug("About to create test dataset")
        test = K2SpeechRecognitionDataset(
            input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
            return_cuts=True,
        )
        sampler = SimpleCutSampler(cuts, max_duration=100)
        # sampler = BucketingSampler(cuts, max_duration=200, shuffle=False, num_buckets=2)
        # sampler = DynamicBucketingSampler(
        #     cuts,
        #     max_duration=200,
        #     shuffle=False,
        #     num_buckets=2,
        # )

        # logging.debug("About to create test dataloader")
        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
        )
        return test_dl
    
    def prepare_cuts(self, wav_file, text):
        rec = Recording.from_file(wav_file)

        sup = SupervisionSegment(
            id=rec.id,
            recording_id=rec.id,
            start=0, 
            duration=rec.duration,
            channel=0,
            text=text
        )

        cut1 = MonoCut(
            id='rec1-cut1', 
            start=0, 
            duration=rec.duration, 
            channel=0, 
            recording=rec,
            supervisions=[sup]
        )

        cuts = CutSet.from_cuts([cut1])
        return cuts
    
    def get_frames(self, frame_alignment, frame_scores):
        assert len(frame_alignment) == len(frame_scores)

        frames = []
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
        return frames


    def align(self, wav_file, text):
        ############ make lhotse cuts ############
        cuts = self.prepare_cuts(wav_file, text)

        dl = self.test_dataloaders(cuts)
        _, batch = next(enumerate(dl))

        ############ get nnet_output ############
        feature = batch["inputs"]

        # at entry, feature is [N, T, C]
        assert feature.ndim == 3
        feature = feature.to(self.device)

        supervisions = batch["supervisions"]
        cut_list = supervisions["cut"]

        for cut in cut_list:
            assert len(cut.supervisions) == 1, f"{len(cut.supervisions)}"

        feature_lens = supervisions["num_frames"].to(self.device)

        nnet_output, encoder_out_lens = self.model(x=feature, x_lens=feature_lens)

        # nnet_output is [N, T, C]
        supervision_segments, texts = encode_supervisions(
            supervisions, subsampling_factor=self.params.subsampling_factor
        )

        new2old = supervision_segments[:, 0].tolist()

        ############ get alignment through decoding ############
        cut_list = [cut_list[i] for i in new2old]

        token_ids = self.ctc_graph_compiler.texts_to_ids(texts)
        tokens = [self.bpe_model.id_to_piece(tid) for tid in token_ids]
        decoding_graph = self.ctc_graph_compiler.compile(token_ids)

        dense_fsa_vec = k2.DenseFsaVec(
            nnet_output,
            supervision_segments,
            allow_truncate=self.params.subsampling_factor - 1,
        )

        lattice = k2.intersect_dense(
            decoding_graph,
            dense_fsa_vec,
            self.params.output_beam,
        )

        best_path = one_best_decoding(
            lattice=lattice,
            use_double_scores=self.params.use_double_scores,
        )

        labels_ali = get_alignments(best_path, kind="labels")
        aux_labels_ali = get_alignments(best_path, kind="aux_labels")
        
        ############ get confidence scores and change frame rate ############
        # Ref: https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html
        assert nnet_output.size(1) == len(labels_ali[0])

        frame_alignment = torch.LongTensor(labels_ali[0])
        frame_scores = nnet_output[0][torch.arange(len(frame_alignment)), frame_alignment]
        # frame_scores = frame_scores.exp()

        frame_alignment = frame_alignment.repeat_interleave(self.params.subsampling_factor)
        frame_scores = frame_scores.repeat_interleave(self.params.subsampling_factor)
        
        frames = self.get_frames(frame_alignment, frame_scores)

        return tokens[0], token_ids[0], frame_alignment, frame_scores, frames



