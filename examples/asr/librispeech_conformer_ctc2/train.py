import pathlib
from argparse import ArgumentParser
from tqdm import tqdm

import sentencepiece as spm
from lexicon import Lexicon
from tokenizer import Tokenizer

import torch
from lightning import ConformerCTCModule
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
from pytorch_lightning.strategies import DDPStrategy
from transforms import get_data_module as get_data_module_librispeech
from buckeye_transforms import get_data_module as get_data_module_buckeye

from config import load_config, update_config, save_config
import logging
import pickle


logging.basicConfig(
    format = "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    level = 10
)
logging.getLogger("lightning.pytorch").setLevel(logging.INFO)

class MyFitStartCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        pl_module.initialize_loss_func(
            # topo_type=pl_module.config["topo_type"], 
            # subsampling_factor=pl_module.config["rnnt_config"]["time_reduction_stride"],
        )

class MyTrainStartCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        if pl_module.global_rank == 0:
            logging.info("Training is starting ...")

            print("----------------- Training Configuration -------------------")
            print(pl_module.config)
            print("------------------------------------------------------------")

            config_file = pathlib.Path(pl_module.config["training_config"]["exp_dir"]) / "train_config.yaml"
            config_file = config_file.absolute()
            logging.info(f"Saving config to: {config_file}")
            save_config(pl_module.config, config_file)


class MyTrainEpochEndCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # import pdb; pdb.set_trace()
        log_priors_sums = pl_module.all_gather(pl_module.loss.log_priors_sum)
        log_priors_sums = torch.logsumexp(log_priors_sums, dim=0, keepdim=True)
        priors_Ts = pl_module.all_gather(pl_module.loss.priors_T)
        # log_priors_Ts = torch.Tensor([priors_Ts]).log().to(log_priors_sums.device)
        log_priors_Ts = priors_Ts.sum().log().to(log_priors_sums.device)
        new_log_prior = log_priors_sums - log_priors_Ts
        
        _a1 = log_priors_sums.exp().sum()
        _b1 = priors_Ts.sum()
        assert abs(_a1 - _b1) / _b1 < 1e-4, f"{_a1} vs. {_b1}"

        if pl_module.global_rank == 0:
            print("new_priors: ", ["{0:0.2f}".format(i) for i in new_log_prior[0][0].exp().tolist()])
            print("new_log_prior: ", ["{0:0.2f}".format(i) for i in new_log_prior[0][0].tolist()])
            if pl_module.loss.log_priors is not None:
                _a1 = new_log_prior.exp()
                _b1 = pl_module.loss.log_priors.exp()
                print("diff%: ", ["{0:0.2f}".format(i) for i in ((_a1 - _b1)/_b1*100)[0][0].tolist()])

        prior_threshold = -12.0
        new_log_prior = torch.where(new_log_prior < prior_threshold, prior_threshold, new_log_prior)
        if pl_module.global_rank == 0:
            print("new_log_prior (clipped): ", ["{0:0.2f}".format(i) for i in new_log_prior[0][0].tolist()])

        # new_log_prior = [-1.18, -3.50, -4.33, -4.15, -3.52, -5.91, -4.17, -3.85, -3.18, -4.20, -4.64, -11.05, -5.83, -4.20, -4.82, -3.40, -2.97, -3.77, -11.22, -5.10, -4.43, -12.00, -4.28, -12.00, -5.73, -6.69, -4.54, -5.69, -4.32, -5.44, -7.63, -5.38, -5.54, -5.21, -3.59, -4.74, -4.45, -3.96, -5.72, -4.91, -5.07, -5.08, -4.69, -5.66, -5.95, -5.30, -4.91, -4.51, -3.98, -4.14, -4.70, -4.43, -7.89, -12.00, -12.00, -8.19, -5.45, -7.07, -12.00, -6.02, -5.23, -5.55, -4.57, -6.17, -4.58, -10.39, -4.91, -7.79, -7.70, -5.67, -5.07, -4.57, -6.74, -5.06, -6.25, -6.17, -12.00, -8.91, -12.00, -6.96, -12.00, -7.99, -12.00, -12.00, -7.57, -12.00, -12.00, -12.00, -9.81, -4.11, -9.70, -12.00, -12.00, -12.00]
        # new_log_prior = torch.tensor(new_log_prior).to(log_priors_sums.device)
        pl_module.loss.log_priors = new_log_prior
        pl_module.loss.log_priors_sum = None
        pl_module.loss.priors_T = 0

        if pl_module.global_rank == 0:
            ck_path = pathlib.Path(pl_module.config["training_config"]["exp_dir"]) / "checkpoints"
            ck_path.mkdir(parents=True, exist_ok=True)
            torch.save(new_log_prior, ck_path / f"log_priors_epoch_{pl_module.current_epoch}.pt")


class MyTrainEpochEndCallbackAlignment(Callback):
    # https://lightning.ai/docs/pytorch/stable/deploy/production_basic.html
    # https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html#on-train-epoch-end

    def on_train_epoch_end(self, trainer, pl_module):
        if pl_module.mode != "align":
            return
        
        ali_dir = pathlib.Path(pl_module.config["training_config"]["exp_dir"]) / "ali"
        ali_dir.mkdir(parents=True, exist_ok=True)
        model_name = pathlib.Path(pl_module.config["training_config"]["checkpoint_path"]).stem
        # torch.save(pl_module.scratch_space["ali"], ali_dir / f"ali_{model_name}_{pl_module.global_rank}.pt")
        print(f"Saving alignment results for worker {pl_module.global_rank}: " + str(ali_dir / f"ali_{model_name}_{pl_module.global_rank}.pkl"))
        with open(ali_dir / f"ali_{model_name}_{pl_module.global_rank}.pkl", 'wb') as file:
            pickle.dump(pl_module.scratch_space["ali"], file)
        
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.Strategy.html#lightning.pytorch.strategies.Strategy.barrier
        trainer.strategy.barrier()


class MyTrainEpochEndCallback_Priors(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # https://github.com/Lightning-AI/lightning/issues/5552
        # self.eval()
        # for idx, batch in enumerate(self.train_dataloader()):

        # pl_module.eval()
        # a=next(enumerate(trainer.train_dataloader))

        update_priors_freq = 1
        if pl_module.current_epoch % update_priors_freq != 0:
            return
        
        if pl_module.global_rank == 0:
            print("Computing priors ...")        

        # import pdb; pdb.set_trace()
        pl_module.eval()
        priors_sum = None
        priors_T = 0
        for idx, batch in tqdm(enumerate(trainer.train_dataloader)):
            labels_ali_, aux_labels_ali_, log_probs_ = pl_module.get_ali_for_priors(batch, idx)

            labels_ali = []
            for ali_ in labels_ali_:
                labels_ali[0:0] = ali_
            labels_ali = torch.tensor(labels_ali)
            ids, cnts = labels_ali.unique(return_counts=True)

            if priors_sum is None:
                priors_sum = torch.zeros((1, log_probs_.size(-1)))
            priors_sum[0][ids] += cnts
            priors_T += len(labels_ali)

        pl_module.loss.log_priors_sum = priors_sum.log()
        pl_module.loss.priors_T = priors_T
    
        cb = MyTrainEpochEndCallback()
        cb.on_train_epoch_end(trainer, pl_module)


def get_tokenizer(config):
    if config["model_unit"] == "bpe":
        # sp_model_path = "/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_rnnt/spm_unigram_1023.model"
        sp_model_path = "/exp/rhuang/meta/audio/examples/asr/librispeech_conformer_ctc/spm_unigram_1023.model"
        sp_model = spm.SentencePieceProcessor(model_file=str(sp_model_path))
        token2id = {sp_model.id_to_piece(i): i + 1 for i in range(sp_model.vocab_size())}
        assert "-" not in token2id
        token2id["-"] = 0
        blank_token = '-'
        unk_token = '<unk>'
        modeling_unit = "bpe"
    elif config["model_unit"] == "char":
        token2id = {'-': 0, '@': 1, 'e': 2, 't': 3, 'a': 4, 'o': 5, 'n': 6, 'i': 7, 'h': 8, 's': 9, 'r': 10, 'd': 11, 'l': 12, 'u': 13, 'm': 14, 'w': 15, 'c': 16, 'f': 17, 'g': 18, 'y': 19, 'p': 20, 'b': 21, 'v': 22, 'k': 23, "'": 24, 'x': 25, 'j': 26, 'q': 27, 'z': 28}
        blank_token = '-'
        unk_token = '@'
        modeling_unit = "char"
        sp_model_path = None
    elif config["model_unit"] == "phoneme":
        phone_set = ['ə', 'ɛ', 'd', 'ɪ', 'ɾ', 't', 'm', 'n', 'ɫ', 'i', 'ɫ̩', 'a', 'ɚ', 'ʔ', 'ɹ', 's', 'z', 'ɔ', 'ɐ', 'v', 'spn', 'ej', 'e', 'ɑ', 'ɑː', 'ɒ', 'dʲ', 'iː', 'dʒ', 'vʲ', 'ɒː', 'bʲ', 'tʃ', 'æ', 'b', 'ow', 'aj', 'cʰ', 'p', 'kʰ', 'pʰ', 'k', 'j', 'ʊ', 'ɡ', 'ʎ', 'l', 'w', 'f', 'h', 'ʉː', 'ʉ', 'uː', 'u', 'ɛː', 'ɲ', 'pʲ', 'o', 'əw', 'θ', 'tʲ', 'ʃ', 'c', 'tʰ', 'n̩', 'ŋ', 'ʒ', 'tʷ', 'mʲ', 'ç', 'ɝ', 'ɔj', 'aw', 'ɟ', 'fʲ', 'aː', 'ɜː', 'vʷ', 'kʷ', 'ɜ', 'cʷ', 'ɾʲ', 'ɡb', 'ð', 'ɾ̃', 'kp', 'ɡʷ', 'ɟʷ', 'd̪', 't̪', 'pʷ', 'm̩', 'fʷ']
        token2id = {p: i + 1 for i, p in enumerate(phone_set)}
        token2id["-"] = 0
        blank_token = "-"
        unk_token = "spn"
        modeling_unit = "phoneme"
        sp_model_path = None

    lexicon = Lexicon(
        # files=[
        #     "/fsx/users/huangruizhe/audio_ruizhe/librispeech_conformer_ctc/librispeech_english_us_mfa.prob.dict",
        #     "/fsx/users/huangruizhe/audio_ruizhe/librispeech_conformer_ctc/librispeech_english_us_mfa.new_words.dict",
        #     "/fsx/users/huangruizhe/datasets/Buckeye_Corpus2/buckeye_words.dict",
        # ],
        files=[
            "/exp/rhuang/meta/audio_ruizhe/librispeech_conformer_ctc/librispeech_english_us_mfa.prob.dict",
            "/exp/rhuang/meta/audio_ruizhe/librispeech_conformer_ctc/librispeech_english_us_mfa.new_words.dict",
            "/exp/rhuang/buckeye/datasets/Buckeye_Corpus2/buckeye_words.dict",
        ],
        modeling_unit=modeling_unit,
    )

    tokenizer = Tokenizer(
        has_boundary=False,
        modeling_unit=modeling_unit,
        lexicon=lexicon,
        token2id=token2id,
        blank_token=blank_token,
        unk_token=unk_token,
        sp_model_path=sp_model_path,
    )

    if modeling_unit != "phoneme":
        lexicon.populate_lexicon_with_tokenizer(tokenizer)

    return tokenizer, lexicon


def run_train_libri(args, config):
    seed_everything(1)
    checkpoint_dir = pathlib.Path(config["training_config"]["exp_dir"]) / "checkpoints"
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Losses/val_loss",
        mode="min",
        save_top_k=config["training_config"]["save_top_k"],
        save_weights_only=False,
        verbose=True,
    )
    train_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Losses/train_loss",
        mode="min",
        save_top_k=config["training_config"]["save_top_k"],
        save_weights_only=False,
        verbose=True,
        # every_n_epochs=1,
        every_n_epochs=10,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [
        checkpoint,
        train_checkpoint,
        lr_monitor,
        MyFitStartCallback(),
        MyTrainStartCallback(),
        MyTrainEpochEndCallback(),
        # MyTrainEpochEndCallback_Priors(),
    ]
    trainer = Trainer(
        default_root_dir=pathlib.Path(config["training_config"]["exp_dir"]),
        max_epochs=config["training_config"]["epochs"],
        # max_steps=500,
        num_nodes=config["training_config"]["nodes"],
        devices=config["training_config"]["gpus"],
        accelerator="gpu" if config["training_config"]["gpus"] > 0 else "cpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
        gradient_clip_val=config["training_config"]["gradient_clip_val"],
        # accumulate_grad_batches=3,
        # limit_train_batches=100,
        # limit_val_batches=10,
    )

    tokenizer, lexicon = get_tokenizer(config)
    model = ConformerCTCModule(tokenizer, lexicon, config)
        
    if trainer.global_rank == 0:
        print(f"Model: \n{model}")
    data_module = get_data_module_librispeech(str(args.librispeech_path), str(args.global_stats_path), tokenizer, config)
    trainer.fit(model, data_module, ckpt_path=config["training_config"]["checkpoint_path"])


def run_train_buckeye(args, config):
    seed_everything(1)
    checkpoint_dir = pathlib.Path(config["training_config"]["exp_dir"]) / "checkpoints"
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Losses/val_loss",
        mode="min",
        save_top_k=config["training_config"]["save_top_k"],
        save_weights_only=False,
        verbose=True,
        every_n_epochs=1,
    )
    train_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Losses/train_loss",
        mode="min",
        save_top_k=config["training_config"]["save_top_k"],
        save_weights_only=False,
        verbose=True,
        every_n_epochs=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    if args.mode == "train" or args.mode == "pseudo":
        callbacks = [
            checkpoint,
            train_checkpoint,
            lr_monitor,
            MyFitStartCallback(),
            MyTrainStartCallback(),
            # MyTrainEpochEndCallback(),
        ]
    elif args.mode == "align":
        callbacks = [
            MyFitStartCallback(),
            MyTrainStartCallback(),
            MyTrainEpochEndCallbackAlignment()
        ]

    checkpoint_epoch = str(config["training_config"]["checkpoint_path"].stem)
    checkpoint_epoch = int(checkpoint_epoch[6: checkpoint_epoch.find("-")])
    assert checkpoint_epoch < config["training_config"]["epochs"]

    trainer = Trainer(
        default_root_dir=pathlib.Path(config["training_config"]["exp_dir"]),
        max_epochs=checkpoint_epoch + 2 if args.mode == "align" else config["training_config"]["epochs"],
        num_nodes=config["training_config"]["nodes"],
        devices=config["training_config"]["gpus"],
        accelerator="gpu" if config["training_config"]["gpus"] > 0 else "cpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
        gradient_clip_val=config["training_config"]["gradient_clip_val"],
        # accumulate_grad_batches=3,
        # limit_train_batches=10,
        # min_epochs=checkpoint_epoch,

        # align:
        limit_val_batches=0,
        num_sanity_val_steps=0
    )

    tokenizer, lexicon = get_tokenizer(config)
    model = ConformerCTCModule(tokenizer, lexicon, config)

    model.mode = args.mode
    if trainer.global_rank == 0:
        print(f"Model: \n{model}")
    data_module = get_data_module_buckeye(str(args.buckeye_path), str(args.global_stats_path), tokenizer, config, train_shuffle=(model.mode!="align"))
    trainer.fit(model, data_module, ckpt_path=config["training_config"]["checkpoint_path"])


def cli_main():
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        type=pathlib.Path,
        help="Path to checkpoint to use for evaluation.",
    )
    parser.add_argument(
        "--exp-dir",
        default=pathlib.Path("./exp"),
        type=pathlib.Path,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
    )
    parser.add_argument(
        "--global-stats-path",
        default=pathlib.Path("global_stats.json"),
        type=pathlib.Path,
        help="Path to JSON file containing feature means and stddevs.",
    )
    parser.add_argument(
        "--librispeech-path",
        type=pathlib.Path,
        help="Path to LibriSpeech datasets.",
        # required=True,
        default=None,
    )
    parser.add_argument(
        "--buckeye-path",
        type=pathlib.Path,
        help="Path to Buckeye datasets.",
        # required=True,
        default=None,
    )
    parser.add_argument(
        "--sp-model-path",
        type=pathlib.Path,
        help="Path to SentencePiece model.",
        required=True,
    )
    parser.add_argument(
        "--nodes",
        default=4,
        type=int,
        help="Number of nodes to use for training. (Default: 4)",
    )
    parser.add_argument(
        "--gpus",
        default=8,
        type=int,
        help="Number of GPUs per node to use for training. (Default: 8)",
    )
    parser.add_argument(
        "--epochs",
        default=120,
        type=int,
        help="Number of epochs to train for. (Default: 120)",
    )
    parser.add_argument(
        "--train-config",
        default=None,
        type=pathlib.Path,
        help="Path to config file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="align or train (train on buckeye is actually finetune)",
        default="align"
    )
    args = parser.parse_args()

    config = load_config(args.train_config)
    config = update_config(config, args)

    if args.librispeech_path is not None:
        run_train_libri(args, config)
    elif args.buckeye_path is not None:
        run_train_buckeye(args, config)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    cli_main()
