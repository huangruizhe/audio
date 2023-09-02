import pathlib
from argparse import ArgumentParser
from tqdm import tqdm

import sentencepiece as spm
from tokenizer_char import CharTokenizer
from tokenizer_char_boundary import CharTokenizerBoundary
from tokenizer_phone_boundary import PhonemeTokenizerBoundary

import torch
from lightning import ConformerCTCModule
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
from pytorch_lightning.strategies import DDPStrategy
from transforms import get_data_module

from config import load_config, update_config, save_config
import logging

logging.getLogger("lightning.pytorch").setLevel(logging.INFO)

logging.basicConfig(
    format = "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    level = 10
)

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


def run_train(args, config):
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
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
        gradient_clip_val=config["training_config"]["gradient_clip_val"],
        # accumulate_grad_batches=3,
        # limit_train_batches=100,
        # limit_val_batches=10,
    )

    if config["model_unit"] == "bpe":
        # sp_model = spm.SentencePieceProcessor(model_file=str(args.sp_model_path))
        sp_model = PhonemeTokenizerBoundary(has_boundary=False, modeling_unit="bpe")
    elif config["model_unit"] == "char":
        # sp_model = CharTokenizer()
        sp_model = PhonemeTokenizerBoundary(has_boundary=False, modeling_unit="char")
    elif config["model_unit"] == "char_boundary":
        sp_model = CharTokenizerBoundary()
    elif config["model_unit"] == "phoneme":
        sp_model = PhonemeTokenizerBoundary(has_boundary=False)
    elif config["model_unit"] == "phoneme_boundary":
        sp_model = PhonemeTokenizerBoundary(has_boundary=True)
    
    if True:
        model = ConformerCTCModule(sp_model, config)
        
        if trainer.global_rank == 0:
            print(f"Model: \n{model}")
        data_module = get_data_module(str(args.librispeech_path), str(args.global_stats_path), sp_model, config)
        trainer.fit(model, data_module, ckpt_path=config["training_config"]["checkpoint_path"])
    else:
        model = ConformerCTCModule.load_from_checkpoint(config["training_config"]["checkpoint_path"], sp_model=sp_model, config=config, strict=False)    
        model.trainer = trainer
        
        if trainer.global_rank == 0:
            print(f"Model: \n{model}")
        data_module = get_data_module(str(args.librispeech_path), str(args.global_stats_path), sp_model, config)
        trainer.fit(model, data_module)


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
        required=True,
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
    args = parser.parse_args()

    config = load_config(args.train_config)
    config = update_config(config, args)

    run_train(args, config)


if __name__ == "__main__":
    cli_main()
