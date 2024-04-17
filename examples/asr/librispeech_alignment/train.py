import pathlib
from argparse import ArgumentParser

import sentencepiece as spm
from alignment.tokenizer import (
    EnglishCharTokenizer, 
    EnglishBPETokenizer,
    EnglishPhonemeTokenizer,
)

import torch
from lightning import AcousticModelModule
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
from pytorch_lightning.strategies import DDPStrategy
from transforms import get_data_module


class LabelPriorsCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        log_priors_sums = pl_module.all_gather(pl_module.loss.log_priors_sum)
        log_priors_sums = torch.logsumexp(log_priors_sums, dim=0, keepdim=True)
        num_samples = pl_module.all_gather(pl_module.loss.num_samples)
        num_samples = num_samples.sum().log().to(log_priors_sums.device)
        new_log_prior = log_priors_sums - num_samples
        
        if pl_module.global_rank == 0:
            print("new_priors: ", ["{0:0.2f}".format(i) for i in new_log_prior[0][0].exp().tolist()])
            print("new_log_prior: ", ["{0:0.2f}".format(i) for i in new_log_prior[0][0].tolist()])
            if pl_module.loss.log_priors is not None:
                _a1 = new_log_prior.exp()
                _b1 = pl_module.loss.log_priors.exp()
                print("diff%: ", ["{0:0.2f}".format(i) for i in ((_a1 - _b1)/_b1*100)[0][0].tolist()])

        prior_threshold = -12.0
        new_log_prior = torch.where(new_log_prior < prior_threshold, prior_threshold, new_log_prior)

        pl_module.loss.log_priors = new_log_prior
        pl_module.loss.log_priors_sum = None
        pl_module.loss.num_samples = 0

        if pl_module.global_rank == 0:
            exp_dir = pathlib.Path(trainer.default_root_dir)
            checkpoint_dir = exp_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            label_priors_path = checkpoint_dir / f"log_priors_epoch_{pl_module.current_epoch}.pt"
            torch.save(new_log_prior, label_priors_path)


def run_train(args):
    seed_everything(1)
    checkpoint_dir = args.exp_dir / "checkpoints"
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Losses/val_loss",
        mode="min",
        save_top_k=5,
        save_weights_only=False,
        verbose=True,
    )
    train_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Losses/train_loss",
        mode="min",
        save_top_k=5,
        save_weights_only=False,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [
        checkpoint,
        train_checkpoint,
        lr_monitor,
        LabelPriorsCallback(),
    ]
    trainer = Trainer(
        default_root_dir=args.exp_dir,
        max_epochs=args.epochs,
        num_nodes=args.nodes,
        devices=args.gpus,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
        gradient_clip_val=10.0,
    )

    tokenizer = EnglishPhonemeTokenizer()

    model = AcousticModelModule(tokenizer, prior_scaling_factor=args.alpha)
    data_module = get_data_module(str(args.librispeech_path), str(args.global_stats_path), tokenizer)
    trainer.fit(model, data_module, ckpt_path=args.checkpoint_path)


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
        help="[Optional] Path to SentencePiece model.",
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
        "--alpha",
        default=0.0,
        type=float,
        help="The scaling factor of the label priors",
    )
    args = parser.parse_args()
    run_train(args)


if __name__ == "__main__":
    cli_main()
