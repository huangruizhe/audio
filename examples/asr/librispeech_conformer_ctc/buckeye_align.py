import pathlib
from argparse import ArgumentParser

import sentencepiece as spm
from tokenizer_char import CharTokenizer
from tokenizer_char_boundary import CharTokenizerBoundary
from tokenizer_phone_boundary import PhonemeTokenizerBoundary

from lightning import ConformerCTCModule
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
from pytorch_lightning.strategies import DDPStrategy
from buckeye_transforms import get_data_module

from config import load_config, update_config, save_config
import logging
import pickle

logging.getLogger("lightning.pytorch").setLevel(logging.INFO)

# logging.basicConfig(
#     format = "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
#     level = 10
# )

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
    # https://lightning.ai/docs/pytorch/stable/deploy/production_basic.html
    # https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html#on-train-epoch-end

    def on_train_epoch_end(self, trainer, pl_module):
        if pl_module.mode != "align":
            return
        
        logging.info(f"Saving alignment results for worker {pl_module.global_rank} ...")

        ali_dir = pathlib.Path(pl_module.config["training_config"]["exp_dir"]) / "ali"
        ali_dir.mkdir(parents=True, exist_ok=True)
        model_name = pathlib.Path(pl_module.config["training_config"]["checkpoint_path"]).stem
        # torch.save(pl_module.scratch_space["ali"], ali_dir / f"ali_{model_name}_{pl_module.global_rank}.pt")
        with open(ali_dir / f"ali_{model_name}_{pl_module.global_rank}.pkl", 'wb') as file:
            pickle.dump(pl_module.scratch_space["ali"], file)


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
        # save_top_k=config["training_config"]["save_top_k"],
        save_weights_only=False,
        verbose=True,
        every_n_epochs=10,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    if args.mode == "train":
        callbacks = [
            checkpoint,
            train_checkpoint,
            lr_monitor,
            MyFitStartCallback(),
            MyTrainStartCallback(),
        ]
    elif args.mode == "align":
        callbacks = [
            MyFitStartCallback(),
            MyTrainStartCallback(),
            MyTrainEpochEndCallback(),
        ]

    trainer = Trainer(
        default_root_dir=pathlib.Path(config["training_config"]["exp_dir"]),
        max_epochs=config["training_config"]["epochs"],
        num_nodes=config["training_config"]["nodes"],
        devices=config["training_config"]["gpus"],
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
        gradient_clip_val=config["training_config"]["gradient_clip_val"],
        # accumulate_grad_batches=3,
        # limit_train_batches=10,
    )

    if config["model_unit"] == "bpe":
        sp_model = spm.SentencePieceProcessor(model_file=str(args.sp_model_path))
    elif config["model_unit"] == "char":
        sp_model = CharTokenizer()
    elif config["model_unit"] == "char_boundary":
        sp_model = CharTokenizerBoundary()
    elif config["model_unit"] == "phoneme":
        sp_model = PhonemeTokenizerBoundary(has_boundary=False)
    elif config["model_unit"] == "phoneme_boundary":
        sp_model = PhonemeTokenizerBoundary(has_boundary=True)
    model = ConformerCTCModule(sp_model, config)
    
    if trainer.global_rank == 0:
        print(f"Model: \n{model}")
    data_module = get_data_module(str(args.buckeye_path), str(args.global_stats_path), sp_model, config)
    data_module.mode = args.mode
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
        "--buckeye-path",
        type=pathlib.Path,
        help="Path to Buckeye datasets.",
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
    parser.add_argument(
        "--mode",
        type=str,
        help="align or train",
        default="align"
    )
    args = parser.parse_args()

    config = load_config(args.train_config)
    config = update_config(config, args)

    run_train(args, config)


if __name__ == "__main__":
    cli_main()
