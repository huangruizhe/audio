import pathlib
from argparse import ArgumentParser

import sentencepiece as spm

from lightning import ConformerRNNTModule
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from transforms import get_data_module


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

    sp_model = spm.SentencePieceProcessor(model_file=str(args.sp_model_path))
    model = ConformerRNNTModule(sp_model)

    # import numpy as np
    
    # total_params = sum([np.prod(p.size()) for p in model.model.parameters()])
    # trainable_model_parameters = filter(lambda p: p.requires_grad, model.model.parameters())
    # trainable_params = sum([np.prod(p.size()) for p in trainable_model_parameters])
    
    # encoder_params = sum([np.prod(p.size()) for p in model.model.transcriber.parameters()])
    # encoder_input_linear_params = sum([np.prod(p.size()) for p in model.model.transcriber.input_linear.parameters()])
    # encoder_conformer_params = sum([np.prod(p.size()) for p in model.model.transcriber.conformer.parameters()])
    # encoder_conformer0_params = sum([np.prod(p.size()) for p in model.model.transcriber.conformer.conformer_layers[0].parameters()])

    # decoder_params = sum([np.prod(p.size()) for p in model.model.predictor.parameters()])
    # joint_network_params = sum([np.prod(p.size()) for p in model.model.joiner.parameters()])

    # with open("/fsx/users/huangruizhe/audio_ruizhe/librispeech_conformer_rnnt/model_info.txt", "w") as fout:
    #     print(model, file=fout)
    #     print("", file=fout)
    #     print(f"#total_params={total_params}", file=fout)
    #     print(f"#trainable_params={trainable_params}", file=fout)
    #     print("", file=fout)
        
    #     print(f"#encoder_params={encoder_params}", file=fout)
    #     print(f"#encoder_input_linear_params={encoder_input_linear_params}", file=fout)
    #     print(f"#encoder_conformer_params={encoder_conformer_params}", file=fout)
    #     print(f"#encoder_conformer0_params={encoder_conformer0_params}", file=fout)
    #     print("", file=fout)

    #     print(f"#decoder_params={decoder_params}", file=fout)
    #     print("", file=fout)

    #     print(f"#joint_network_params={joint_network_params}", file=fout)

    data_module = get_data_module(str(args.librispeech_path), str(args.global_stats_path), str(args.sp_model_path))
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
        default=200,
        type=int,
        help="Number of epochs to train for. (Default: 120)",
    )
    args = parser.parse_args()
    run_train(args)


if __name__ == "__main__":
    cli_main()
