from typing import Any
import yaml
import pathlib
import logging
import datetime
import math

logging.getLogger("lightning.pytorch").setLevel(logging.INFO)

class config_dict(dict):
    # https://stackoverflow.com/questions/2328235/pythonextend-the-dict-class
    def __getitem__(self, __key: Any) -> Any:
        return super().__getitem__(__key)
    

default_config = {
    # model:
    # "spm_vocab_size": 1023,
    # "spm_vocab_size": 29,  # char
    # "spm_vocab_size": 1024,  # bpe
    "spm_vocab_size": 94,  # phoneme
    # "spm_vocab_size": 55,
    # "spm_vocab_size": 181,

    # # Xiaohui's
    # "rnnt_config": {
    #     "input_dim": 80,
    #     "encoding_dim": 512,
    #     "subsampling_type": "conv",  # splice, conv
    #     "time_reduction_stride": 2,
    #     "conformer_input_dim": 512,
    #     "conformer_ffn_dim": 2048,
    #     "conformer_num_layers": 12,
    #     "conformer_num_heads": 8,
    #     "conformer_depthwise_conv_kernel_size": 31,
    #     "conformer_dropout": 0.1,
    #     "num_symbols": 94,  ########
    #     "symbol_embedding_dim": 1024,
    #     "num_lstm_layers": 2,
    #     "lstm_hidden_dim": 512,
    #     "lstm_layer_norm": True,
    #     "lstm_layer_norm_epsilon": 1e-5,
    #     "lstm_dropout": 0.3,
    #     "joiner_activation": "tanh",
    # },

    # # Default
    # "rnnt_config": {
    #     "input_dim": 80,
    #     "encoding_dim": 1024,
    #     "time_reduction_stride": 4,
    #     "conformer_input_dim": 256,
    #     "conformer_ffn_dim": 1024,
    #     "conformer_num_layers": 16,
    #     "conformer_num_heads": 4,
    #     "conformer_depthwise_conv_kernel_size": 31,
    #     "conformer_dropout": 0.1,
    #     "num_symbols": 1024,
    #     "symbol_embedding_dim": 256,
    #     "num_lstm_layers": 2,
    #     "lstm_hidden_dim": 512,
    #     "lstm_layer_norm": True,
    #     "lstm_layer_norm_epsilon": 1e-5,
    #     "lstm_dropout": 0.3,
    #     "joiner_activation": "tanh",
    # },

    # "rnnt_config": {
    #     "input_dim": 80,
    #     "encoding_dim": 512,
    #     "subsampling_type": "conv",  # splice, conv
    #     "time_reduction_stride": 2,
    #     "conformer_input_dim": 512,
    #     "conformer_ffn_dim": 2048,
    #     "conformer_num_layers": 4,
    #     "conformer_num_heads": 8,
    #     "conformer_depthwise_conv_kernel_size": 31,
    #     "conformer_dropout": 0.2,
    #     "num_symbols": 181,
    #     "symbol_embedding_dim": 1024,
    #     "num_lstm_layers": 2,
    #     "lstm_hidden_dim": 512,
    #     "lstm_layer_norm": True,
    #     "lstm_layer_norm_epsilon": 1e-5,
    #     "lstm_dropout": 0.3,
    #     "joiner_activation": "tanh",
    # },

    # "tdnn_blstm_config": None,
    "tdnn_blstm_config": {
        "input_dim": 80,
        # "num_symbols": 29,    # char
        "num_symbols": 94,    # phoneme
        # "num_symbols": 1024,  # bpe
        "hidden_dim": 640,
        # "hidden_dim": 1024,
        "drop_out": 0.1,
        "tdnn_blstm_spec": [
            # ["tdnn", 5, 2],
            # ["tdnn", 3, 1],  # 2
            # ["blstm"],
            # ["tdnn", 3, 1],
            # ["blstm"],
            # ["tdnn", 3, 1],
            # ["blstm"],
            # # ["blstm"],
            # # ["blstm"],
            # ["ffn", 2],

            # ["blstm"],
            # ["blstm"],
            # ["blstm"],

            # # ["tdnn", 3, 2],
            # # ["tdnn", 3, 2],
            # ('tdnn', kernel_size, stride, dilation) 
            ["tdnn", 5, 2, 1],  # 5/9
            ["tdnn", 3, 1, 1],  # 3
            ["tdnn", 3, 1, 1],
            ["ffn", 5],

            # # stride=1 for 533
            # # ('tdnn', kernel_size, stride, dilation) 
            # ["tdnn", 5, 1, 1],
            # ["tdnn", 3, 1, 2],
            # ["tdnn", 3, 1, 2],
            # ["ffn", 5],

            # # stride=1
            # # ('tdnn', kernel_size, stride, dilation) 
            # ["tdnn", 9, 1, 1],
            # ["tdnn", 3, 1, 1],
            # ["tdnn", 3, 1, 1],
            # ["ffn", 5],
        ]
    },

    "subsampling_factor": 2,

    # training:
    "training_config": {
        "seed": 1,
        "save_top_k": 20,
        "checkpoint_path": None,
        "exp_dir": "./exp",
        "nodes": 4,
        "gpus": 8,
        "epochs": 200,
        "gradient_clip_val": 10.0,
        "full_libri": True,
    },

    "optim_config": {
        # "lr_scheduler": "warmup",
        "lr_scheduler": "step",
        # "lr_scheduler": "simple",
        "lr": 8e-4,
        "warmup_steps": 40,
        "force_anneal_step": 120,
        "anneal_factor": 0.96,        
        "batch_size": None,
        "max_tokens": 2000,
        # "max_tokens": 1000,   # conformer stride=2
        # "max_tokens": 550,  # for stride=1, and set `accumulate_grad_batches=3` in train.py
        "train_num_buckets": 50,
        "reduction": "sum",
        "weight_decay": 1e-6,
    },

    # Xiaohui's:
    "specaug_conf": {
        "new_spec_aug_api": False,
        "n_time_masks": 0,
        "time_mask_param": 100,
        "p": 0.2,
        "n_freq_masks": 0,
        "freq_mask_param": 27,
        "iid_masks": True,
        "zero_masking": True,
    },

    # # Default:
    # "specaug_conf": {
    #     "new_spec_aug_api": False,
    #     "n_time_masks": 2,
    #     "time_mask_param": 100,
    #     "p": 0.2,
    #     "n_freq_masks": 2,
    #     "freq_mask_param": 27,
    #     "iid_masks": True,
    #     "zero_masking": True,
    # },

    # # Espnet's:  f30,2; t40, 2
    # "specaug_conf": {
    #     "new_spec_aug_api": False,
    #     "n_time_masks": 2,
    #     "time_mask_param": 40,
    #     "p": 0.2,
    #     "n_freq_masks": 2,
    #     "freq_mask_param": 30,
    #     "iid_masks": True,
    #     "zero_masking": True,
    # },

    "speed_perturbation": True,
    "musan_noise": False,
    # "musan_noise": {
    #     "subsets": ["noise", "music"],  # "music", "speech"
    #     # "snr": [15, 30],
    #     "snr": [15, 25],
    #     "p": 0.5,
    # },

    # # inference:
    # "inference_config": {
    #     "temperature": 1.0,
    #     "step_max_tokens": 100,
    #     "beam_width": 20,  # espnet: 10  # https://github.com/espnet/espnet/blob/master/egs2/librispeech/asr1/conf/tuning/transducer/decode.yaml
    # },

    "updated": False,

    "topo_type": "ctc",
    "model_unit": 'phoneme',  # "phoneme", "char", "bpe"
    "k2_loss": True,
    "sil_penalty_inter_word": 0.0,  # The larger, the more penalty for the <sil> arcs
    "sil_penalty_intra_word": 0.0,  # 10000000000
    "self_loop_bonus": 0.0,

    "prior_scaling_factor": 0.0,
    "frame_dropout": 0.0,

    "ctc_beam_size": 10,  # default: 10
}


def update_missing_fields(d, d_ref):
    updated_or_not = False
    for k, v in d_ref.items():
        if k not in d:
            if k == "tdnn_blstm_config" and "rnnt_config" in d:
                continue
            if k == "rnnt_config" and "tdnn_blstm_config" in d:
                continue
            d[k] = d_ref[k]
            updated_or_not = True
        elif type(v) is dict:
            if type(d[k]) is bool:
                continue
            _dk, _updated_or_not = update_missing_fields(d[k], d_ref[k])
            d[k] = _dk
            updated_or_not = updated_or_not or _updated_or_not
    return d, updated_or_not


def sanity_check(config):
    if "rnnt_config" in config and config["rnnt_config"] is not None:
        assert config["subsampling_factor"] == config["rnnt_config"]["time_reduction_stride"]
        assert config["spm_vocab_size"] == config["rnnt_config"]["num_symbols"], f'{config["spm_vocab_size"]} vs {config["rnnt_config"]["num_symbols"]}'
        assert config["optim_config"]["lr_scheduler"] != "simple"
    elif "tdnn_blstm_config" in config and config["tdnn_blstm_config"] is not None:
        subsampling_factor = 1
        for x in config["tdnn_blstm_config"]["tdnn_blstm_spec"]:
            if x[0] == "tdnn":
                subsampling_factor *= x[2]
        assert config["subsampling_factor"] == subsampling_factor
        assert config["spm_vocab_size"] == config["tdnn_blstm_config"]["num_symbols"], f'{config["spm_vocab_size"]} vs. {config["tdnn_blstm_config"]["num_symbols"]}'


# https://python.land/data-processing/python-yaml
def load_config(config_file):
    if config_file is None or not pathlib.Path(config_file).exists():
        logging.info(f"No config_file found. Using default config.")
        sanity_check(default_config)
        return default_config
    
    logging.info(f"Loading config file from: {config_file}")
    with open(config_file, 'r') as fin:
        config = yaml.safe_load(fin)
    
    _, updated_or_not = update_missing_fields(config, default_config)
    config["updated"] = updated_or_not
    logging.info(f"The config file has been update or not? {updated_or_not}")
    sanity_check(config)
    return config


def my_stringify_dict(d):
    str_d = dict()

    for k, v in d.items():
        if isinstance(v, pathlib.Path): # type(v) is pathlib.Path:
            str_d[k] = str(v.absolute())
        elif type(v) is dict:
            str_d[k] = my_stringify_dict(v)
        else:
            str_d[k] = v
    return str_d


def save_config(config, config_file, forced_save=False):
    _str_config = my_stringify_dict(config)

    pp = pathlib.Path(config_file)
    if not pp.exists() or config["updated"] or forced_save:
        if pp.exists():
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            new_path = pp.with_suffix(f".{timestamp}.yaml")
            pp.rename(new_path)
            logging.info(f"Existing config file has been updated. Previous one saved to: {new_path}")

        with open(config_file, 'w') as fout:
            yaml.dump(_str_config, fout)
    else:
        logging.info(f"Skipped saving config file. Existed: {config_file}")


def update_config(config, args):
    if args.checkpoint_path is not None:
        config["training_config"]["checkpoint_path"] = args.checkpoint_path
    if args.exp_dir is not None:
        config["training_config"]["exp_dir"] = args.exp_dir
    if args.nodes is not None:
        config["training_config"]["nodes"] = args.nodes
    if args.gpus is not None:
        config["training_config"]["gpus"] = args.gpus
    if args.epochs is not None:
        config["training_config"]["epochs"] = args.epochs
    return config


if __name__ == '__main__':
    # exp_dir = "/checkpoints/lisun/ruizhe/audio/examples/asr/librispeech_conformer_ctc/experiments/exp_0828_dnn533_ctc_0.0_0.0_bpe"
    # exp_dir = "/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc/experiments/exp_20230803_12"
    # exp_dir = "/exp/rhuang/meta/audio/examples/asr/librispeech_conformer_ctc/exp_0901/conformer_ctc_0.1_0.5_bpe_p0.0"
    exp_dir = "/exp/rhuang/meta/audio/examples/asr/librispeech_conformer_ctc2/experiments/dnn_phone_k2_p0.0_inter0.0_intra0.0_loop0.0_stride2"
    # exp_dir = "/exp/rhuang/meta/audio_ruizhe/zhaoheng/exp_0906_5/dnn_phone_k2_p0.0_inter0.0_intra0.0_loop0.0_stride2"
    config = load_config(None)
    config["training_config"]["exp_dir"] = exp_dir
    import os
    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{exp_dir}/ali", exist_ok=True)
    save_config(config, f"{exp_dir}/train_config.yaml", forced_save=True)
    print(f"Done: " + f"{exp_dir}/train_config.yaml")
