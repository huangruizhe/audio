from typing import Any
import yaml
import pathlib
import logging
import datetime

logging.getLogger("lightning.pytorch").setLevel(logging.INFO)

class config_dict(dict):
    # https://stackoverflow.com/questions/2328235/pythonextend-the-dict-class
    def __getitem__(self, __key: Any) -> Any:
        return super().__getitem__(__key)
    

default_config = {
    # model:
    "spm_vocab_size": 1023,

    # Xiaohui's
    "rnnt_config": {
        "input_dim": 80,
        "encoding_dim": 512,
        "subsampling_type": "splice",  # splice, conv
        "time_reduction_stride": 4,
        "conformer_input_dim": 512,
        "conformer_ffn_dim": 2048,
        "conformer_num_layers": 12,
        "conformer_num_heads": 8,
        "conformer_depthwise_conv_kernel_size": 31,
        "conformer_dropout": 0.1,
        "num_symbols": 1024,
        "symbol_embedding_dim": 1024,
        "num_lstm_layers": 2,
        "lstm_hidden_dim": 512,
        "lstm_layer_norm": True,
        "lstm_layer_norm_epsilon": 1e-5,
        "lstm_dropout": 0.3,
        "joiner_activation": "tanh",
    },

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
        "warmup_steps": 40,
        "force_anneal_step": 120,
        "anneal_factor": 0.96,
        "lr": 8e-4,
        "batch_size": None,
        "max_tokens": 1200,
        "train_num_buckets": 50,
        "reduction": "sum",
        "weight_decay": 2e-6,
    },

    # Xiaohui's:
    "specaug_conf": {
        "new_spec_aug_api": False,
        "n_time_masks": 10,
        "time_mask_param": 100,
        "p": 0.2,
        "n_freq_masks": 2,
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

    "speed_perturbation": False,
    "musan_noise": False,

    # # inference:
    # "inference_config": {
    #     "temperature": 1.0,
    #     "step_max_tokens": 100,
    #     "beam_width": 20,  # espnet: 10  # https://github.com/espnet/espnet/blob/master/egs2/librispeech/asr1/conf/tuning/transducer/decode.yaml
    # },

    "updated": False,
}


def update_missing_fields(d, d_ref):
    updated_or_not = False
    for k, v in d_ref.items():
        if k not in d:
            d[k] = d_ref[k]
            updated_or_not = True
        elif type(v) is dict:
            _dk, _updated_or_not = update_missing_fields(d[k], d_ref[k])
            d[k] = _dk
            updated_or_not = updated_or_not or _updated_or_not
    return d, updated_or_not


# https://python.land/data-processing/python-yaml
def load_config(config_file):
    if config_file is None or not pathlib.Path(config_file).exists():
        logging.info(f"No config_file found. Using default config.")
        return default_config
    
    with open(config_file, 'r') as fin:
        config = yaml.safe_load(fin)
    
    _, updated_or_not = update_missing_fields(config, default_config)
    config["updated"] = updated_or_not
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


def save_config(config, config_file):
    _str_config = my_stringify_dict(config)

    pp = pathlib.Path(config_file)
    if not pp.exists() or config["updated"]:
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
