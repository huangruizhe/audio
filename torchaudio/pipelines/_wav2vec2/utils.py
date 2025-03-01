from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor

from torchaudio._internal import load_state_dict_from_url
from torchaudio.models import wav2vec2_model, Wav2Vec2Model, wavlm_model


def _get_model(type_, params):
    factories = {
        "Wav2Vec2": wav2vec2_model,
        "WavLM": wavlm_model,
    }
    if type_ not in factories:
        raise ValueError(f"Supported model types are {tuple(factories.keys())}. Found: {type_}")
    factory = factories[type_]
    return factory(**params)


class _Wav2Vec2Model(nn.Module):
    """Wrapper class for :py:class:`~torchaudio.models.Wav2Vec2Model`.

    This is used for layer normalization at the input
    """

    def __init__(self, model: Wav2Vec2Model):
        super().__init__()
        self.model = model

    def forward(self, waveforms: Tensor, lengths: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        waveforms = nn.functional.layer_norm(waveforms, waveforms.shape)
        return self.model(waveforms, lengths)

    @torch.jit.export
    def extract_features(
        self,
        waveforms: Tensor,
        lengths: Optional[Tensor] = None,
        num_layers: Optional[int] = None,
    ) -> Tuple[List[Tensor], Optional[Tensor]]:
        waveforms = nn.functional.layer_norm(waveforms, waveforms.shape)
        return self.model.extract_features(waveforms, lengths, num_layers)


def _apply_input_layer_norm(module):
    """Add extra layer_norm to the model"""
    return _Wav2Vec2Model(module)


def _remove_aux_axes(state_dict, axes):
    # Remove the seemingly unnecessary axis
    # For ASR task, the pretrained weights originated from fairseq has unrelated dimensions at index 1, 2, 3
    # It's originated from the Dictionary implementation of fairseq, which was intended for NLP tasks,
    # but not used during the ASR training.
    # https://github.com/pytorch/fairseq/blob/c5ff181125c7e6126b49a85e5ebdd5f5b6a07914/fairseq/data/dictionary.py#L21-L37
    # https://github.com/pytorch/fairseq/blob/c5ff181125c7e6126b49a85e5ebdd5f5b6a07914/fairseq/criterions/ctc.py#L126-L129
    #
    # Also, some pretrained weights originated from voxpopuli has an extra dimensions that almost never used and
    # that resembles mistake.
    # The label `1` shows up in the training dataset of German (1 out of 16M),
    # English (1 / 28M), Spanish (1 / 9.4M), Romanian (1 / 4.7M) and Polish (6 / 5.8M)
    for key in ["aux.weight", "aux.bias"]:
        mat = state_dict[key]
        state_dict[key] = torch.stack([mat[i] for i in range(mat.size(0)) if i not in axes])


def _add_star_dim(state_dict):
    w, b = state_dict["aux.weight"], state_dict["aux.bias"]
    zeros = torch.zeros((1, w.size(1)), device=w.device, dtype=w.dtype)
    state_dict["aux.weight"] = torch.cat((zeros, w), dim=0)
    ones = torch.ones((1,), device=b.device, dtype=b.dtype)
    state_dict["aux.bias"] = torch.cat((b, ones), dim=0)


def _get_state_dict(url, dl_kwargs, remove_axes=None, add_star=False):
    if not url.startswith("https"):
        url = f"https://download.pytorch.org/torchaudio/models/{url}"
    dl_kwargs = {} if dl_kwargs is None else dl_kwargs
    state_dict = load_state_dict_from_url(url, **dl_kwargs)
    if remove_axes:
        _remove_aux_axes(state_dict, remove_axes)
    if add_star:
        _add_star_dim(state_dict)
    return state_dict


def _get_en_labels():
    return (
        "|",
        "E",
        "T",
        "A",
        "O",
        "N",
        "I",
        "H",
        "S",
        "R",
        "D",
        "L",
        "U",
        "M",
        "W",
        "C",
        "F",
        "G",
        "Y",
        "P",
        "B",
        "V",
        "K",
        "'",
        "X",
        "J",
        "Q",
        "Z",
    )


def _get_de_labels():
    return (
        "|",
        "e",
        "n",
        "i",
        "r",
        "s",
        "t",
        "a",
        "d",
        "h",
        "u",
        "l",
        "g",
        "c",
        "m",
        "o",
        "b",
        "w",
        "f",
        "k",
        "z",
        "p",
        "v",
        "ü",
        "ä",
        "ö",
        "j",
        "ß",
        "y",
        "x",
        "q",
    )


def _get_vp_en_labels():
    return (
        "|",
        "e",
        "t",
        "o",
        "i",
        "a",
        "n",
        "s",
        "r",
        "h",
        "l",
        "d",
        "c",
        "u",
        "m",
        "p",
        "f",
        "g",
        "w",
        "y",
        "b",
        "v",
        "k",
        "x",
        "j",
        "q",
        "z",
    )


def _get_es_labels():
    return (
        "|",
        "e",
        "a",
        "o",
        "s",
        "n",
        "r",
        "i",
        "l",
        "d",
        "c",
        "t",
        "u",
        "p",
        "m",
        "b",
        "q",
        "y",
        "g",
        "v",
        "h",
        "ó",
        "f",
        "í",
        "á",
        "j",
        "z",
        "ñ",
        "é",
        "x",
        "ú",
        "k",
        "w",
        "ü",
    )


def _get_fr_labels():
    return (
        "|",
        "e",
        "s",
        "n",
        "i",
        "t",
        "r",
        "a",
        "o",
        "u",
        "l",
        "d",
        "c",
        "p",
        "m",
        "é",
        "v",
        "q",
        "f",
        "g",
        "b",
        "h",
        "x",
        "à",
        "j",
        "è",
        "y",
        "ê",
        "z",
        "ô",
        "k",
        "ç",
        "œ",
        "û",
        "ù",
        "î",
        "â",
        "w",
        "ï",
        "ë",
        "ü",
        "æ",
    )


def _get_it_labels():
    return (
        "|",
        "e",
        "i",
        "a",
        "o",
        "n",
        "t",
        "r",
        "l",
        "s",
        "c",
        "d",
        "u",
        "p",
        "m",
        "g",
        "v",
        "h",
        "z",
        "f",
        "b",
        "q",
        "à",
        "è",
        "ù",
        "é",
        "ò",
        "ì",
        "k",
        "y",
        "x",
        "w",
        "j",
        "ó",
        "í",
        "ï",
    )


def _get_mms_labels():
    return (
        "a",
        "i",
        "e",
        "n",
        "o",
        "u",
        "t",
        "s",
        "r",
        "m",
        "k",
        "l",
        "d",
        "g",
        "h",
        "y",
        "b",
        "p",
        "w",
        "c",
        "v",
        "j",
        "z",
        "f",
        "'",
        "q",
        "x",
    )
