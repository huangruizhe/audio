import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Blstm_with_skip(nn.Module):
    # https://www.kaggle.com/code/khalildmk/simple-two-layer-bidirectional-lstm-with-pytorch

    def __init__(self, input_dim, hidden_dim, out_dim, skip=True, drop_out=0.1) -> None:        
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        self.linear = nn.Linear(hidden_dim * 2, out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.bnorm = nn.BatchNorm1d(num_features=out_dim, affine=False)
        if drop_out > 0:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = None

        self.skip = skip
    
    def forward(self, x, lengths) -> torch.Tensor:
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (ht, ct) = self.lstm(packed_input)
        lstm_out, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        lstm_out = self.linear(lstm_out)
        lstm_out = self.relu(lstm_out)  # This is very useful (great improvement: blstm1 vs. tdnn_blstm2) # IMPORTANT
        lstm_out = self.bnorm(lstm_out.transpose(1, 2)).transpose(1, 2)
        if self.skip:
            lstm_out = lstm_out + x  # skip connections
        
        if self.drop_out is not None:
            lstm_out = self.drop_out(lstm_out)

        # assert torch.equal(input_sizes, lengths.cpu())
        return lstm_out


class Lstm_with_skip(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, skip=True, drop_out=0.1) -> None:        
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True, 
        )
        self.bnorm = nn.BatchNorm1d(num_features=out_dim, affine=False)
        if drop_out > 0:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = None

        self.skip = skip
    
    def forward(self, x, lengths) -> torch.Tensor:
        

        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (ht, ct) = self.lstm(packed_input)
        lstm_out, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        lstm_out = self.linear(lstm_out)
        lstm_out = self.relu(lstm_out)
        lstm_out = self.bnorm(lstm_out.transpose(1, 2)).transpose(1, 2)
        if self.skip:
            lstm_out = lstm_out + x  # skip connections
        
        if self.drop_out is not None:
            lstm_out = self.drop_out(lstm_out)

        # assert torch.equal(input_sizes, lengths.cpu())
        return lstm_out


class TdnnBlstm(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim = 640, drop_out = 0.1, tdnn_blstm_spec = [],
    ) -> None:
        """
        Args:

          tdnn_blstm_spec:
            It is a list of network specifications. It can be either:
            - ('tdnn', kernel_size, stride) 
            - ('blstm')
        """
        super().__init__()

        self.tdnn_blstm_spec = tdnn_blstm_spec

        layers = nn.ModuleList([])
        layers_info = []
        for i_layer, spec in enumerate(tdnn_blstm_spec):
            if spec[0] == 'tdnn':
                ll = []
                ll.append(
                    nn.Conv1d(
                        in_channels=input_dim if len(layers) == 0 else hidden_dim,
                        out_channels=hidden_dim,
                        kernel_size=spec[1],  # 3
                        dilation=1,
                        stride=spec[2],  # 1
                        padding=int((spec[1] - 1)/2),  # 1
                    )
                )
                ll.append(nn.ReLU(inplace=True))
                ll.append(nn.BatchNorm1d(num_features=hidden_dim, affine=False))
                if drop_out > 0:
                    ll.append(nn.Dropout(drop_out))
                
                # The last dimension indicates the stride size
                # If stride > 1, then we need to recompute the lengths of input after this layer
                layers.append(nn.Sequential(*ll))
                layers_info.append(('tdnn', spec))

            elif spec[0] == 'blstm':
                layers.append(
                    Blstm_with_skip(
                        input_dim=input_dim if len(layers) == 0 else hidden_dim, 
                        hidden_dim=hidden_dim,
                        out_dim=hidden_dim,
                        skip=False if len(layers) == 0 and input_dim != hidden_dim else True,
                        drop_out=drop_out,
                    )
                )
                layers_info.append(("blstm", None))
        
        self.layers = layers
        self.layers_info = layers_info

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:
            Its shape is [N, T, C]

        Returns:
          The output tensor has shape [N, T, C]
        """
        x = input
        # import pdb; pdb.set_trace()
        for layer, (layer_type, spec) in zip(self.layers, self.layers_info):
            if layer_type == "tdnn":
                mask = (torch.arange(lengths.max(), device=x.device)[None, :] < lengths[:, None]).float()
                x = x * mask.unsqueeze(2)  # masking/padding
                x = x.permute(0, 2, 1)  # (N, T, C) ->(N, C, T)
                x = layer(x)
                x = x.permute(0, 2, 1)  # (N, C, T) ->(N, T, C)

                stride = spec[2]
                if stride > 1:
                    kernel_size = spec[1]
                    padding = int((spec[1] - 1)/2)
                    lengths = lengths + 2 * padding - 1 * (kernel_size - 1) - 1
                    lengths = lengths / stride + 1
                    lengths = torch.floor(lengths)
            elif layer_type == "blstm":
                x = layer(x, lengths)
        return x, lengths