import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.convolutions import Bottleneck, Block, Bottleblock
from ..utils.geometry import warp_features  # assumes geometry utils copied over


class SpatialGRU(nn.Module):
    """A GRU cell that takes an input tensor [BxTxCxHxW] and an optional state."""

    def __init__(self, input_size, hidden_size, gru_bias_init=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_bias_init = gru_bias_init

        self.conv_update = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_reset = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_state_tilde = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_decoder = nn.Conv2d(hidden_size, input_size, kernel_size=1, bias=False)

    def forward(self, x, state=None):
        assert len(x.size()) == 5, 'Input tensor must be BxTxCxHxW.'

        rnn_output = []
        b, timesteps, c, h, w = x.size()
        rnn_state = torch.zeros(b, self.hidden_size, h, w, device=x.device) if state is None else state
        for t in range(timesteps):
            x_t = x[:, t]

            rnn_state = self.gru_cell(x_t, rnn_state)
            rnn_output.append(self.conv_decoder(rnn_state))

        return torch.stack(rnn_output, dim=1)

    def gru_cell(self, x, state):
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = self.conv_update(x_and_state)
        reset_gate = self.conv_reset(x_and_state)
        update_gate = torch.sigmoid(update_gate + self.gru_bias_init)
        reset_gate = torch.sigmoid(reset_gate + self.gru_bias_init)

        state_tilde = self.conv_state_tilde(torch.cat([x, (1.0 - reset_gate) * state], dim=1))

        output = (1.0 - update_gate) * state + update_gate * state_tilde
        return output


class Dual_GRU(nn.Module):
    def __init__(self, in_channels, latent_dim, n_future, mixture=True, gru_bias_init=0.0):
        super().__init__()

        input_size = in_channels
        hidden_size = latent_dim

        self.n_future = n_future
        self.mixture = mixture

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_bias_init = gru_bias_init

        self.conv_update_1 = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_reset_1 = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_state_tilde_1 = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)

        self.conv_update_2 = nn.Conv2d(hidden_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_reset_2 = nn.Conv2d(hidden_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_state_tilde_2 = nn.Conv2d(hidden_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_decoder_2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)

        self.trusting_gate = nn.Sequential(
            Bottleblock(hidden_size + hidden_size, hidden_size),
            nn.Conv2d(hidden_size, 2, kernel_size=1, bias=False),
        )

    def forward(self, x, state):
        b, s, c, h, w = x.shape
        assert c == self.input_size, f'feature sizes must match, got input {c} for layer with size {self.input_size}'
        n_present = state.shape[1]

        h = state[:, 0]
        pred_state = []

        for t in range(n_present - 1):
            cur_state = state[:, t]
            h = self.gru_cell_2(cur_state, h)

        rnn_state1 = state[:, -1]
        rnn_state2 = state[:, -1]
        x = x[:, 0]

        for _ in range(self.n_future):
            rnn_state1 = self.gru_cell_1(x, rnn_state1)
            h = self.gru_cell_2(rnn_state2, h)
            rnn_state2 = self.conv_decoder_2(h)

            mix_state = torch.cat([rnn_state1, rnn_state2], dim=1)
            trust_gate = self.trusting_gate(mix_state)
            trust_gate = torch.softmax(trust_gate, dim=1)
            cur_state = rnn_state2 * trust_gate[:, 0:1] + rnn_state1 * trust_gate[:, 1:]

            pred_state.append(cur_state)

            if self.mixture:
                rnn_state1 = cur_state
                rnn_state2 = cur_state

        x = torch.stack(pred_state, dim=1)

        return x

    def gru_cell_1(self, x, state):
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = self.conv_update_1(x_and_state)
        reset_gate = self.conv_reset_1(x_and_state)
        update_gate = torch.sigmoid(update_gate + self.gru_bias_init)
        reset_gate = torch.sigmoid(reset_gate + self.gru_bias_init)

        state_tilde = self.conv_state_tilde_1(torch.cat([x, (1.0 - reset_gate) * state], dim=1))

        output = (1.0 - update_gate) * state + update_gate * state_tilde
        return output

    def gru_cell_2(self, x, state):
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = self.conv_update_2(x_and_state)
        reset_gate = self.conv_reset_2(x_and_state)
        update_gate = torch.sigmoid(update_gate + self.gru_bias_init)
        reset_gate = torch.sigmoid(reset_gate + self.gru_bias_init)

        state_tilde = self.conv_state_tilde_2(torch.cat([x, (1.0 - reset_gate) * state], dim=1))

        output = (1.0 - update_gate) * state + update_gate * state_tilde
        return output


class BiGRU(nn.Module):
    def __init__(self, in_channels, gru_bias_init=0.0):
        super().__init__()
        input_size = in_channels
        hidden_size = in_channels

        self.input_size = in_channels
        self.hidden_size = in_channels
        self.gru_bias_init = gru_bias_init

        self.conv_update_1 = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_reset_1 = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_state_tilde_1 = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_decoder_1 = Bottleblock(hidden_size, input_size)

        self.conv_update_2 = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_reset_2 = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_state_tilde_2 = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_decoder_2 = Bottleblock(hidden_size, input_size)

        self.res_blocks = nn.Sequential(
            Bottleblock(in_channels + in_channels, in_channels),
            Block(in_channels, in_channels),
            Block(in_channels, in_channels),
        )

    def forward(self, x):
        b, s, c, h, w = x.shape

        rnn_state1 = x[:, 0]
        rnn_state2 = x[:, -1]

        states_1 = []
        states_2 = []

        for t in range(s):
            x_t_1 = x[:, t]
            rnn_state1 = self.gru_cell_1(x_t_1, rnn_state1)
            states_1.append(self.conv_decoder_1(rnn_state1))

            x_t_2 = x[:, s - 1 - t]
            rnn_state2 = self.gru_cell_2(x_t_2, rnn_state2)
            states_2.append(self.conv_decoder_2(rnn_state2))

        states_1 = torch.stack(states_1, dim=1)
        states_2 = torch.stack(states_2[::-1], dim=1)

        mix_state = torch.cat([states_1, states_2], dim=2)
        mix_state = self.res_blocks(mix_state.view(b * s, *mix_state.shape[2:]))
        mix_state = mix_state.view(b, s, self.input_size, h, w)
        return mix_state

    def gru_cell_1(self, x, state):
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = self.conv_update_1(x_and_state)
        reset_gate = self.conv_reset_1(x_and_state)
        update_gate = torch.sigmoid(update_gate + self.gru_bias_init)
        reset_gate = torch.sigmoid(reset_gate + self.gru_bias_init)

        state_tilde = self.conv_state_tilde_1(torch.cat([x, (1.0 - reset_gate) * state], dim=1))

        output = (1.0 - update_gate) * state + update_gate * state_tilde
        return output

    def gru_cell_2(self, x, state):
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = self.conv_update_2(x_and_state)
        reset_gate = self.conv_reset_2(x_and_state)
        update_gate = torch.sigmoid(update_gate + self.gru_bias_init)
        reset_gate = torch.sigmoid(reset_gate + self.gru_bias_init)

        state_tilde = self.conv_state_tilde_2(torch.cat([x, (1.0 - reset_gate) * state], dim=1))

        output = (1.0 - update_gate) * state + update_gate * state_tilde
        return output


__all__ = ['SpatialGRU', 'Dual_GRU', 'BiGRU']
