"""Standard GRU cell implementations used by the ODE modules."""

import torch
import torch.nn as nn

from ..utils.convolutions import ConvBlock, Bottleblock


class SpatialGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, gru_bias_init=0.0, norm='bn', activation='relu', bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.gru_bias_init = gru_bias_init

        self.conv_update = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_reset = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)

        self.conv_state_tilde = ConvBlock(
            input_size + hidden_size,
            hidden_size,
            kernel_size=3,
            bias=False,
            norm=norm,
            activation=activation,
        )

    def forward(self, x, state):
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = self.conv_update(x_and_state)
        reset_gate = self.conv_reset(x_and_state)
        update_gate = torch.sigmoid(update_gate + self.gru_bias_init)
        reset_gate = torch.sigmoid(reset_gate + self.gru_bias_init)

        state_tilde = self.conv_state_tilde(torch.cat([x, (1.0 - reset_gate) * state], dim=1))

        output = (1.0 - update_gate) * state + update_gate * state_tilde
        return output


class DualGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, gru_bias_init=0.0, norm='bn', activation='relu', bias=True):
        super().__init__()
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
        if len(x.shape) == 4:
            x = x.unsqueeze(0)
            state = state.unsqueeze(0)
        b, s, c, h, w = x.shape
        assert c == self.input_size, f'feature sizes must match, got input {c} for layer with size {self.input_size}'
        n_present = state.shape[1]

        h = state[:, 0]

        for t in range(n_present - 1):
            cur_state = state[:, t]
            h = self.gru_cell_2(cur_state, h)

        rnn_state1 = state[:, -1]
        rnn_state2 = state[:, -1]
        x = x[:, 0]

        rnn_state1 = self.gru_cell_1(x, rnn_state1)
        h = self.gru_cell_2(rnn_state2, h)
        rnn_state2 = self.conv_decoder_2(h)

        mix_state = torch.cat([rnn_state1, rnn_state2], dim=1)
        trust_gate = self.trusting_gate(mix_state)
        trust_gate = torch.softmax(trust_gate, dim=1)
        cur_state = rnn_state2 * trust_gate[:, 0:1] + rnn_state1 * trust_gate[:, 1:]

        return cur_state

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


__all__ = ['SpatialGRUCell', 'DualGRUCell']
