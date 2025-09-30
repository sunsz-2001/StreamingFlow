"""Neural Negative Feedback ODE with Bayesian jumps."""

import numpy as np
import torch
from torch import nn

from ..cells.gru_ode_cells import DualGRUODECell
from ..cells.observation_cell import GRUObservationCell
from ..utils.encoders import init_weights
from ..utils.models import SmallEncoder, SmallDecoder, ConvNet, rsample_normal


class NNFOwithBayesianJumps(nn.Module):
    """StreamingFlow implementation of NNFO with Bayesian jumps."""

    def __init__(
        self,
        input_size,
        hidden_size,
        cfg,
        bias=True,
        logvar=True,
        mixing=1,
        solver="euler",
        min_log_sigma=-5.0,
        max_log_sigma=5.0,
        impute=False,
    ):
        super().__init__()

        self.impute = cfg.MODEL.IMPUTE
        self.cfg = cfg

        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma
        self.p_model = ConvNet(hidden_size, hidden_size * 2)

        self.gru_c = DualGRUODECell(input_size, hidden_size, bias=bias)
        self.gru_obs = GRUObservationCell(
            input_size,
            hidden_size,
            min_log_sigma=self.min_log_sigma,
            max_log_sigma=self.max_log_sigma,
            bias=bias,
        )

        self.skipco = self.cfg.MODEL.SMALL_ENCODER.SKIPCO
        self.srvp_encoder = SmallEncoder(
            self.cfg.MODEL.ENCODER.OUT_CHANNELS,
            self.cfg.MODEL.ENCODER.OUT_CHANNELS,
            self.cfg.MODEL.SMALL_ENCODER.FILTER_SIZE,
        )
        self.srvp_decoder = SmallDecoder(
            self.cfg.MODEL.ENCODER.OUT_CHANNELS,
            self.cfg.MODEL.ENCODER.OUT_CHANNELS,
            self.cfg.MODEL.SMALL_ENCODER.FILTER_SIZE,
            self.cfg.MODEL.SMALL_ENCODER.SKIPCO,
        )

        self.solver = cfg.MODEL.SOLVER
        self.use_variable_ode_step = cfg.MODEL.FUTURE_PRED.USE_VARIABLE_ODE_STEP
        assert self.solver in ["euler", "midpoint"], "Solver must be either 'euler' or 'midpoint'."

        self.input_size = input_size
        self.logvar = logvar
        self.mixing = mixing

        self.apply(init_weights)

    def srvp_decode(self, x, skip=None):
        b, t, c, h, w = x.shape
        _x = x.reshape(b * t, c, h, w)
        if skip:
            skip = [s.unsqueeze(1).expand(b, t, *s.shape[1:]) for s in skip]
            skip = [s.reshape(t * b, *s.shape[2:]) for s in skip]
        x_out = self.srvp_decoder(_x, skip=skip)
        return x_out.view(b, t, *x_out.shape[1:])

    def srvp_encode(self, x):
        b, t, c, h, w = x.shape
        _x = x.view(b * t, c, h, w)
        hx, skips = self.srvp_encoder(_x, return_skip=True)
        hx = hx.view(b, t, *hx.shape[1:])
        if self.skipco:
            if self.training:
                tx = torch.randint(t, size=(b,)).to(hx.device)
                index = torch.arange(b).to(hx.device)
                skips = [s.view(b, t, *s.shape[1:])[index, tx] for s in skips]
            else:
                skips = [s.view(b, t, *s.shape[1:])[:, -1] for s in skips]
        else:
            skips = None
        return hx, skips

    def ode_step(self, state, input, delta_t, current_time):
        eval_times = torch.tensor([0], device=state.device, dtype=torch.float64)
        eval_ps = torch.tensor([0], device=state.device, dtype=torch.float32)

        if self.impute is False:
            input = torch.zeros_like(input)

        if self.solver == "euler":
            state = state + delta_t * self.gru_c(input, state)
            input = self.infer_state(state)[0]

        elif self.solver == "midpoint":
            k = state + delta_t / 2 * self.gru_c(input, state)
            pk = self.infer_state(k)[0]

            state = state + delta_t * self.gru_c(pk, k)
            input = self.infer_state(state)[0]

        current_time += delta_t
        return state, input, current_time, eval_times, eval_ps

    def infer_state(self, x, deterministic=False):
        q_y0_params = self.p_model(x)
        y_0 = rsample_normal(
            q_y0_params,
            max_log_sigma=self.max_log_sigma,
            min_log_sigma=self.min_log_sigma,
        )
        return y_0, q_y0_params

    def forward(self, times, input, obs, delta_t, T, return_path=True):
        hx_obs, _ = self.srvp_encode(obs)
        input_encoded, _ = self.srvp_encode(input)
        bs, seq, c, h, w = input_encoded.shape
        input_processed = input_encoded.view(bs * seq, c, h, w)

        state = torch.zeros_like(input_processed)
        current_time = times.min().item()

        path_t = []
        path_h = []

        eval_times_total = torch.tensor([], dtype=torch.float64, device=state.device)
        eval_vals_total = torch.tensor([], dtype=torch.float32, device=state.device)

        for i, obs_time in enumerate(times):
            while current_time <= (obs_time - delta_t):
                if self.solver == "dopri5":
                    state, input_processed, current_time, eval_times, eval_ps = self.ode_step(
                        state, input_processed, obs_time - current_time, current_time
                    )
                else:
                    if self.use_variable_ode_step:
                        state, input_processed, current_time, eval_times, eval_ps = self.ode_step(
                            state, input_processed, obs_time - current_time, current_time
                        )
                    else:
                        state, input_processed, current_time, eval_times, eval_ps = self.ode_step(
                            state, input_processed, delta_t, current_time
                        )
                eval_times_total = torch.cat((eval_times_total, eval_times))
                eval_vals_total = torch.cat((eval_vals_total, eval_ps))
                if isinstance(current_time, torch.Tensor):
                    current_time = current_time.item()

            X_obs = hx_obs[:, i, :, :, :]

            state, _ = self.gru_obs(state, input_processed, X_obs)
            input_processed = self.infer_state(state)[0]

            if return_path:
                path_t.append(obs_time.item())
                path_h.append(state)

        for predict_time in T:
            while current_time < predict_time:
                if self.solver == "dopri5":
                    state, input_processed, current_time, eval_times, eval_ps = self.ode_step(
                        state, input_processed, predict_time - current_time, current_time
                    )
                else:
                    if self.use_variable_ode_step:
                        state, input_processed, current_time, eval_times, eval_ps = self.ode_step(
                            state, input_processed, predict_time - current_time, current_time
                        )
                    else:
                        state, input_processed, current_time, eval_times, eval_ps = self.ode_step(
                            state, input_processed, delta_t, current_time
                        )
                eval_times_total = torch.cat((eval_times_total, eval_times))
                eval_vals_total = torch.cat((eval_vals_total, eval_ps))
                if isinstance(current_time, torch.Tensor):
                    current_time = current_time.item()
                if current_time > predict_time - 0.5 * delta_t and current_time < predict_time + 0.5 * delta_t:
                    path_t.append(current_time)
                    path_h.append(state)

        x = []
        path_t = np.array(path_t)

        for time_stamp in T:
            if isinstance(time_stamp, torch.Tensor):
                time_stamp = time_stamp.item()
            A = np.where(path_t > time_stamp - 0.5 * delta_t)[0]
            B = np.where(path_t < time_stamp + 0.5 * delta_t)[0]

            if np.any(np.in1d(A, B)):
                idx = np.max(A[np.in1d(A, B)])
            else:
                idx = np.argmin(np.abs(path_t - time_stamp))
            x.append(path_h[idx])

        x = torch.stack(x, dim=1)
        x = self.srvp_decode(x)

        loss = 0
        return state, loss, x
