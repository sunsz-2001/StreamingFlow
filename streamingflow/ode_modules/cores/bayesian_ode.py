"""带贝叶斯跳变的神经负反馈 ODE 模型实现。"""

import numpy as np
import torch
from torch import nn

from ..cells.gru_ode_cells import DualGRUODECell
from ..cells.observation_cell import GRUObservationCell
from ..utils.encoders import init_weights
from ..utils.models import SmallEncoder, SmallDecoder, ConvNet, rsample_normal


class NNFOwithBayesianJumps(nn.Module):
    """StreamingFlow 中 NNFOwithBayesianJumps 的实现。

    模块在两个阶段之间循环：
      1. **预测阶段**：通过 ODE 形式的 GRU (`DualGRUODECell`) 在连续时间里积分
         隐状态。
      2. **观测修正（跳变）阶段**：当到达新的观测时间戳时，使用
         `GRUObservationCell` 将隐状态与观测融合。该过程依赖
         `ConvNet` 学到的后验分布，对隐特征重新采样，具有随机性。

    连续隐状态定义在 SRVP 特征空间（Small Reversible PatchNet）。`SmallEncoder`
    / `SmallDecoder` 负责在 BEV 特征张量与该隐空间之间转换。详细流程见
    `forward`。
    """

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

        # 连续时间动态（预测阶段）：类似 GRU，读取隐式输入，输出 d(state)/dt。
        # 在数值积分 `ode_step` 内被调用。
        self.gru_c = DualGRUODECell(input_size, hidden_size, bias=bias)
        # 离散观测校正（更新阶段）：当抵达观测时间戳时被调用，使隐状态“跳变”到观测流形附近。
        self.gru_obs = GRUObservationCell(
            input_size,
            hidden_size,
            min_log_sigma=self.min_log_sigma,
            max_log_sigma=self.max_log_sigma,
            bias=bias,
        )

        self.skipco = self.cfg.MODEL.SMALL_ENCODER.SKIPCO
        # SRVP 编码器/解码器负责在 BEV 特征与 ODE 运算的隐空间之间映射，使相机和
        # LiDAR 特征能共享同一通道数。
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
        """将 SRVP 隐空间特征还原为 BEV 特征张量。"""
        b, t, c, h, w = x.shape
        _x = x.reshape(b * t, c, h, w)
        if skip:
            skip = [s.unsqueeze(1).expand(b, t, *s.shape[1:]) for s in skip]
            skip = [s.reshape(t * b, *s.shape[2:]) for s in skip]
        x_out = self.srvp_decoder(_x, skip=skip)
        return x_out.view(b, t, *x_out.shape[1:])

    def srvp_encode(self, x):
        """把 BEV 特征体编码到 SRVP 隐空间，可选返回跳连。"""
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
        """执行一次数值积分（预测阶段）。

        参数
        ----
        state : torch.Tensor
            当前位置的 SRVP 隐状态。
        input : torch.Tensor
            由 `infer_state` 得到的隐式输入；若 `self.impute` 为 False，会被置零，
            表示在观测前仅靠状态自身演化。
        delta_t : float
            数值积分步长。Euler 直接前进一步；Midpoint 先走半步估计斜率，
            属于显式中点法（二阶精度）。
        current_time : float
            与当前状态对应的绝对时间。
        """
        eval_times = torch.tensor([0], device=state.device, dtype=torch.float64)
        eval_ps = torch.tensor([0], device=state.device, dtype=torch.float32)

        if self.impute is False:
            input = torch.zeros_like(input)

        if self.solver == "euler":
            # 前向 Euler：x_{t+dt} = x_t + f(x_t, input) * dt
            state = state + delta_t * self.gru_c(input, state)
            input = self.infer_state(state)[0]

        elif self.solver == "midpoint":
            # 显式中点法（两步 Runge-Kutta）。中点需要重新采样隐输入；
            # 由于 `infer_state` 含随机性，这里是近似。
            k = state + delta_t / 2 * self.gru_c(input, state)
            pk = self.infer_state(k)[0]

            state = state + delta_t * self.gru_c(pk, k)
            input = self.infer_state(state)[0]

        current_time += delta_t
        return state, input, current_time, eval_times, eval_ps

    def infer_state(self, x, deterministic=False):
        """从学习到的后验中采样隐特征。

        `ConvNet` 会输出高斯的均值与对数方差；我们使用 `rsample_normal` 做重参数化，
        使梯度可以穿过随机节点。若需要确定性流程，可设 `deterministic=True`
        （当前未用到）。
        """
        q_y0_params = self.p_model(x)
        y_0 = rsample_normal(
            q_y0_params,
            max_log_sigma=self.max_log_sigma,
            min_log_sigma=self.min_log_sigma,
        )
        return y_0, q_y0_params

    def forward(self, times, input, obs, delta_t, T, return_path=True):
        """执行完整的预测-更新循环，并在目标时间输出特征。

        参数
        ----
        times : torch.Tensor
            观测时间戳（升序）。每个时间戳都会触发一次跳变更新。
        input : torch.Tensor
            当前时刻的隐特征，用于初始化轨迹。
        obs : torch.Tensor
            与 `times` 对齐的观测特征。
        delta_t : float
            基础积分步长。
        T : torch.Tensor
            需要返回预测结果的未来时间戳，模型会积分到这些时刻。

        返回
        ----
        state : torch.Tensor
            最终隐状态。
        loss : torch.Tensor
            辅助损失占位符（目前为 0，原始 NNFO 中的 KL 项在别处近似）。
        x : torch.Tensor
            在目标时间戳下解析出来的 BEV 特征。
        """
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
            # --- 预测循环 --------------------------------------------------------
            # 持续积分直到下一次观测到来。启用 `use_variable_ode_step` 时会直接迈到
            # 观测边界，相当于假设该区间内状态演化恒定，是一种近似处理。
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

            # --- 更新 / 跳变 ----------------------------------------------------
            X_obs = hx_obs[:, i, :, :, :]

            # 将预测得到的隐状态与观测融合。观测 GRU 会返回更新后的状态及其后验分布；
            # 这里忽略损失，仅保留新的隐状态，实现“贝叶斯跳变”。
            state, _ = self.gru_obs(state, input_processed, X_obs)
            input_processed = self.infer_state(state)[0]

            if return_path:
                path_t.append(obs_time.item())
                path_h.append(state)

        for predict_time in T:
            # 对未来时间重复纯预测循环；因没有观测，故不会触发跳变更新。
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

        loss = 0  # 占位：原 NNFO 目标包含 KL 项，这里近似为 0。
        return state, loss, x
