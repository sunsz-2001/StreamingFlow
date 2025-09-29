"""
Neural Negative Feedback ODE with Bayesian jumps.

This is the core ODE system that handles continuous-time dynamics with discrete observation updates.
"""

import torch
import numpy as np
from torch import nn

from ..utils.models import SmallEncoder, SmallDecoder, ConvNet, rsample_normal
from ..cells.gru_ode_cells import DualGRUODECell
from ..cells.observation_cell import GRUObservationCell
from ..utils.encoders import init_weights


class NNFOwithBayesianJumps(torch.nn.Module):
    """带贝叶斯跳跃的神经负反馈 ODE 核心模块。

    这是 StreamingFlow 中实现"ODE 驱动的变分循环神经网络"的关键组件。
    模块把 BEV 特征图看成随时间连续变化的隐状态：当没有新观测时，
    用神经 ODE（`DualGRUODECell`）预测状态的漂移；一旦有新的相机/激光
    观测到达，则通过 `GRUObservationCell` 进行一次"贝叶斯跳跃"，快速
    修正状态。这样能够在多传感器异步、时间间隔不均匀的情况下，依然
    预测未来场景。

    初学者需要把握的要点：

    * "连续时间"意味着我们耦合细粒度的时间步长 `delta_t`，而不仅仅是帧序列。
    * ODE 部分负责无观测时的平滑演化，跳跃模块负责有观测时的快速融合。
    * 由于 BEV 特征维度大，先用 `SmallEncoder` 映射到尺寸更小的潜空间，
      计算稳定且高效，再通过 `SmallDecoder` 解回原始空间。
    """

    def __init__(self, input_size, hidden_size, cfg, bias=True, logvar=True, mixing=1,
                 solver="euler", min_log_sigma=-5.0, max_log_sigma=5.0, impute=False):
        """初始化子模块。

        Args:
            input_size (int): 进入 ODE 的潜空间通道数。
            hidden_size (int): GRU ODE 与观测更新维护的隐状态通道数。
            cfg (CfgNode): 训练配置，控制是否插值、使用何种求解器等。
            bias (bool): 是否在 GRU 卷积中使用偏置。
            logvar (bool): 是否输出 log-variance 以实现变分采样。
            mixing (int): 融合 ODE 损失与观测损失的权重。
            solver (str): 积分方法，支持 'euler' 与 'midpoint'。
            min_log_sigma/max_log_sigma (float): 随机采样时的方差截断范围。
            impute (bool): 为 True 时，即使没有观测也把当前预测参数回填给 ODE。
        """
        super().__init__()

        self.impute = cfg.MODEL.IMPUTE
        self.cfg = cfg

        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma
        self.p_model = ConvNet(hidden_size, hidden_size * 2)

        self.gru_c = DualGRUODECell(input_size, hidden_size, bias=bias)
        self.gru_obs = GRUObservationCell(input_size, hidden_size,
                                          min_log_sigma=self.min_log_sigma,
                                          max_log_sigma=self.max_log_sigma, bias=bias)

        self.skipco = self.cfg.MODEL.SMALL_ENCODER.SKIPCO
        self.srvp_encoder = SmallEncoder(self.cfg.MODEL.ENCODER.OUT_CHANNELS,
                                         self.cfg.MODEL.ENCODER.OUT_CHANNELS,
                                         self.cfg.MODEL.SMALL_ENCODER.FILTER_SIZE)
        self.srvp_decoder = SmallDecoder(self.cfg.MODEL.ENCODER.OUT_CHANNELS,
                                         self.cfg.MODEL.ENCODER.OUT_CHANNELS,
                                         self.cfg.MODEL.SMALL_ENCODER.FILTER_SIZE,
                                         self.cfg.MODEL.SMALL_ENCODER.SKIPCO)

        self.solver = cfg.MODEL.SOLVER
        self.use_variable_ode_step = cfg.MODEL.FUTURE_PRED.USE_VARIABLE_ODE_STEP
        assert self.solver in ["euler", "midpoint"], "Solver must be either 'euler' or 'midpoint'."

        self.input_size = input_size
        self.logvar = logvar
        self.mixing = mixing  # mixing hyperparameter for loss_1 and loss_2 aggregation.

        self.apply(init_weights)

    def srvp_decode(self, x, skip=None):
        """将 SRVP 潜空间特征解码回 BEV 特征空间。

        Args:
            x (Tensor): 形状 ``[B, T, C, H, W]`` 的潜空间序列，每一帧对应 ODE 输出。
            skip (List[Tensor] | None): 可选的跳连信息，来自编码器，用于恢复空间细节。

        Returns:
            Tensor: 解码后的 BEV 特征，保持 ``[B, T, C, H, W]`` 形状。
        """
        b, t, c, h, w = x.shape
        _x = x.reshape(b * t, c, h, w)
        if skip:
            skip = [s.unsqueeze(1).expand(b, t, *s.shape[1:]) for s in skip]
            skip = [s.reshape(t * b, *s.shape[2:]) for s in skip]
        x_out = self.srvp_decoder(_x, skip=skip)
        return x_out.view(b, t, *x_out.shape[1:])

    def srvp_encode(self, x):
        """把 BEV 特征编码到 SRVP 潜空间，方便 ODE 处理。

        编码器还会在训练时随机选帧保存跳连信息，以便解码器利用。

        Args:
            x (Tensor): 输入形状 ``[B, T, C, H, W]``。

        Returns:
            Tuple[Tensor, Optional[List[Tensor]]]: 编码后的潜空间序列，以及可选的跳连。
        """
        b, t, c, h, w = x.shape
        _x = x.view(b * t, c, h, w)
        hx, skips = self.srvp_encoder(_x, return_skip=True)
        hx = hx.view(b, t, *hx.shape[1:])
        if self.skipco:
            if self.training:
                # When training, take a random frame to compute the skip connections
                tx = torch.randint(t, size=(b,)).to(hx.device)
                index = torch.arange(b).to(hx.device)
                skips = [s.view(b, t, *s.shape[1:])[index, tx] for s in skips]
            else:
                # When testing, choose the last frame
                skips = [s.view(b, t, *s.shape[1:])[:, -1] for s in skips]
        else:
            skips = None
        return hx, skips

    def ode_step(self, state, input, delta_t, current_time):
        """在 ``delta_t`` 时间间隔内积分一次 ODE。

        Args:
            state (Tensor): 当前隐状态，形状 ``[B*T, C, H, W]``。
            input (Tensor): 当前作为驱动的潜空间输入，若 `impute=False`，无观测时置零。
            delta_t (float): 时间步长。
            current_time (float): 步进前的仿真时间。

        Returns:
            Tuple[Tensor, Tensor, float, Tensor, Tensor]: 更新后的状态、输入、时间
            以及预留的调试张量（目前为空）。
        """
        eval_times = torch.tensor([0], device=state.device, dtype=torch.float64)
        eval_ps = torch.tensor([0], device=state.device, dtype=torch.float32)

        if self.impute is False:
            input = torch.zeros_like(input)

        if self.solver == "euler":
            state = state + delta_t * self.gru_c(input, state)
            # Keep input unchanged for consistent dimensions

        elif self.solver == "midpoint":
            k = state + delta_t / 2 * self.gru_c(input, state)
            pk = self.infer_state(k)[0]

            state = state + delta_t * self.gru_c(pk, k)
            # Keep input unchanged for consistent dimensions

        current_time += delta_t
        return state, input, current_time, eval_times, eval_ps

    def infer_state(self, x, deterministic=False):
        """从编码结果中采样初始隐状态。

        Args:
            x (Tensor): 形状 ``[B, T, C, H, W]`` 的潜空间特征。
            deterministic (bool): True 时取均值，False 时使用重参数化采样。

        Returns:
            Tuple[Tensor, Tensor]: 采样后的隐状态 ``y_0``，以及均值/方差参数。
        """
        q_y0_params = self.p_model(x)
        y_0 = rsample_normal(q_y0_params, max_log_sigma=self.max_log_sigma,
                             min_log_sigma=self.min_log_sigma)
        return y_0, q_y0_params

    def forward(self, times, input, obs, delta_t, T, return_path=True):
        """完整执行 ODE 动态 + 观测跳跃，并输出未来潜空间序列。

        Args:
            times (Tensor): 观测时间戳，升序排列。
            input (Tensor): 当前时间点的潜空间输入，形状 ``[B, 1, C, H, W]``。
            obs (Tensor): 时间轴上的观测序列，形状 ``[B, K, C, H, W]``。
            delta_t (float): 基础积分步长。
            T (Tensor): 需要预测的未来时间戳列表。
            return_path (bool): 为兼容旧代码而保留，True 时记录中间状态。

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                * state: 最终积分完的隐状态。
                * loss: 额外损失（当前为 0，占位）。
                * x: 在每个目标时间点对应的潜空间输出，形状 ``[B, len(T), C, H, W]``。
        """
        hx_obs, _ = self.srvp_encode(obs)
        input_encoded, _ = self.srvp_encode(input)
        bs, seq, c, h, w = input_encoded.shape  # [1, 1, 64, 200, 200]
        input_processed = input_encoded.view(bs * seq, c, h, w)

        state = torch.zeros_like(input_processed)  # constant init of temporal state
        current_time = times.min().item()

        path_t = []
        path_h = []

        eval_times_total = torch.tensor([], dtype=torch.float64, device=state.device)
        eval_vals_total = torch.tensor([], dtype=torch.float32, device=state.device)

        # Iterate through every observation timestamp and update the latent
        # state accordingly.  Between observations we integrate the ODE forward
        # using small steps of size delta_t (or a bigger step when the solver
        # allows it and variable step mode is enabled).
        for i, obs_time in enumerate(times):
            ## Propagation of the ODE until next observation
            while current_time <= (obs_time - delta_t):  # 0.0001 delta_t used for numerical consistency.

                if self.solver == "dopri5":
                    state, input_processed, current_time, eval_times, eval_ps = self.ode_step(
                        state, input_processed, obs_time - current_time, current_time)
                else:
                    if self.use_variable_ode_step:
                        state, input_processed, current_time, eval_times, eval_ps = self.ode_step(
                            state, input_processed, obs_time - current_time, current_time)
                    else:
                        state, input_processed, current_time, eval_times, eval_ps = self.ode_step(
                            state, input_processed, delta_t, current_time)
                eval_times_total = torch.cat((eval_times_total, eval_times))
                eval_vals_total = torch.cat((eval_vals_total, eval_ps))
                if isinstance(current_time, torch.Tensor):
                    current_time = current_time.item()

            ## Reached an observation
            X_obs = hx_obs[:, i, :, :, :]

            # Using GRUObservationCell to update state. Also updating input and loss
            state, _ = self.gru_obs(state, input_processed, X_obs)

            input_processed = self.infer_state(state)[0]

            if return_path:
                path_t.append(obs_time.item())
                path_h.append(state)

        # Having processed all observations, integrate the model further until
        # we have reached every requested prediction timestamp in T.
        for predict_time in T:
            while current_time < predict_time:
                if self.solver == "dopri5":
                    state, input_processed, current_time, eval_times, eval_ps = self.ode_step(
                        state, input_processed, predict_time - current_time, current_time)
                else:
                    if self.use_variable_ode_step:
                        state, input_processed, current_time, eval_times, eval_ps = self.ode_step(
                            state, input_processed, predict_time - current_time, current_time)
                    else:
                        state, input_processed, current_time, eval_times, eval_ps = self.ode_step(
                            state, input_processed, delta_t, current_time)
                eval_times_total = torch.cat((eval_times_total, eval_times))
                eval_vals_total = torch.cat((eval_vals_total, eval_ps))
                if isinstance(current_time, torch.Tensor):
                    current_time = current_time.item()
                # Storing the predictions
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
