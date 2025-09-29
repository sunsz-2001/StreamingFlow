"""
ODE模块的最小配置示例

这个文件展示了使用ODE模块所需的最小配置参数集合。
"""

from types import SimpleNamespace


def create_minimal_ode_config():
    """创建ODE模块的最小配置。

    Returns:
        SimpleNamespace: 包含ODE模块所需的最小配置
    """
    cfg = SimpleNamespace()

    # 模型配置
    cfg.MODEL = SimpleNamespace()

    # === 核心ODE参数 (必需) ===
    cfg.MODEL.IMPUTE = False                    # 是否启用缺失数据填充
    cfg.MODEL.SOLVER = "euler"                  # ODE求解器: 'euler' 或 'midpoint'

    # === 编码器配置 (必需) ===
    cfg.MODEL.ENCODER = SimpleNamespace()
    cfg.MODEL.ENCODER.OUT_CHANNELS = 64         # 编码器输出通道数

    # === 小编码器配置 (必需) ===
    cfg.MODEL.SMALL_ENCODER = SimpleNamespace()
    cfg.MODEL.SMALL_ENCODER.FILTER_SIZE = 64    # 过滤器大小
    cfg.MODEL.SMALL_ENCODER.SKIPCO = False      # 是否使用跳跃连接

    # === 未来预测配置 (必需) ===
    cfg.MODEL.FUTURE_PRED = SimpleNamespace()
    cfg.MODEL.FUTURE_PRED.USE_VARIABLE_ODE_STEP = False  # 是否使用变步长
    cfg.MODEL.FUTURE_PRED.N_GRU_BLOCKS = 2      # 空间GRU层数 (可选)
    cfg.MODEL.FUTURE_PRED.N_RES_LAYERS = 1      # 残差块层数 (可选)
    cfg.MODEL.FUTURE_PRED.MIXTURE = True        # 是否使用混合分布 (可选)
    cfg.MODEL.FUTURE_PRED.DELTA_T = 0.05        # 时间步长 (可选)

    # === 分布配置 (可选但建议设置) ===
    cfg.MODEL.DISTRIBUTION = SimpleNamespace()
    cfg.MODEL.DISTRIBUTION.LATENT_DIM = 64      # 隐藏层维度
    cfg.MODEL.DISTRIBUTION.MIN_LOG_SIGMA = -5.0 # 最小对数方差
    cfg.MODEL.DISTRIBUTION.MAX_LOG_SIGMA = 5.0  # 最大对数方差

    return cfg


def create_custom_ode_config(
    out_channels=64,
    latent_dim=64,
    solver="euler",
    use_skipco=False,
    use_variable_step=False,
    delta_t=0.05
):
    """创建自定义ODE配置。

    Args:
        out_channels (int): 编码器输出通道数
        latent_dim (int): 隐藏层维度
        solver (str): ODE求解器 ('euler' 或 'midpoint')
        use_skipco (bool): 是否使用跳跃连接
        use_variable_step (bool): 是否使用变步长
        delta_t (float): 时间步长

    Returns:
        SimpleNamespace: 自定义配置
    """
    cfg = create_minimal_ode_config()

    # 自定义参数
    cfg.MODEL.ENCODER.OUT_CHANNELS = out_channels
    cfg.MODEL.DISTRIBUTION.LATENT_DIM = latent_dim
    cfg.MODEL.SOLVER = solver
    cfg.MODEL.SMALL_ENCODER.SKIPCO = use_skipco
    cfg.MODEL.FUTURE_PRED.USE_VARIABLE_ODE_STEP = use_variable_step
    cfg.MODEL.FUTURE_PRED.DELTA_T = delta_t

    return cfg


# === 使用示例 ===
if __name__ == "__main__":
    # 示例1: 最小配置
    minimal_cfg = create_minimal_ode_config()

    # 示例2: 自定义配置
    custom_cfg = create_custom_ode_config(
        out_channels=128,
        latent_dim=128,
        solver="midpoint",
        use_variable_step=True
    )

    print("✅ ODE配置创建完成!")
    print(f"最小配置通道数: {minimal_cfg.MODEL.ENCODER.OUT_CHANNELS}")
    print(f"自定义配置通道数: {custom_cfg.MODEL.ENCODER.OUT_CHANNELS}")