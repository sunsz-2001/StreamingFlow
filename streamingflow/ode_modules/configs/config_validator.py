"""
ODE模块配置验证工具

这个工具用于验证配置是否包含ODE模块所需的所有参数。
"""

def validate_ode_config(cfg):
    """验证ODE模块配置的完整性。

    Args:
        cfg: 配置对象 (支持字典或SimpleNamespace)

    Returns:
        tuple: (is_valid: bool, missing_keys: list, warnings: list)
    """
    missing_keys = []
    warnings = []

    # 必需的配置项
    required_keys = [
        'MODEL.IMPUTE',
        'MODEL.SOLVER',
        'MODEL.ENCODER.OUT_CHANNELS',
        'MODEL.SMALL_ENCODER.FILTER_SIZE',
        'MODEL.SMALL_ENCODER.SKIPCO',
        'MODEL.FUTURE_PRED.USE_VARIABLE_ODE_STEP'
    ]

    # 推荐的配置项
    recommended_keys = [
        'MODEL.DISTRIBUTION.LATENT_DIM',
        'MODEL.DISTRIBUTION.MIN_LOG_SIGMA',
        'MODEL.DISTRIBUTION.MAX_LOG_SIGMA',
        'MODEL.FUTURE_PRED.DELTA_T'
    ]

    def get_nested_value(obj, key_path):
        """获取嵌套属性值."""
        keys = key_path.split('.')
        current = obj
        for key in keys:
            if hasattr(current, key):
                current = getattr(current, key)
            elif isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    # 检查必需配置
    for key in required_keys:
        value = get_nested_value(cfg, key)
        if value is None:
            missing_keys.append(key)

    # 检查推荐配置
    for key in recommended_keys:
        value = get_nested_value(cfg, key)
        if value is None:
            warnings.append(f"推荐设置 {key} (当前使用默认值)")

    # 检查配置一致性
    out_channels = get_nested_value(cfg, 'MODEL.ENCODER.OUT_CHANNELS')
    filter_size = get_nested_value(cfg, 'MODEL.SMALL_ENCODER.FILTER_SIZE')
    latent_dim = get_nested_value(cfg, 'MODEL.DISTRIBUTION.LATENT_DIM')

    if out_channels and filter_size and out_channels != filter_size:
        warnings.append(f"OUT_CHANNELS ({out_channels}) 与 FILTER_SIZE ({filter_size}) 不匹配，可能影响性能")

    if out_channels and latent_dim and out_channels != latent_dim:
        warnings.append(f"OUT_CHANNELS ({out_channels}) 与 LATENT_DIM ({latent_dim}) 不匹配，建议保持一致")

    # 检查求解器选项
    solver = get_nested_value(cfg, 'MODEL.SOLVER')
    if solver and solver not in ['euler', 'midpoint', 'dopri5']:
        warnings.append(f"不支持的求解器 '{solver}'，支持的选项: euler, midpoint, dopri5")

    is_valid = len(missing_keys) == 0
    return is_valid, missing_keys, warnings


def print_config_report(cfg):
    """打印配置验证报告。

    Args:
        cfg: 配置对象
    """
    is_valid, missing_keys, warnings = validate_ode_config(cfg)

    print("🔍 ODE模块配置验证报告")
    print("=" * 40)

    if is_valid:
        print("✅ 配置验证通过!")
    else:
        print("❌ 配置验证失败!")
        print("\n缺少必需配置:")
        for key in missing_keys:
            print(f"  - {key}")

    if warnings:
        print(f"\n⚠️  警告信息 ({len(warnings)}个):")
        for warning in warnings:
            print(f"  - {warning}")

    print("\n" + "=" * 40)
    return is_valid


def get_config_summary(cfg):
    """获取配置摘要。

    Args:
        cfg: 配置对象

    Returns:
        dict: 配置摘要
    """
    def get_nested_value(obj, key_path):
        keys = key_path.split('.')
        current = obj
        for key in keys:
            if hasattr(current, key):
                current = getattr(current, key)
            elif isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return "未设置"
        return current

    summary = {
        "编码器通道数": get_nested_value(cfg, 'MODEL.ENCODER.OUT_CHANNELS'),
        "隐藏层维度": get_nested_value(cfg, 'MODEL.DISTRIBUTION.LATENT_DIM'),
        "ODE求解器": get_nested_value(cfg, 'MODEL.SOLVER'),
        "时间步长": get_nested_value(cfg, 'MODEL.FUTURE_PRED.DELTA_T'),
        "是否使用跳跃连接": get_nested_value(cfg, 'MODEL.SMALL_ENCODER.SKIPCO'),
        "是否使用变步长": get_nested_value(cfg, 'MODEL.FUTURE_PRED.USE_VARIABLE_ODE_STEP'),
    }

    return summary


if __name__ == "__main__":
    # 测试配置验证
    from .minimal_ode_config import create_minimal_ode_config

    print("测试最小配置...")
    cfg = create_minimal_ode_config()
    is_valid = print_config_report(cfg)

    if is_valid:
        print("\n📋 配置摘要:")
        summary = get_config_summary(cfg)
        for key, value in summary.items():
            print(f"  {key}: {value}")