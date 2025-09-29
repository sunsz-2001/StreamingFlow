# ODE模块配置指南

## 📋 配置概览

ODE模块需要的配置分为**必需配置**和**可选配置**。只有提供了必需配置，模块才能正常工作。

## ✅ 必需配置参数

| 配置路径 | 类型 | 默认值 | 说明 |
|---------|------|--------|------|
| `MODEL.IMPUTE` | bool | `False` | 是否启用缺失数据填充 |
| `MODEL.SOLVER` | str | `'euler'` | ODE求解器 (`'euler'`, `'midpoint'`, `'dopri5'`) |
| `MODEL.ENCODER.OUT_CHANNELS` | int | `64` | 编码器输出通道数 |
| `MODEL.SMALL_ENCODER.FILTER_SIZE` | int | `64` | 小编码器过滤器大小 |
| `MODEL.SMALL_ENCODER.SKIPCO` | bool | `False` | 是否使用跳跃连接 |
| `MODEL.FUTURE_PRED.USE_VARIABLE_ODE_STEP` | bool | `False` | 是否使用变步长求解 |

## ⚙️ 推荐配置参数

| 配置路径 | 类型 | 默认值 | 说明 |
|---------|------|--------|------|
| `MODEL.DISTRIBUTION.LATENT_DIM` | int | `64` | 隐藏层维度 |
| `MODEL.DISTRIBUTION.MIN_LOG_SIGMA` | float | `-5.0` | 最小对数方差 |
| `MODEL.DISTRIBUTION.MAX_LOG_SIGMA` | float | `5.0` | 最大对数方差 |
| `MODEL.FUTURE_PRED.DELTA_T` | float | `0.05` | 时间步长 |
| `MODEL.FUTURE_PRED.N_GRU_BLOCKS` | int | `2` | 空间GRU层数 |
| `MODEL.FUTURE_PRED.N_RES_LAYERS` | int | `1` | 残差块层数 |
| `MODEL.FUTURE_PRED.MIXTURE` | bool | `True` | 是否使用混合分布 |

## 🚀 快速开始

### 方法1: 使用Python配置

```python
from streamingflow.ode_modules.configs.minimal_ode_config import create_minimal_ode_config
from streamingflow.ode_modules import NNFOwithBayesianJumps, FuturePredictionODE

# 创建最小配置
cfg = create_minimal_ode_config()

# 创建ODE模型
ode_model = NNFOwithBayesianJumps(
    input_size=cfg.MODEL.ENCODER.OUT_CHANNELS,  # 64
    hidden_size=cfg.MODEL.DISTRIBUTION.LATENT_DIM,  # 64
    cfg=cfg
)

# 创建未来预测模型
future_predictor = FuturePredictionODE(
    in_channels=cfg.MODEL.ENCODER.OUT_CHANNELS,  # 64
    latent_dim=cfg.MODEL.DISTRIBUTION.LATENT_DIM,  # 64
    cfg=cfg
)
```

### 方法2: 使用YAML配置

```python
import yaml
from types import SimpleNamespace
from streamingflow.ode_modules import NNFOwithBayesianJumps

# 加载YAML配置
with open('streamingflow/ode_modules/configs/ode_config_template.yml') as f:
    config_dict = yaml.safe_load(f)

# 转换为SimpleNamespace
def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d

cfg = dict_to_namespace(config_dict)

# 使用配置
ode_model = NNFOwithBayesianJumps(
    input_size=cfg.MODEL.ENCODER.OUT_CHANNELS,
    hidden_size=cfg.MODEL.DISTRIBUTION.LATENT_DIM,
    cfg=cfg
)
```

### 方法3: 自定义配置

```python
from streamingflow.ode_modules.configs.minimal_ode_config import create_custom_ode_config

# 创建自定义配置
cfg = create_custom_ode_config(
    out_channels=128,        # 更大的通道数
    latent_dim=128,          # 匹配的隐藏层维度
    solver="midpoint",       # 更精确的求解器
    use_variable_step=True,  # 启用变步长
    delta_t=0.02            # 更小的时间步
)
```

## 🔍 配置验证

在使用配置之前，建议先进行验证：

```python
from streamingflow.ode_modules.configs.config_validator import print_config_report

# 验证配置
cfg = create_minimal_ode_config()
is_valid = print_config_report(cfg)

if not is_valid:
    print("❌ 配置有误，请检查缺失的参数")
else:
    print("✅ 配置正确，可以使用")
```

## 📐 常用配置组合

### 轻量级配置 (快速测试)
```python
lightweight_cfg = create_custom_ode_config(
    out_channels=32,
    latent_dim=32,
    delta_t=0.1
)
```

### 标准配置 (推荐)
```python
standard_cfg = create_custom_ode_config(
    out_channels=64,
    latent_dim=64,
    solver="euler",
    delta_t=0.05
)
```

### 高性能配置 (大模型)
```python
high_performance_cfg = create_custom_ode_config(
    out_channels=256,
    latent_dim=256,
    solver="midpoint",
    use_variable_step=True,
    delta_t=0.02
)
```

## ⚠️ 重要注意事项

1. **通道数匹配**: `ENCODER.OUT_CHANNELS`, `SMALL_ENCODER.FILTER_SIZE`, 和 `DISTRIBUTION.LATENT_DIM` 建议保持一致
2. **求解器选择**:
   - `euler`: 最快，精度一般
   - `midpoint`: 平衡速度和精度
   - `dopri5`: 最精确，但最慢
3. **内存使用**: 通道数越大，内存占用越高
4. **变步长**: 启用时可能提高精度，但增加计算开销

## 🔧 配置来源

配置参数来源于StreamingFlow的主配置系统：

- **基础配置**: `streamingflow/config.py` - 定义所有默认值
- **YAML覆盖**: `streamingflow/configs/*.yml` - 针对具体任务的配置
- **命令行参数**: 可通过 `--config-file` 指定YAML文件

## 📞 支持

如果遇到配置问题，请：
1. 使用配置验证工具检查配置完整性
2. 参考 `minimal_ode_config.py` 中的示例
3. 检查配置参数的数据类型和取值范围