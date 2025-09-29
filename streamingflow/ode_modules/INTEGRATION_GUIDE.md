# ODE模块集成指南

## 🎯 在其他系统中使用ODE模块

这个指南将教你如何在任何PyTorch项目中集成和使用ODE模块。

## 📦 1. 安装和导入

### 方法1: 直接复制模块 (推荐)

```bash
# 将整个ode_modules文件夹复制到你的项目中
cp -r /path/to/StreamingFlow/streamingflow/ode_modules /your/project/path/

# 项目结构应该是:
your_project/
├── ode_modules/          # 复制的ODE模块
│   ├── __init__.py
│   ├── cores/
│   ├── cells/
│   ├── utils/
│   └── configs/
├── your_model.py         # 你的模型代码
└── main.py              # 主程序
```

### 方法2: Python包安装

```python
# 在你的项目中导入
import sys
sys.path.append('/path/to/StreamingFlow')  # 添加StreamingFlow路径

from streamingflow.ode_modules import NNFOwithBayesianJumps, FuturePredictionODE
```

## 🚀 2. 基础使用示例

### 2.1 最简单的使用方式

```python
import torch
from ode_modules import NNFOwithBayesianJumps
from ode_modules.configs.minimal_ode_config import create_minimal_ode_config

# 创建配置
cfg = create_minimal_ode_config()

# 创建模型
ode_model = NNFOwithBayesianJumps(
    input_size=64,    # 输入特征通道数
    hidden_size=64,   # 隐藏层维度
    cfg=cfg
)

# 准备输入数据
batch_size = 2
seq_len = 3
channels = 64
height, width = 32, 32

# 输入数据格式: [B, 1, C, H, W]
current_input = torch.randn(batch_size, 1, channels, height, width)

# 观测序列: [B, T, C, H, W]
observations = torch.randn(batch_size, seq_len, channels, height, width)

# 时间戳
times = torch.tensor([0.0, 0.5, 1.0])  # 观测时间点
target_times = torch.tensor([1.5, 2.0, 2.5])  # 预测时间点

# 前向传播
final_state, loss, predictions = ode_model(
    times=times,
    input=current_input,
    obs=observations,
    delta_t=0.1,
    T=target_times
)

print(f"预测结果形状: {predictions.shape}")  # [B, T_future, C, H, W]
```

### 2.2 集成到自定义模型中

```python
import torch
import torch.nn as nn
from ode_modules import NNFOwithBayesianJumps, FuturePredictionODE
from ode_modules.configs.minimal_ode_config import create_custom_ode_config

class MyVideoModelWithODE(nn.Module):
    """
    示例: 将ODE模块集成到视频预测模型中
    """

    def __init__(self, input_channels=3, feature_channels=64, num_future_frames=4):
        super().__init__()

        # 特征提取器 (例如CNN编码器)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, feature_channels, 3, padding=1),
            nn.ReLU()
        )

        # 创建ODE配置
        self.cfg = create_custom_ode_config(
            out_channels=feature_channels,
            latent_dim=feature_channels,
            solver="euler",
            delta_t=0.05
        )

        # 贝叶斯ODE模块用于时序建模
        self.ode_predictor = NNFOwithBayesianJumps(
            input_size=feature_channels,
            hidden_size=feature_channels,
            cfg=self.cfg
        )

        # 输出解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, input_channels, 3, padding=1),
            nn.Sigmoid()
        )

        self.num_future_frames = num_future_frames

    def forward(self, video_sequence, target_times=None):
        """
        Args:
            video_sequence: [B, T, C, H, W] 输入视频序列
            target_times: [T_future] 目标预测时间点

        Returns:
            predicted_frames: [B, T_future, C, H, W] 预测的未来帧
        """
        batch_size, seq_len, channels, height, width = video_sequence.shape

        # 1. 特征提取
        # 将时序维度展平进行特征提取
        flat_frames = video_sequence.view(-1, channels, height, width)
        features = self.encoder(flat_frames)  # [B*T, feature_channels, H, W]

        # 恢复时序维度
        feature_channels = features.shape[1]
        features = features.view(batch_size, seq_len, feature_channels, height, width)

        # 2. 时间戳 (假设等间隔)
        times = torch.linspace(0, seq_len-1, seq_len)

        # 3. 准备ODE输入
        current_input = features[:, -1:, :, :, :]  # 最后一帧作为当前输入
        observations = features  # 所有帧作为观测

        # 4. 目标时间点
        if target_times is None:
            target_times = torch.linspace(seq_len, seq_len + self.num_future_frames - 1,
                                        self.num_future_frames)

        # 5. ODE预测
        try:
            final_state, ode_loss, ode_predictions = self.ode_predictor(
                times=times,
                input=current_input,
                obs=observations,
                delta_t=0.1,
                T=target_times
            )

            # 6. 解码到像素空间
            # ode_predictions: [B, T_future, feature_channels, H, W]
            batch_size, future_len = ode_predictions.shape[:2]
            flat_predictions = ode_predictions.view(-1, feature_channels, height, width)
            decoded_frames = self.decoder(flat_predictions)

            # 恢复形状
            predicted_frames = decoded_frames.view(batch_size, future_len, channels, height, width)

            return predicted_frames, ode_loss

        except Exception as e:
            print(f"ODE预测失败: {e}")
            # 返回零预测作为后备
            predicted_frames = torch.zeros(batch_size, len(target_times), channels, height, width)
            return predicted_frames, 0.0

# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = MyVideoModelWithODE(input_channels=3, feature_channels=64, num_future_frames=4)

    # 模拟输入
    batch_size = 1
    seq_len = 5
    video_input = torch.randn(batch_size, seq_len, 3, 64, 64)

    # 预测
    predicted_frames, loss = model(video_input)
    print(f"输入形状: {video_input.shape}")
    print(f"预测形状: {predicted_frames.shape}")
    print(f"ODE损失: {loss}")
```

## 🔧 3. 高级用法和自定义

### 3.1 自定义时间序列数据

```python
class TimeSeriesODEModel(nn.Module):
    """时间序列预测模型示例"""

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()

        # 将1D时间序列映射到2D特征图
        self.feature_mapper = nn.Linear(input_dim, hidden_dim * 4 * 4)

        # ODE配置
        cfg = create_custom_ode_config(
            out_channels=hidden_dim,
            latent_dim=hidden_dim,
            solver="midpoint",  # 更精确的求解器
            delta_t=0.02
        )

        self.ode_model = NNFOwithBayesianJumps(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            cfg=cfg
        )

        # 输出映射
        self.output_mapper = nn.Linear(hidden_dim * 4 * 4, input_dim)
        self.hidden_dim = hidden_dim

    def forward(self, time_series, timestamps, target_timestamps):
        """
        Args:
            time_series: [B, T, input_dim] 时间序列数据
            timestamps: [T] 观测时间戳
            target_timestamps: [T_future] 预测时间戳
        """
        batch_size, seq_len, input_dim = time_series.shape

        # 映射到2D特征图
        features_flat = self.feature_mapper(time_series.view(-1, input_dim))
        features_2d = features_flat.view(batch_size, seq_len, self.hidden_dim, 4, 4)

        # ODE预测
        current_input = features_2d[:, -1:, :, :, :]

        final_state, loss, predictions = self.ode_model(
            times=timestamps,
            input=current_input,
            obs=features_2d,
            delta_t=0.02,
            T=target_timestamps
        )

        # 映射回1D
        pred_flat = predictions.view(-1, self.hidden_dim * 4 * 4)
        output_series = self.output_mapper(pred_flat)
        output_series = output_series.view(batch_size, len(target_timestamps), input_dim)

        return output_series, loss

# 使用示例
ts_model = TimeSeriesODEModel(input_dim=10, hidden_dim=32)
time_series = torch.randn(2, 8, 10)  # [batch, time, features]
timestamps = torch.linspace(0, 7, 8)
target_timestamps = torch.linspace(8, 11, 4)

predictions, loss = ts_model(time_series, timestamps, target_timestamps)
print(f"时间序列预测形状: {predictions.shape}")  # [2, 4, 10]
```

### 3.2 多模态融合示例

```python
class MultimodalODEModel(nn.Module):
    """多模态数据融合的ODE模型"""

    def __init__(self, camera_channels=64, lidar_channels=64, fusion_channels=128):
        super().__init__()

        # 模态特定编码器
        self.camera_encoder = nn.Conv2d(camera_channels, fusion_channels//2, 1)
        self.lidar_encoder = nn.Conv2d(lidar_channels, fusion_channels//2, 1)

        # 融合层
        self.fusion_layer = nn.Conv2d(fusion_channels, fusion_channels, 3, padding=1)

        # 创建配置
        cfg = create_custom_ode_config(
            out_channels=fusion_channels,
            latent_dim=fusion_channels,
            solver="euler",
            delta_t=0.05
        )

        # 未来预测ODE (更适合多模态场景)
        self.future_predictor = FuturePredictionODE(
            in_channels=fusion_channels,
            latent_dim=fusion_channels,
            cfg=cfg,
            n_gru_blocks=3,  # 更多GRU层处理复杂特征
            delta_t=0.05
        )

    def forward(self, camera_features, lidar_features, camera_timestamps,
                lidar_timestamps, target_timestamps):
        """
        Args:
            camera_features: [B, T_cam, C_cam, H, W]
            lidar_features: [B, T_lidar, C_lidar, H, W]
            camera_timestamps: [T_cam]
            lidar_timestamps: [T_lidar]
            target_timestamps: [T_future]
        """
        # 编码各模态特征
        batch_size = camera_features.shape[0]

        # 处理相机特征
        camera_flat = camera_features.view(-1, *camera_features.shape[2:])
        camera_encoded = self.camera_encoder(camera_flat)
        camera_encoded = camera_encoded.view(*camera_features.shape[:2], *camera_encoded.shape[1:])

        # 处理激光雷达特征
        lidar_flat = lidar_features.view(-1, *lidar_features.shape[2:])
        lidar_encoded = self.lidar_encoder(lidar_flat)
        lidar_encoded = lidar_encoded.view(*lidar_features.shape[:2], *lidar_encoded.shape[1:])

        # 当前融合特征 (假设最新的相机和激光雷达特征)
        current_camera = camera_encoded[:, -1:, :, :, :]
        current_lidar = lidar_encoded[:, -1:, :, :, :]
        current_fused = torch.cat([current_camera, current_lidar], dim=2)
        current_fused = self.fusion_layer(current_fused.squeeze(1)).unsqueeze(1)

        # 未来预测
        predictions, loss = self.future_predictor(
            future_prediction_input=current_fused,
            camera_states=camera_encoded,
            lidar_states=lidar_encoded,
            camera_timestamp=camera_timestamps.unsqueeze(0).repeat(batch_size, 1),
            lidar_timestamp=lidar_timestamps.unsqueeze(0).repeat(batch_size, 1),
            target_timestamp=target_timestamps.unsqueeze(0).repeat(batch_size, 1)
        )

        return predictions, loss
```

## 🛠️ 4. 配置和优化

### 4.1 性能优化配置

```python
# 快速配置 (用于实时应用)
fast_cfg = create_custom_ode_config(
    out_channels=32,          # 较小的通道数
    latent_dim=32,
    solver="euler",           # 最快的求解器
    delta_t=0.1,             # 较大的时间步
    use_variable_step=False   # 固定步长更快
)

# 精确配置 (用于研究/离线处理)
accurate_cfg = create_custom_ode_config(
    out_channels=128,         # 更大的通道数
    latent_dim=128,
    solver="midpoint",        # 更精确的求解器
    delta_t=0.01,            # 更小的时间步
    use_variable_step=True    # 自适应步长
)

# 平衡配置 (推荐)
balanced_cfg = create_custom_ode_config(
    out_channels=64,
    latent_dim=64,
    solver="euler",
    delta_t=0.05,
    use_variable_step=False
)
```

### 4.2 GPU内存优化

```python
import torch.utils.checkpoint as checkpoint

class MemoryEfficientODEModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ode_model = NNFOwithBayesianJumps(64, 64, cfg)

    def forward(self, *args, **kwargs):
        # 使用梯度检查点节省内存
        return checkpoint.checkpoint(self.ode_model, *args, **kwargs)

# 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler

model = MemoryEfficientODEModel(cfg)
scaler = GradScaler()

with autocast():
    predictions, loss = model(times, input_data, obs, delta_t, target_times)
```

## 📊 5. 测试和验证

### 5.1 基础功能测试

```python
def test_ode_integration():
    """测试ODE模块基础功能"""
    cfg = create_minimal_ode_config()
    model = NNFOwithBayesianJumps(64, 64, cfg)

    # 测试数据
    batch_size = 2
    current_input = torch.randn(batch_size, 1, 64, 32, 32)
    observations = torch.randn(batch_size, 3, 64, 32, 32)
    times = torch.tensor([0.0, 0.5, 1.0])
    target_times = torch.tensor([1.5, 2.0])

    # 前向传播
    try:
        final_state, loss, predictions = model(
            times=times,
            input=current_input,
            obs=observations,
            delta_t=0.1,
            T=target_times
        )

        print("✅ ODE模块测试通过")
        print(f"预测形状: {predictions.shape}")
        print(f"损失值: {loss}")
        return True

    except Exception as e:
        print(f"❌ ODE模块测试失败: {e}")
        return False

# 运行测试
if __name__ == "__main__":
    test_ode_integration()
```

## 🚨 6. 常见问题和解决方案

### 6.1 形状不匹配问题

```python
# 问题: 输入形状错误
# 解决: 确保输入格式正确

# 错误示例
# input_wrong = torch.randn(2, 64, 32, 32)  # 缺少时序维度

# 正确示例
input_correct = torch.randn(2, 1, 64, 32, 32)  # [B, 1, C, H, W]
obs_correct = torch.randn(2, 3, 64, 32, 32)    # [B, T, C, H, W]
```

### 6.2 配置兼容性问题

```python
# 使用配置验证工具
from ode_modules.configs.config_validator import validate_ode_config

cfg = create_custom_ode_config(out_channels=128, latent_dim=64)  # 不匹配!
is_valid, missing, warnings = validate_ode_config(cfg)

if warnings:
    print("⚠️ 配置警告:")
    for warning in warnings:
        print(f"  - {warning}")
```

### 6.3 性能优化建议

```python
# 1. 批处理优化
def batch_predict(model, inputs, batch_size=4):
    """批量预测减少GPU内存占用"""
    results = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size]
        with torch.no_grad():
            pred = model(batch)
            results.append(pred)
    return torch.cat(results, dim=0)

# 2. 缓存重用
@lru_cache(maxsize=128)
def get_ode_model(channels, solver):
    """缓存模型避免重复创建"""
    cfg = create_custom_ode_config(out_channels=channels, solver=solver)
    return NNFOwithBayesianJumps(channels, channels, cfg)
```

这个指南涵盖了在其他系统中使用ODE模块的所有关键方面。你可以根据具体需求选择合适的集成方式！