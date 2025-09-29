"""
ODE模块包装器示例

这个文件展示了如何将输入适配器、ODE模块和输出适配器组合成完整的解决方案。
"""

import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, Optional

# 导入适配器
from .input_adapters import (
    TimeSeriesAdapter, ImageSequenceAdapter, PointCloudAdapter, MultiModalAdapter
)
from .output_adapters import (
    TimeSeriesOutputAdapter, ImageSequenceOutputAdapter,
    PointCloudOutputAdapter, SegmentationOutputAdapter
)

# 导入ODE模块
try:
    from .. import NNFOwithBayesianJumps, FuturePredictionODE
    from ..configs.minimal_ode_config import create_custom_ode_config
except ImportError:
    # 如果相对导入失败，可能需要调整路径
    pass


class ODEWrapper(nn.Module):
    """ODE模块包装器基类"""

    def __init__(self, ode_model, input_adapter, output_adapter):
        """
        Args:
            ode_model: ODE模型实例
            input_adapter: 输入适配器
            output_adapter: 输出适配器
        """
        super().__init__()
        self.ode_model = ode_model
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter

    def forward(self, *args, **kwargs):
        """子类需要实现具体的前向传播逻辑"""
        raise NotImplementedError


class TimeSeriesODEWrapper(ODEWrapper):
    """时间序列ODE包装器"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64,
                 feature_size: int = 32, **ode_kwargs):
        """
        Args:
            input_dim: 输入时间序列维度
            output_dim: 输出时间序列维度
            hidden_dim: ODE隐藏层维度
            feature_size: 中间特征图尺寸
            **ode_kwargs: ODE配置参数
        """
        # 创建适配器
        input_adapter = TimeSeriesAdapter(
            input_dim=input_dim,
            output_channels=hidden_dim,
            feature_size=feature_size
        )

        output_adapter = TimeSeriesOutputAdapter(
            feature_channels=hidden_dim,
            feature_size=feature_size,
            output_dim=output_dim
        )

        # 创建ODE配置
        cfg = create_custom_ode_config(
            out_channels=hidden_dim,
            latent_dim=hidden_dim,
            **ode_kwargs
        )

        # 创建ODE模型
        ode_model = NNFOwithBayesianJumps(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            cfg=cfg
        )

        super().__init__(ode_model, input_adapter, output_adapter)

    def forward(self, time_series: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None,
                target_times: Optional[torch.Tensor] = None,
                delta_t: float = 0.1) -> Dict[str, Any]:
        """
        时间序列预测

        Args:
            time_series: [B, T, input_dim] 输入时间序列
            timestamps: [T] 观测时间戳
            target_times: [T_future] 预测时间戳
            delta_t: 积分步长

        Returns:
            dict: 包含预测结果和相关信息
        """
        # 1. 输入适配
        adapted_input = self.input_adapter.adapt_input(time_series, timestamps)

        # 2. 生成默认目标时间
        if target_times is None:
            seq_len = time_series.shape[1]
            target_times = torch.linspace(seq_len, seq_len + 3, 4, dtype=torch.float32)

        # 3. ODE预测
        final_state, ode_loss, ode_predictions = self.ode_model(
            times=adapted_input['times'],
            input=adapted_input['current_input'],
            obs=adapted_input['observations'],
            delta_t=delta_t,
            T=target_times
        )

        # 4. 输出适配
        output_result = self.output_adapter.adapt_output(ode_predictions, target_times)

        # 5. 组合结果
        result = {
            'predictions': output_result['predictions'],
            'timestamps': target_times,
            'ode_loss': ode_loss,
            'statistics': output_result.get('statistics', {}),
            'raw_ode_output': ode_predictions
        }

        return result


class VideoODEWrapper(ODEWrapper):
    """视频预测ODE包装器"""

    def __init__(self, input_channels: int = 3, feature_channels: int = 64,
                 target_size: tuple = (128, 128), **ode_kwargs):
        """
        Args:
            input_channels: 输入视频通道数
            feature_channels: 特征通道数
            target_size: 目标视频尺寸
            **ode_kwargs: ODE配置参数
        """
        # 创建适配器
        input_adapter = ImageSequenceAdapter(
            target_size=target_size,
            target_channels=feature_channels
        )

        output_adapter = ImageSequenceOutputAdapter(
            feature_channels=feature_channels,
            target_size=target_size,
            target_channels=input_channels
        )

        # 创建ODE配置
        cfg = create_custom_ode_config(
            out_channels=feature_channels,
            latent_dim=feature_channels,
            **ode_kwargs
        )

        # 创建ODE模型
        ode_model = NNFOwithBayesianJumps(
            input_size=feature_channels,
            hidden_size=feature_channels,
            cfg=cfg
        )

        super().__init__(ode_model, input_adapter, output_adapter)

    def forward(self, video_sequence: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None,
                target_times: Optional[torch.Tensor] = None,
                delta_t: float = 0.1) -> Dict[str, Any]:
        """
        视频序列预测

        Args:
            video_sequence: [B, T, C, H, W] 输入视频序列
            timestamps: [T] 观测时间戳
            target_times: [T_future] 预测时间戳
            delta_t: 积分步长

        Returns:
            dict: 包含预测结果和相关信息
        """
        # 1. 输入适配
        adapted_input = self.input_adapter.adapt_input(video_sequence, timestamps)

        # 2. 生成默认目标时间
        if target_times is None:
            seq_len = video_sequence.shape[1]
            target_times = torch.linspace(seq_len, seq_len + 2, 3, dtype=torch.float32)

        # 3. ODE预测
        final_state, ode_loss, ode_predictions = self.ode_model(
            times=adapted_input['times'],
            input=adapted_input['current_input'],
            obs=adapted_input['observations'],
            delta_t=delta_t,
            T=target_times
        )

        # 4. 输出适配
        output_result = self.output_adapter.adapt_output(ode_predictions, target_times)

        # 5. 组合结果
        result = {
            'predictions': output_result['predictions'],
            'timestamps': target_times,
            'ode_loss': ode_loss,
            'quality_metrics': output_result.get('quality_metrics', {}),
            'raw_ode_output': ode_predictions
        }

        return result


class LidarODEWrapper(ODEWrapper):
    """激光雷达点云ODE包装器"""

    def __init__(self, grid_size: tuple = (64, 64), feature_channels: int = 64,
                 x_range: tuple = (-50, 50), y_range: tuple = (-50, 50),
                 max_points: int = 1000, **ode_kwargs):
        """
        Args:
            grid_size: BEV网格尺寸
            feature_channels: 特征通道数
            x_range: X轴范围
            y_range: Y轴范围
            max_points: 最大输出点数
            **ode_kwargs: ODE配置参数
        """
        # 创建适配器
        input_adapter = PointCloudAdapter(
            grid_size=grid_size,
            feature_channels=feature_channels,
            x_range=x_range,
            y_range=y_range
        )

        output_adapter = PointCloudOutputAdapter(
            feature_channels=feature_channels,
            grid_size=grid_size,
            x_range=x_range,
            y_range=y_range,
            max_points=max_points
        )

        # 创建ODE配置
        cfg = create_custom_ode_config(
            out_channels=feature_channels,
            latent_dim=feature_channels,
            **ode_kwargs
        )

        # 创建ODE模型
        ode_model = NNFOwithBayesianJumps(
            input_size=feature_channels,
            hidden_size=feature_channels,
            cfg=cfg
        )

        super().__init__(ode_model, input_adapter, output_adapter)

    def forward(self, point_clouds: List[torch.Tensor],
                timestamps: Optional[torch.Tensor] = None,
                target_times: Optional[torch.Tensor] = None,
                delta_t: float = 0.1) -> Dict[str, Any]:
        """
        点云序列预测

        Args:
            point_clouds: List[Tensor] 点云序列，每个点云形状为[N, 4]
            timestamps: [T] 观测时间戳
            target_times: [T_future] 预测时间戳
            delta_t: 积分步长

        Returns:
            dict: 包含预测结果和相关信息
        """
        # 1. 输入适配
        adapted_input = self.input_adapter.adapt_input(point_clouds, timestamps)

        # 2. 生成默认目标时间
        if target_times is None:
            seq_len = len(point_clouds)
            target_times = torch.linspace(seq_len, seq_len + 2, 3, dtype=torch.float32)

        # 3. ODE预测
        final_state, ode_loss, ode_predictions = self.ode_model(
            times=adapted_input['times'],
            input=adapted_input['current_input'],
            obs=adapted_input['observations'],
            delta_t=delta_t,
            T=target_times
        )

        # 4. 输出适配
        output_result = self.output_adapter.adapt_output(ode_predictions, target_times)

        # 5. 组合结果
        result = {
            'predictions': output_result['predictions'],
            'timestamps': target_times,
            'ode_loss': ode_loss,
            'statistics': output_result.get('statistics', {}),
            'raw_ode_output': ode_predictions
        }

        return result


class SegmentationODEWrapper(ODEWrapper):
    """语义分割ODE包装器"""

    def __init__(self, input_channels: int = 3, feature_channels: int = 64,
                 num_classes: int = 21, target_size: tuple = (256, 256),
                 **ode_kwargs):
        """
        Args:
            input_channels: 输入图像通道数
            feature_channels: 特征通道数
            num_classes: 分割类别数
            target_size: 目标分割图尺寸
            **ode_kwargs: ODE配置参数
        """
        # 创建适配器
        input_adapter = ImageSequenceAdapter(
            target_size=target_size,
            target_channels=feature_channels
        )

        output_adapter = SegmentationOutputAdapter(
            feature_channels=feature_channels,
            num_classes=num_classes,
            target_size=target_size
        )

        # 创建ODE配置
        cfg = create_custom_ode_config(
            out_channels=feature_channels,
            latent_dim=feature_channels,
            **ode_kwargs
        )

        # 创建ODE模型
        ode_model = NNFOwithBayesianJumps(
            input_size=feature_channels,
            hidden_size=feature_channels,
            cfg=cfg
        )

        super().__init__(ode_model, input_adapter, output_adapter)
        self.num_classes = num_classes

    def forward(self, image_sequence: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None,
                target_times: Optional[torch.Tensor] = None,
                delta_t: float = 0.1) -> Dict[str, Any]:
        """
        图像序列分割预测

        Args:
            image_sequence: [B, T, C, H, W] 输入图像序列
            timestamps: [T] 观测时间戳
            target_times: [T_future] 预测时间戳
            delta_t: 积分步长

        Returns:
            dict: 包含预测结果和相关信息
        """
        # 1. 输入适配
        adapted_input = self.input_adapter.adapt_input(image_sequence, timestamps)

        # 2. 生成默认目标时间
        if target_times is None:
            seq_len = image_sequence.shape[1]
            target_times = torch.linspace(seq_len, seq_len + 1, 2, dtype=torch.float32)

        # 3. ODE预测
        final_state, ode_loss, ode_predictions = self.ode_model(
            times=adapted_input['times'],
            input=adapted_input['current_input'],
            obs=adapted_input['observations'],
            delta_t=delta_t,
            T=target_times
        )

        # 4. 输出适配
        output_result = self.output_adapter.adapt_output(ode_predictions, target_times)

        # 5. 组合结果
        result = {
            'logits': output_result['logits'],
            'probabilities': output_result['probabilities'],
            'predictions': output_result['predictions'],
            'timestamps': target_times,
            'ode_loss': ode_loss,
            'quality_metrics': output_result.get('quality_metrics', {}),
            'raw_ode_output': ode_predictions
        }

        return result


class CustomODEWrapper(ODEWrapper):
    """自定义ODE包装器 - 用户可以提供自己的适配器"""

    def __init__(self, input_adapter, output_adapter, ode_config=None, **ode_kwargs):
        """
        Args:
            input_adapter: 用户自定义的输入适配器
            output_adapter: 用户自定义的输出适配器
            ode_config: ODE配置，如果为None则自动创建
            **ode_kwargs: ODE配置参数
        """
        # 创建ODE配置
        if ode_config is None:
            ode_config = create_custom_ode_config(**ode_kwargs)

        # 从适配器推断通道数
        if hasattr(input_adapter, 'output_channels'):
            channels = input_adapter.output_channels
        elif hasattr(input_adapter, 'target_channels'):
            channels = input_adapter.target_channels
        else:
            channels = ode_kwargs.get('out_channels', 64)

        # 创建ODE模型
        ode_model = NNFOwithBayesianJumps(
            input_size=channels,
            hidden_size=channels,
            cfg=ode_config
        )

        super().__init__(ode_model, input_adapter, output_adapter)

    def forward(self, input_data, timestamps=None, target_times=None, delta_t=0.1, **kwargs):
        """
        通用前向传播

        Args:
            input_data: 输入数据 (格式取决于输入适配器)
            timestamps: 观测时间戳
            target_times: 预测时间戳
            delta_t: 积分步长
            **kwargs: 其他参数

        Returns:
            dict: 预测结果
        """
        # 1. 输入适配
        adapted_input = self.input_adapter.adapt_input(input_data, timestamps)

        # 2. 生成默认目标时间
        if target_times is None:
            if hasattr(input_data, 'shape') and len(input_data.shape) > 1:
                seq_len = input_data.shape[1]
            else:
                seq_len = len(input_data) if isinstance(input_data, (list, tuple)) else 5
            target_times = torch.linspace(seq_len, seq_len + 2, 3, dtype=torch.float32)

        # 3. ODE预测
        final_state, ode_loss, ode_predictions = self.ode_model(
            times=adapted_input['times'],
            input=adapted_input['current_input'],
            obs=adapted_input['observations'],
            delta_t=delta_t,
            T=target_times
        )

        # 4. 输出适配
        output_result = self.output_adapter.adapt_output(ode_predictions, target_times)

        # 5. 组合结果
        result = output_result.copy()
        result.update({
            'timestamps': target_times,
            'ode_loss': ode_loss,
            'raw_ode_output': ode_predictions
        })

        return result


# 便捷的包装器工厂函数
def create_ode_wrapper(wrapper_type: str, **kwargs) -> ODEWrapper:
    """
    根据类型创建ODE包装器

    Args:
        wrapper_type: 包装器类型
        **kwargs: 包装器参数

    Returns:
        ODEWrapper实例
    """
    wrappers = {
        'timeseries': TimeSeriesODEWrapper,
        'video': VideoODEWrapper,
        'lidar': LidarODEWrapper,
        'segmentation': SegmentationODEWrapper,
        'custom': CustomODEWrapper
    }

    if wrapper_type not in wrappers:
        raise ValueError(f"不支持的包装器类型: {wrapper_type}")

    return wrappers[wrapper_type](**kwargs)


# 使用示例
if __name__ == "__main__":
    print("🔧 ODE包装器使用示例")

    # 时间序列预测示例
    print("\n1. 时间序列预测:")
    ts_wrapper = TimeSeriesODEWrapper(input_dim=10, output_dim=10, hidden_dim=32)
    time_series = torch.randn(2, 5, 10)  # [batch, time, features]

    with torch.no_grad():
        ts_result = ts_wrapper(time_series)
        print(f"   输入形状: {time_series.shape}")
        print(f"   预测形状: {ts_result['predictions'].shape}")

    # 视频预测示例
    print("\n2. 视频预测:")
    video_wrapper = VideoODEWrapper(feature_channels=32, target_size=(64, 64))
    video_data = torch.randn(1, 4, 3, 64, 64)  # [batch, time, channels, h, w]

    with torch.no_grad():
        video_result = video_wrapper(video_data)
        print(f"   输入形状: {video_data.shape}")
        print(f"   预测形状: {video_result['predictions'].shape}")

    print("\n✅ 所有包装器示例完成!")