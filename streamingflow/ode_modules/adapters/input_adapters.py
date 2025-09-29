"""
ODE模块输入适配器

这个文件包含各种输入格式到ODE模块标准格式的转换函数。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputAdapter:
    """输入适配器基类"""

    @staticmethod
    def adapt_input(data, target_format):
        """适配输入数据到目标格式"""
        raise NotImplementedError


class TimeSeriesAdapter(InputAdapter):
    """时间序列数据适配器 - 将1D时间序列转换为2D特征图"""

    def __init__(self, input_dim, output_channels=64, feature_size=32):
        """
        Args:
            input_dim: 输入时间序列的维度
            output_channels: 输出特征通道数
            feature_size: 输出特征图的空间尺寸
        """
        self.input_dim = input_dim
        self.output_channels = output_channels
        self.feature_size = feature_size

        # 创建映射网络
        self.mapper = nn.Sequential(
            nn.Linear(input_dim, output_channels * feature_size * feature_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def adapt_input(self, time_series, timestamps=None):
        """
        将时间序列转换为ODE输入格式

        Args:
            time_series: [B, T, input_dim] 时间序列数据
            timestamps: [T] 时间戳，如果为None则自动生成

        Returns:
            dict: 包含适配后的ODE输入
        """
        batch_size, seq_len, input_dim = time_series.shape

        # 1. 映射到2D特征图
        flat_series = time_series.view(-1, input_dim)
        features_flat = self.mapper(flat_series)
        features_2d = features_flat.view(
            batch_size, seq_len, self.output_channels,
            self.feature_size, self.feature_size
        )

        # 2. 准备ODE输入格式
        current_input = features_2d[:, -1:, :, :, :]  # 最后一帧作为当前输入
        observations = features_2d  # 所有帧作为观测

        # 3. 生成时间戳
        if timestamps is None:
            timestamps = torch.linspace(0, seq_len-1, seq_len, dtype=torch.float32)

        return {
            'current_input': current_input,  # [B, 1, C, H, W]
            'observations': observations,     # [B, T, C, H, W]
            'times': timestamps              # [T]
        }


class ImageSequenceAdapter(InputAdapter):
    """图像序列适配器 - 处理视频/图像序列"""

    def __init__(self, target_size=(64, 64), target_channels=64):
        """
        Args:
            target_size: 目标空间尺寸
            target_channels: 目标通道数
        """
        self.target_size = target_size
        self.target_channels = target_channels

        # 特征提取器 (可替换为更复杂的网络)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, target_channels, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(target_size)
        )

    def adapt_input(self, image_sequence, timestamps=None):
        """
        将图像序列转换为ODE输入格式

        Args:
            image_sequence: [B, T, C, H, W] 图像序列
            timestamps: [T] 时间戳

        Returns:
            dict: 适配后的ODE输入
        """
        batch_size, seq_len, channels, height, width = image_sequence.shape

        # 1. 特征提取
        flat_images = image_sequence.view(-1, channels, height, width)
        features = self.feature_extractor(flat_images)

        # 2. 恢复时序维度
        feature_c, feature_h, feature_w = features.shape[1:]
        features = features.view(batch_size, seq_len, feature_c, feature_h, feature_w)

        # 3. 准备ODE输入
        current_input = features[:, -1:, :, :, :]
        observations = features

        if timestamps is None:
            timestamps = torch.linspace(0, seq_len-1, seq_len, dtype=torch.float32)

        return {
            'current_input': current_input,
            'observations': observations,
            'times': timestamps
        }


class PointCloudAdapter(InputAdapter):
    """点云数据适配器 - 将3D点云转换为2D特征图"""

    def __init__(self, grid_size=(64, 64), feature_channels=64,
                 x_range=(-50, 50), y_range=(-50, 50)):
        """
        Args:
            grid_size: BEV网格尺寸
            feature_channels: 特征通道数
            x_range: X轴范围
            y_range: Y轴范围
        """
        self.grid_size = grid_size
        self.feature_channels = feature_channels
        self.x_range = x_range
        self.y_range = y_range

        # 点云特征处理器
        self.point_processor = nn.Sequential(
            nn.Linear(4, 32),  # 假设输入是 [x, y, z, intensity]
            nn.ReLU(),
            nn.Linear(32, feature_channels),
            nn.ReLU()
        )

    def adapt_input(self, point_clouds, timestamps=None):
        """
        将点云序列转换为ODE输入格式

        Args:
            point_clouds: List[Tensor] 长度为T的点云列表，每个点云形状为[N, 4]
            timestamps: [T] 时间戳

        Returns:
            dict: 适配后的ODE输入
        """
        seq_len = len(point_clouds)
        batch_size = 1  # 假设单batch

        bev_features = []

        for pc in point_clouds:
            # 1. 点云特征提取
            point_features = self.point_processor(pc)  # [N, feature_channels]

            # 2. 转换为BEV网格
            bev_grid = self._points_to_bev_grid(pc[:, :2], point_features)
            bev_features.append(bev_grid)

        # 3. 堆叠为序列
        bev_sequence = torch.stack(bev_features, dim=1)  # [1, T, C, H, W]

        current_input = bev_sequence[:, -1:, :, :, :]
        observations = bev_sequence

        if timestamps is None:
            timestamps = torch.linspace(0, seq_len-1, seq_len, dtype=torch.float32)

        return {
            'current_input': current_input,
            'observations': observations,
            'times': timestamps
        }

    def _points_to_bev_grid(self, points_xy, features):
        """将点云投影到BEV网格"""
        h, w = self.grid_size

        # 归一化坐标到网格
        x_norm = (points_xy[:, 0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0])
        y_norm = (points_xy[:, 1] - self.y_range[0]) / (self.y_range[1] - self.y_range[0])

        # 转换为网格索引
        x_idx = (x_norm * (w - 1)).long().clamp(0, w - 1)
        y_idx = (y_norm * (h - 1)).long().clamp(0, h - 1)

        # 创建BEV网格
        bev_grid = torch.zeros(self.feature_channels, h, w, device=features.device)

        # 简单的最大池化聚合
        for i in range(len(points_xy)):
            bev_grid[:, y_idx[i], x_idx[i]] = torch.max(
                bev_grid[:, y_idx[i], x_idx[i]], features[i]
            )

        return bev_grid.unsqueeze(0)  # [1, C, H, W]


class MultiModalAdapter(InputAdapter):
    """多模态数据适配器"""

    def __init__(self, camera_adapter, lidar_adapter, fusion_channels=128):
        """
        Args:
            camera_adapter: 相机数据适配器
            lidar_adapter: 激光雷达数据适配器
            fusion_channels: 融合后的特征通道数
        """
        self.camera_adapter = camera_adapter
        self.lidar_adapter = lidar_adapter

        # 特征融合层
        total_channels = (camera_adapter.target_channels +
                         lidar_adapter.feature_channels)
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(total_channels, fusion_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(fusion_channels, fusion_channels, 3, padding=1),
            nn.ReLU()
        )

    def adapt_input(self, camera_data, lidar_data,
                   camera_timestamps=None, lidar_timestamps=None):
        """
        融合多模态数据

        Args:
            camera_data: 相机数据
            lidar_data: 激光雷达数据
            camera_timestamps: 相机时间戳
            lidar_timestamps: 激光雷达时间戳

        Returns:
            dict: 融合后的ODE输入
        """
        # 1. 分别适配各模态数据
        camera_adapted = self.camera_adapter.adapt_input(camera_data, camera_timestamps)
        lidar_adapted = self.lidar_adapter.adapt_input(lidar_data, lidar_timestamps)

        # 2. 时间对齐 (简化版本，实际中需要更复杂的对齐)
        # 这里假设两个模态的最后时刻对齐
        camera_current = camera_adapted['current_input']  # [B, 1, C1, H, W]
        lidar_current = lidar_adapted['current_input']    # [B, 1, C2, H, W]

        # 3. 特征融合
        fused_current = torch.cat([camera_current, lidar_current], dim=2)  # [B, 1, C1+C2, H, W]
        fused_current = self.fusion_layer(fused_current.squeeze(1)).unsqueeze(1)

        # 4. 为观测序列也进行融合 (简化处理)
        fused_obs = fused_current.repeat(1, max(len(camera_timestamps or [0]),
                                               len(lidar_timestamps or [0])), 1, 1, 1)

        return {
            'current_input': fused_current,
            'observations': fused_obs,
            'times': camera_timestamps or lidar_timestamps or torch.tensor([0.0]),
            'camera_data': camera_adapted,
            'lidar_data': lidar_adapted
        }


# 便捷的适配器工厂函数
def create_adapter(data_type, **kwargs):
    """
    根据数据类型创建适配器

    Args:
        data_type: 'timeseries', 'images', 'pointcloud', 'multimodal'
        **kwargs: 适配器参数

    Returns:
        适配器实例
    """
    adapters = {
        'timeseries': TimeSeriesAdapter,
        'images': ImageSequenceAdapter,
        'pointcloud': PointCloudAdapter,
        'multimodal': MultiModalAdapter
    }

    if data_type not in adapters:
        raise ValueError(f"不支持的数据类型: {data_type}")

    return adapters[data_type](**kwargs)


# 使用示例
if __name__ == "__main__":
    print("🔧 ODE模块输入适配器示例")

    # 时间序列示例
    ts_adapter = TimeSeriesAdapter(input_dim=10, output_channels=64)
    time_series = torch.randn(2, 5, 10)  # [batch, time, features]
    ts_adapted = ts_adapter.adapt_input(time_series)
    print(f"时间序列适配: {ts_adapted['current_input'].shape}")

    # 图像序列示例
    img_adapter = ImageSequenceAdapter(target_channels=64)
    images = torch.randn(1, 4, 3, 128, 128)  # [batch, time, channels, h, w]
    img_adapted = img_adapter.adapt_input(images)
    print(f"图像序列适配: {img_adapted['current_input'].shape}")

    print("✅ 适配器创建成功!")