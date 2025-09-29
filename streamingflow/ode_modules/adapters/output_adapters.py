"""
ODE模块输出适配器

这个文件包含将ODE模块输出转换为各种目标格式的函数。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OutputAdapter:
    """输出适配器基类"""

    @staticmethod
    def adapt_output(ode_output, target_format):
        """适配ODE输出到目标格式"""
        raise NotImplementedError


class TimeSeriesOutputAdapter(OutputAdapter):
    """时间序列输出适配器 - 将2D特征图转换回1D时间序列"""

    def __init__(self, feature_channels, feature_size, output_dim):
        """
        Args:
            feature_channels: ODE输出的特征通道数
            feature_size: 特征图空间尺寸
            output_dim: 目标时间序列维度
        """
        self.feature_channels = feature_channels
        self.feature_size = feature_size
        self.output_dim = output_dim

        # 逆映射网络
        self.inverse_mapper = nn.Sequential(
            nn.Linear(feature_channels * feature_size * feature_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )

    def adapt_output(self, ode_predictions, original_timestamps=None):
        """
        将ODE预测转换为时间序列

        Args:
            ode_predictions: [B, T_future, C, H, W] ODE预测结果
            original_timestamps: [T_future] 预测时间戳

        Returns:
            dict: 包含适配后的输出
        """
        batch_size, seq_len, channels, height, width = ode_predictions.shape

        # 1. 展平特征图
        flat_features = ode_predictions.view(batch_size * seq_len, -1)

        # 2. 映射回时间序列
        time_series = self.inverse_mapper(flat_features)

        # 3. 恢复时序维度
        time_series = time_series.view(batch_size, seq_len, self.output_dim)

        result = {
            'predictions': time_series,  # [B, T_future, output_dim]
            'timestamps': original_timestamps
        }

        # 4. 计算预测统计
        result['statistics'] = {
            'mean': time_series.mean(dim=(0, 1)),
            'std': time_series.std(dim=(0, 1)),
            'min': time_series.min(dim=1)[0].min(dim=0)[0],
            'max': time_series.max(dim=1)[0].max(dim=0)[0]
        }

        return result


class ImageSequenceOutputAdapter(OutputAdapter):
    """图像序列输出适配器 - 将特征图转换回图像"""

    def __init__(self, feature_channels, target_size=(128, 128), target_channels=3):
        """
        Args:
            feature_channels: ODE输出的特征通道数
            target_size: 目标图像尺寸
            target_channels: 目标图像通道数 (如RGB=3)
        """
        self.feature_channels = feature_channels
        self.target_size = target_size
        self.target_channels = target_channels

        # 图像重建网络
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, target_channels, 3, padding=1),
            nn.Sigmoid()  # 假设输出在[0,1]范围
        )

    def adapt_output(self, ode_predictions, original_timestamps=None):
        """
        将ODE预测转换为图像序列

        Args:
            ode_predictions: [B, T_future, C, H, W] ODE预测结果
            original_timestamps: [T_future] 预测时间戳

        Returns:
            dict: 包含适配后的输出
        """
        batch_size, seq_len = ode_predictions.shape[:2]

        # 1. 展平时序维度
        flat_features = ode_predictions.view(-1, *ode_predictions.shape[2:])

        # 2. 上采样到目标尺寸
        if flat_features.shape[-2:] != self.target_size:
            upsampled = F.interpolate(flat_features, size=self.target_size,
                                    mode='bilinear', align_corners=False)
        else:
            upsampled = flat_features

        # 3. 解码为图像
        decoded_images = self.decoder(upsampled)

        # 4. 恢复时序维度
        image_sequence = decoded_images.view(batch_size, seq_len,
                                           self.target_channels, *self.target_size)

        result = {
            'predictions': image_sequence,  # [B, T_future, C, H, W]
            'timestamps': original_timestamps
        }

        # 5. 计算图像质量指标
        result['quality_metrics'] = {
            'mean_intensity': image_sequence.mean(),
            'std_intensity': image_sequence.std(),
            'spatial_gradient': self._compute_spatial_gradient(image_sequence)
        }

        return result

    def _compute_spatial_gradient(self, images):
        """计算空间梯度作为图像清晰度指标"""
        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                              dtype=images.dtype, device=images.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                              dtype=images.dtype, device=images.device).view(1, 1, 3, 3)

        # 对每个通道计算梯度
        grad_x = F.conv2d(images.view(-1, 1, *images.shape[-2:]), sobel_x, padding=1)
        grad_y = F.conv2d(images.view(-1, 1, *images.shape[-2:]), sobel_y, padding=1)

        # 梯度幅度
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        return gradient_magnitude.mean()


class PointCloudOutputAdapter(OutputAdapter):
    """点云输出适配器 - 将BEV特征图转换为点云"""

    def __init__(self, feature_channels, grid_size=(64, 64),
                 x_range=(-50, 50), y_range=(-50, 50), max_points=1000):
        """
        Args:
            feature_channels: 输入特征通道数
            grid_size: BEV网格尺寸
            x_range: X轴范围
            y_range: Y轴范围
            max_points: 最大点数
        """
        self.feature_channels = feature_channels
        self.grid_size = grid_size
        self.x_range = x_range
        self.y_range = y_range
        self.max_points = max_points

        # 点云生成网络
        self.point_generator = nn.Sequential(
            nn.Conv2d(feature_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 4, 1),  # 输出 [x, y, z, intensity]
            nn.Tanh()  # 归一化输出
        )

    def adapt_output(self, ode_predictions, original_timestamps=None):
        """
        将ODE预测转换为点云序列

        Args:
            ode_predictions: [B, T_future, C, H, W] ODE预测结果
            original_timestamps: [T_future] 预测时间戳

        Returns:
            dict: 包含适配后的输出
        """
        batch_size, seq_len = ode_predictions.shape[:2]

        point_clouds = []

        for t in range(seq_len):
            batch_points = []

            for b in range(batch_size):
                # 1. 生成点云参数
                bev_features = ode_predictions[b, t]  # [C, H, W]
                point_params = self.point_generator(bev_features.unsqueeze(0))  # [1, 4, H, W]
                point_params = point_params.squeeze(0)  # [4, H, W]

                # 2. 转换为点云
                points = self._bev_to_points(point_params)
                batch_points.append(points)

            point_clouds.append(batch_points)

        result = {
            'predictions': point_clouds,  # List[List[Tensor]] [T_future][B][N, 4]
            'timestamps': original_timestamps
        }

        # 3. 计算点云统计
        result['statistics'] = self._compute_pointcloud_stats(point_clouds)

        return result

    def _bev_to_points(self, point_params):
        """将BEV特征转换为点云"""
        _, h, w = point_params.shape

        # 创建网格坐标
        y_indices, x_indices = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        y_indices = y_indices.to(point_params.device).float()
        x_indices = x_indices.to(point_params.device).float()

        # 转换为世界坐标
        x_world = (x_indices / (w - 1)) * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
        y_world = (y_indices / (h - 1)) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]

        # 提取点云属性
        x_offset = point_params[0] * 2.0  # 位置偏移
        y_offset = point_params[1] * 2.0
        z_height = point_params[2] * 5.0  # 高度
        intensity = torch.sigmoid(point_params[3])  # 强度

        # 应用偏移
        x_final = x_world + x_offset
        y_final = y_world + y_offset

        # 组装点云
        points = torch.stack([
            x_final.flatten(),
            y_final.flatten(),
            z_height.flatten(),
            intensity.flatten()
        ], dim=1)  # [H*W, 4]

        # 过滤有效点 (基于强度阈值)
        valid_mask = intensity.flatten() > 0.1
        valid_points = points[valid_mask]

        # 限制点数
        if len(valid_points) > self.max_points:
            indices = torch.randperm(len(valid_points))[:self.max_points]
            valid_points = valid_points[indices]

        return valid_points

    def _compute_pointcloud_stats(self, point_clouds):
        """计算点云统计信息"""
        all_points = []
        for t_points in point_clouds:
            for b_points in t_points:
                all_points.append(b_points)

        if not all_points:
            return {}

        concatenated = torch.cat(all_points, dim=0)

        return {
            'total_points': len(concatenated),
            'mean_position': concatenated[:, :3].mean(dim=0),
            'std_position': concatenated[:, :3].std(dim=0),
            'mean_intensity': concatenated[:, 3].mean(),
            'point_density': len(concatenated) / len(all_points)
        }


class SegmentationOutputAdapter(OutputAdapter):
    """语义分割输出适配器"""

    def __init__(self, feature_channels, num_classes, target_size=(256, 256)):
        """
        Args:
            feature_channels: 输入特征通道数
            num_classes: 分割类别数
            target_size: 目标分割图尺寸
        """
        self.feature_channels = feature_channels
        self.num_classes = num_classes
        self.target_size = target_size

        # 分割头
        self.seg_head = nn.Sequential(
            nn.Conv2d(feature_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1)
        )

    def adapt_output(self, ode_predictions, original_timestamps=None):
        """
        将ODE预测转换为分割图序列

        Args:
            ode_predictions: [B, T_future, C, H, W] ODE预测结果
            original_timestamps: [T_future] 预测时间戳

        Returns:
            dict: 包含适配后的输出
        """
        batch_size, seq_len = ode_predictions.shape[:2]

        # 1. 展平时序维度
        flat_features = ode_predictions.view(-1, *ode_predictions.shape[2:])

        # 2. 上采样到目标尺寸
        if flat_features.shape[-2:] != self.target_size:
            upsampled = F.interpolate(flat_features, size=self.target_size,
                                    mode='bilinear', align_corners=False)
        else:
            upsampled = flat_features

        # 3. 生成分割图
        seg_logits = self.seg_head(upsampled)

        # 4. 恢复时序维度
        seg_logits = seg_logits.view(batch_size, seq_len, self.num_classes, *self.target_size)

        # 5. 计算概率和预测类别
        seg_probs = F.softmax(seg_logits, dim=2)
        seg_predictions = torch.argmax(seg_logits, dim=2)

        result = {
            'logits': seg_logits,          # [B, T_future, num_classes, H, W]
            'probabilities': seg_probs,     # [B, T_future, num_classes, H, W]
            'predictions': seg_predictions, # [B, T_future, H, W]
            'timestamps': original_timestamps
        }

        # 6. 计算分割质量指标
        result['quality_metrics'] = {
            'confidence': seg_probs.max(dim=2)[0].mean(),
            'entropy': self._compute_entropy(seg_probs),
            'class_distribution': self._compute_class_distribution(seg_predictions)
        }

        return result

    def _compute_entropy(self, probs):
        """计算预测熵 (不确定性指标)"""
        log_probs = torch.log(probs + 1e-8)
        entropy = -(probs * log_probs).sum(dim=2)
        return entropy.mean()

    def _compute_class_distribution(self, predictions):
        """计算类别分布"""
        class_counts = torch.zeros(self.num_classes, device=predictions.device)
        for c in range(self.num_classes):
            class_counts[c] = (predictions == c).sum().float()
        return class_counts / predictions.numel()


# 便捷的适配器工厂函数
def create_output_adapter(output_type, **kwargs):
    """
    根据输出类型创建适配器

    Args:
        output_type: 'timeseries', 'images', 'pointcloud', 'segmentation'
        **kwargs: 适配器参数

    Returns:
        适配器实例
    """
    adapters = {
        'timeseries': TimeSeriesOutputAdapter,
        'images': ImageSequenceOutputAdapter,
        'pointcloud': PointCloudOutputAdapter,
        'segmentation': SegmentationOutputAdapter
    }

    if output_type not in adapters:
        raise ValueError(f"不支持的输出类型: {output_type}")

    return adapters[output_type](**kwargs)


# 使用示例
if __name__ == "__main__":
    print("🎯 ODE模块输出适配器示例")

    # 模拟ODE输出
    ode_output = torch.randn(2, 3, 64, 32, 32)  # [B, T_future, C, H, W]

    # 时间序列适配
    ts_adapter = TimeSeriesOutputAdapter(
        feature_channels=64, feature_size=32, output_dim=10
    )
    ts_result = ts_adapter.adapt_output(ode_output)
    print(f"时间序列输出: {ts_result['predictions'].shape}")

    # 图像序列适配
    img_adapter = ImageSequenceOutputAdapter(
        feature_channels=64, target_size=(128, 128)
    )
    img_result = img_adapter.adapt_output(ode_output)
    print(f"图像序列输出: {img_result['predictions'].shape}")

    print("✅ 输出适配器创建成功!")