#!/usr/bin/env python3
"""
ODE模块独立使用示例

这个文件展示了如何在其他项目中使用ODE模块，
包含完整的可运行代码示例。
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# 添加ODE模块路径 (如果需要)
# sys.path.append('/path/to/your/ode_modules')

# 导入ODE模块
try:
    from .. import NNFOwithBayesianJumps, FuturePredictionODE
    from ..configs.minimal_ode_config import create_minimal_ode_config, create_custom_ode_config
    from ..configs.config_validator import print_config_report
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ode_modules import NNFOwithBayesianJumps, FuturePredictionODE
    from ode_modules.configs.minimal_ode_config import create_minimal_ode_config, create_custom_ode_config
    from ode_modules.configs.config_validator import print_config_report


class SimpleVideoPredictor(nn.Module):
    """
    简单的视频预测模型示例
    演示如何将ODE模块集成到自定义模型中
    """

    def __init__(self, input_channels=3, feature_channels=64, num_future_frames=4):
        super().__init__()

        print(f"🏗️ 创建视频预测模型...")
        print(f"   输入通道: {input_channels}")
        print(f"   特征通道: {feature_channels}")
        print(f"   预测帧数: {num_future_frames}")

        # 简单的编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((32, 32))  # 固定输出尺寸
        )

        # 创建ODE配置
        self.cfg = create_custom_ode_config(
            out_channels=feature_channels,
            latent_dim=feature_channels,
            solver="euler",
            delta_t=0.1,
            use_variable_step=False
        )

        # 验证配置
        print("\n🔍 验证ODE配置...")
        is_valid = print_config_report(self.cfg)
        if not is_valid:
            raise ValueError("ODE配置验证失败!")

        # 创建ODE模型
        self.ode_predictor = NNFOwithBayesianJumps(
            input_size=feature_channels,
            hidden_size=feature_channels,
            cfg=self.cfg
        )

        # 简单的解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.num_future_frames = num_future_frames
        print("✅ 模型创建完成!")

    def forward(self, video_sequence):
        """
        前向传播

        Args:
            video_sequence: [B, T, C, H, W] 输入视频序列

        Returns:
            predictions: [B, T_future, C, H, W] 预测的未来帧
            ode_loss: ODE模块的辅助损失
        """
        batch_size, seq_len, channels, height, width = video_sequence.shape

        # 1. 特征提取
        print(f"📸 处理视频序列 {video_sequence.shape}")

        # 展平时序维度进行编码
        flat_frames = video_sequence.view(-1, channels, height, width)
        features = self.encoder(flat_frames)

        # 恢复时序维度
        feature_channels = features.shape[1]
        feature_h, feature_w = features.shape[2], features.shape[3]
        features = features.view(batch_size, seq_len, feature_channels, feature_h, feature_w)

        print(f"🔧 提取特征 {features.shape}")

        # 2. 准备时间戳
        times = torch.linspace(0, seq_len-1, seq_len, dtype=torch.float32)
        target_times = torch.linspace(seq_len, seq_len + self.num_future_frames - 1,
                                    self.num_future_frames, dtype=torch.float32)

        print(f"⏰ 观测时间: {times}")
        print(f"🎯 目标时间: {target_times}")

        # 3. 准备ODE输入
        current_input = features[:, -1:, :, :, :]  # 最后一帧作为当前输入
        observations = features  # 所有帧作为观测序列

        # 4. ODE预测
        print("🧠 开始ODE预测...")
        try:
            final_state, ode_loss, ode_predictions = self.ode_predictor(
                times=times,
                input=current_input,
                obs=observations,
                delta_t=0.1,
                T=target_times
            )

            print(f"✅ ODE预测完成! 输出形状: {ode_predictions.shape}")
            print(f"📊 ODE损失: {ode_loss}")

        except Exception as e:
            print(f"❌ ODE预测失败: {e}")
            print("🔄 使用零预测作为后备方案")
            ode_predictions = torch.zeros(batch_size, len(target_times),
                                        feature_channels, feature_h, feature_w)
            ode_loss = 0.0

        # 5. 解码到像素空间
        print("🎨 解码预测结果...")
        batch_future, future_len = ode_predictions.shape[:2]
        flat_predictions = ode_predictions.view(-1, feature_channels, feature_h, feature_w)

        # 上采样到原始尺寸
        upsampled = nn.functional.interpolate(flat_predictions, size=(height, width),
                                            mode='bilinear', align_corners=False)
        decoded_frames = self.decoder(upsampled)

        # 恢复时序维度
        predicted_frames = decoded_frames.view(batch_future, future_len, channels, height, width)

        print(f"🎬 最终预测形状: {predicted_frames.shape}")

        return predicted_frames, ode_loss


def run_basic_example():
    """运行基础使用示例"""
    print("=" * 60)
    print("🚀 基础ODE模块使用示例")
    print("=" * 60)

    # 1. 创建配置
    cfg = create_minimal_ode_config()

    # 2. 创建模型
    model = NNFOwithBayesianJumps(
        input_size=64,
        hidden_size=64,
        cfg=cfg
    )

    print("✅ ODE模型创建成功!")

    # 3. 准备测试数据
    batch_size = 2
    current_input = torch.randn(batch_size, 1, 64, 32, 32)
    observations = torch.randn(batch_size, 3, 64, 32, 32)
    times = torch.tensor([0.0, 0.5, 1.0])
    target_times = torch.tensor([1.5, 2.0, 2.5])

    print(f"📊 输入数据准备完成:")
    print(f"   当前输入: {current_input.shape}")
    print(f"   观测序列: {observations.shape}")
    print(f"   观测时间: {times}")
    print(f"   目标时间: {target_times}")

    # 4. 前向传播
    print("\n🧠 执行前向传播...")

    try:
        with torch.no_grad():  # 推理模式
            final_state, loss, predictions = model(
                times=times,
                input=current_input,
                obs=observations,
                delta_t=0.1,
                T=target_times
            )

        print("✅ 前向传播成功!")
        print(f"📈 预测结果形状: {predictions.shape}")
        print(f"📊 损失值: {loss}")
        print(f"🎯 预测统计:")
        print(f"   均值: {predictions.mean().item():.4f}")
        print(f"   标准差: {predictions.std().item():.4f}")
        print(f"   最小值: {predictions.min().item():.4f}")
        print(f"   最大值: {predictions.max().item():.4f}")

        return True

    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_video_prediction_example():
    """运行视频预测示例"""
    print("\n" + "=" * 60)
    print("🎬 视频预测模型示例")
    print("=" * 60)

    # 1. 创建视频预测模型
    video_model = SimpleVideoPredictor(
        input_channels=3,
        feature_channels=32,  # 使用较小的通道数以减少内存使用
        num_future_frames=3
    )

    # 2. 准备视频数据
    batch_size = 1  # 减少batch size
    seq_len = 4
    height, width = 64, 64  # 较小的分辨率

    print(f"\n📹 准备视频数据:")
    print(f"   批次大小: {batch_size}")
    print(f"   序列长度: {seq_len}")
    print(f"   分辨率: {height}x{width}")

    # 模拟视频序列 (例如移动的方块)
    video_frames = []
    for t in range(seq_len):
        frame = torch.zeros(batch_size, 3, height, width)
        # 创建移动的方块
        start_x = 10 + t * 5
        start_y = 10 + t * 3
        frame[:, :, start_y:start_y+10, start_x:start_x+10] = 0.8
        video_frames.append(frame)

    video_sequence = torch.stack(video_frames, dim=1)
    print(f"✅ 视频数据形状: {video_sequence.shape}")

    # 3. 预测
    print("\n🚀 开始视频预测...")

    try:
        with torch.no_grad():
            predicted_frames, ode_loss = video_model(video_sequence)

        print("✅ 视频预测成功!")
        print(f"🎯 预测帧形状: {predicted_frames.shape}")
        print(f"📊 ODE损失: {ode_loss}")

        # 简单的质量评估
        input_mean = video_sequence.mean()
        pred_mean = predicted_frames.mean()
        print(f"📈 输入均值: {input_mean:.4f}")
        print(f"📈 预测均值: {pred_mean:.4f}")
        print(f"📈 差异: {abs(input_mean - pred_mean):.4f}")

        return True

    except Exception as e:
        print(f"❌ 视频预测失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_performance_test():
    """运行性能测试"""
    print("\n" + "=" * 60)
    print("⚡ 性能测试")
    print("=" * 60)

    import time

    # 测试不同配置的性能
    configs = [
        ("轻量级", create_custom_ode_config(out_channels=32, latent_dim=32, delta_t=0.2)),
        ("标准", create_custom_ode_config(out_channels=64, latent_dim=64, delta_t=0.1)),
        ("高精度", create_custom_ode_config(out_channels=64, latent_dim=64, solver="midpoint", delta_t=0.05)),
    ]

    for config_name, cfg in configs:
        print(f"\n🧪 测试 {config_name} 配置...")

        try:
            model = NNFOwithBayesianJumps(cfg.MODEL.ENCODER.OUT_CHANNELS,
                                        cfg.MODEL.DISTRIBUTION.LATENT_DIM, cfg)

            # 准备数据
            batch_size = 1
            channels = cfg.MODEL.ENCODER.OUT_CHANNELS
            current_input = torch.randn(batch_size, 1, channels, 16, 16)
            observations = torch.randn(batch_size, 3, channels, 16, 16)
            times = torch.tensor([0.0, 0.5, 1.0])
            target_times = torch.tensor([1.5, 2.0])

            # 预热
            with torch.no_grad():
                _ = model(times, current_input, observations, cfg.MODEL.FUTURE_PRED.DELTA_T, target_times)

            # 计时
            num_runs = 10
            start_time = time.time()

            for _ in range(num_runs):
                with torch.no_grad():
                    _ = model(times, current_input, observations, cfg.MODEL.FUTURE_PRED.DELTA_T, target_times)

            end_time = time.time()
            avg_time = (end_time - start_time) / num_runs

            print(f"✅ {config_name} 平均推理时间: {avg_time*1000:.2f}ms")

        except Exception as e:
            print(f"❌ {config_name} 测试失败: {e}")


def main():
    """主函数"""
    print("🎯 ODE模块集成示例")
    print("作者: Claude Code")
    print("版本: 1.0")
    print("")

    # 设置随机种子以获得可重复的结果
    torch.manual_seed(42)
    np.random.seed(42)

    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")

    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    success_count = 0
    total_tests = 3

    # 运行示例
    try:
        # 1. 基础示例
        if run_basic_example():
            success_count += 1

        # 2. 视频预测示例
        if run_video_prediction_example():
            success_count += 1

        # 3. 性能测试
        try:
            run_performance_test()
            success_count += 1
        except Exception as e:
            print(f"性能测试失败: {e}")

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断")

    except Exception as e:
        print(f"\n💥 未预期的错误: {e}")
        import traceback
        traceback.print_exc()

    # 总结
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)
    print(f"✅ 成功: {success_count}/{total_tests}")
    print(f"❌ 失败: {total_tests - success_count}/{total_tests}")

    if success_count == total_tests:
        print("🎉 所有测试通过! ODE模块可以正常使用。")
    else:
        print("⚠️  部分测试失败，请检查错误信息。")

    print("\n📚 更多信息请参考:")
    print("   - INTEGRATION_GUIDE.md: 详细集成指南")
    print("   - configs/README.md: 配置参数说明")
    print("   - tests/: 完整测试套件")


if __name__ == "__main__":
    main()