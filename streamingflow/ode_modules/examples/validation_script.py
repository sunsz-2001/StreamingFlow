#!/usr/bin/env python3
"""
ODE模块验证脚本

这个脚本用于验证ODE模块在新环境中的正确性和兼容性。
运行此脚本可以快速检查ODE模块是否能正常工作。
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time
import warnings
from typing import Tuple, Dict, Any

# 抑制警告以获得更清晰的输出
warnings.filterwarnings('ignore')

def check_dependencies():
    """检查依赖项"""
    print("🔍 检查依赖项...")

    required_packages = {
        'torch': torch.__version__,
        'numpy': np.__version__,
    }

    print("✅ 依赖项检查:")
    for package, version in required_packages.items():
        print(f"   {package}: {version}")

    # 检查PyTorch功能
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA设备数: {torch.cuda.device_count()}")
        print(f"   当前设备: {torch.cuda.current_device()}")

    return True


def test_ode_import():
    """测试ODE模块导入"""
    print("\n📦 测试ODE模块导入...")

    try:
        # 尝试导入核心模块
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        from ode_modules import NNFOwithBayesianJumps, FuturePredictionODE
        from ode_modules.configs.minimal_ode_config import create_minimal_ode_config
        from ode_modules.configs.config_validator import validate_ode_config

        print("✅ 核心模块导入成功")

        # 测试配置创建
        cfg = create_minimal_ode_config()
        is_valid, missing, warnings_list = validate_ode_config(cfg)

        if is_valid:
            print("✅ 配置验证通过")
        else:
            print(f"❌ 配置验证失败: {missing}")
            return False

        # 测试模型创建
        model1 = NNFOwithBayesianJumps(input_size=64, hidden_size=64, cfg=cfg)
        model2 = FuturePredictionODE(in_channels=64, latent_dim=64, cfg=cfg)

        print("✅ 模型创建成功")

        return True, (NNFOwithBayesianJumps, FuturePredictionODE, create_minimal_ode_config)

    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("   请检查ODE模块路径是否正确")
        return False, None

    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return False, None


def test_forward_pass(NNFOwithBayesianJumps, create_minimal_ode_config):
    """测试前向传播"""
    print("\n🧠 测试前向传播...")

    try:
        # 创建模型
        cfg = create_minimal_ode_config()
        model = NNFOwithBayesianJumps(input_size=64, hidden_size=64, cfg=cfg)
        model.eval()

        # 准备测试数据
        batch_size = 2
        current_input = torch.randn(batch_size, 1, 64, 32, 32)
        observations = torch.randn(batch_size, 3, 64, 32, 32)
        times = torch.tensor([0.0, 0.5, 1.0])
        target_times = torch.tensor([1.5, 2.0])

        print(f"   输入形状: {current_input.shape}")
        print(f"   观测形状: {observations.shape}")

        # 前向传播
        with torch.no_grad():
            start_time = time.time()
            final_state, loss, predictions = model(
                times=times,
                input=current_input,
                obs=observations,
                delta_t=0.1,
                T=target_times
            )
            end_time = time.time()

        # 验证输出
        expected_pred_shape = (batch_size, len(target_times), 64, 32, 32)
        if predictions.shape == expected_pred_shape:
            print(f"✅ 输出形状正确: {predictions.shape}")
        else:
            print(f"❌ 输出形状错误: 期望 {expected_pred_shape}, 得到 {predictions.shape}")
            return False

        # 检查数值稳定性
        if torch.isfinite(predictions).all():
            print("✅ 输出数值稳定")
        else:
            print("❌ 输出包含NaN或Inf")
            return False

        print(f"✅ 前向传播耗时: {(end_time - start_time)*1000:.2f}ms")

        return True

    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow(NNFOwithBayesianJumps, create_minimal_ode_config):
    """测试梯度流"""
    print("\n🔄 测试梯度流...")

    try:
        # 创建模型
        cfg = create_minimal_ode_config()
        model = NNFOwithBayesianJumps(input_size=32, hidden_size=32, cfg=cfg)  # 小模型加快测试
        model.train()

        # 创建优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # 准备数据
        batch_size = 1
        current_input = torch.randn(batch_size, 1, 32, 16, 16, requires_grad=True)
        observations = torch.randn(batch_size, 2, 32, 16, 16)
        times = torch.tensor([0.0, 0.5])
        target_times = torch.tensor([1.0])

        # 前向传播
        final_state, ode_loss, predictions = model(
            times=times,
            input=current_input,
            obs=observations,
            delta_t=0.1,
            T=target_times
        )

        # 计算损失
        target = torch.randn_like(predictions)
        loss = nn.MSELoss()(predictions, target) + ode_loss

        print(f"   总损失: {loss.item():.6f}")

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 检查梯度
        total_grad_norm = 0
        param_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm
                param_count += 1
            else:
                print(f"⚠️ 参数 {name} 没有梯度")

        avg_grad_norm = total_grad_norm / param_count if param_count > 0 else 0

        print(f"   参数数量: {param_count}")
        print(f"   平均梯度范数: {avg_grad_norm:.6f}")

        if avg_grad_norm > 0:
            print("✅ 梯度流正常")
        else:
            print("❌ 梯度流异常")
            return False

        # 优化器步骤
        optimizer.step()
        print("✅ 优化器更新成功")

        return True

    except Exception as e:
        print(f"❌ 梯度测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_usage(NNFOwithBayesianJumps, create_minimal_ode_config):
    """测试内存使用"""
    print("\n💾 测试内存使用...")

    try:
        if not torch.cuda.is_available():
            print("   跳过GPU内存测试 (CUDA不可用)")
            return True

        # 清空GPU缓存
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        # 创建模型
        cfg = create_minimal_ode_config()
        model = NNFOwithBayesianJumps(input_size=64, hidden_size=64, cfg=cfg).cuda()

        model_memory = torch.cuda.memory_allocated() - initial_memory
        print(f"   模型内存: {model_memory / 1e6:.2f} MB")

        # 测试批次大小
        batch_sizes = [1, 2, 4]
        for batch_size in batch_sizes:
            torch.cuda.empty_cache()
            start_memory = torch.cuda.memory_allocated()

            # 准备数据
            current_input = torch.randn(batch_size, 1, 64, 32, 32).cuda()
            observations = torch.randn(batch_size, 3, 64, 32, 32).cuda()
            times = torch.tensor([0.0, 0.5, 1.0])
            target_times = torch.tensor([1.5, 2.0])

            # 前向传播
            with torch.no_grad():
                final_state, loss, predictions = model(
                    times=times,
                    input=current_input,
                    obs=observations,
                    delta_t=0.1,
                    T=target_times
                )

            peak_memory = torch.cuda.max_memory_allocated() - start_memory
            print(f"   批次大小 {batch_size}: {peak_memory / 1e6:.2f} MB")

            # 重置内存统计
            torch.cuda.reset_peak_memory_stats()

        print("✅ 内存使用测试完成")
        return True

    except Exception as e:
        print(f"❌ 内存测试失败: {e}")
        return False


def test_different_configs(NNFOwithBayesianJumps, create_minimal_ode_config):
    """测试不同配置"""
    print("\n⚙️ 测试不同配置...")

    # 导入自定义配置函数
    try:
        from ode_modules.configs.minimal_ode_config import create_custom_ode_config
    except ImportError:
        print("   跳过配置测试 (无法导入create_custom_ode_config)")
        return True

    configs = [
        ("最小配置", create_minimal_ode_config()),
        ("小模型", create_custom_ode_config(out_channels=32, latent_dim=32)),
        ("标准模型", create_custom_ode_config(out_channels=64, latent_dim=64)),
        ("中点求解器", create_custom_ode_config(out_channels=32, latent_dim=32, solver="midpoint")),
    ]

    for config_name, cfg in configs:
        try:
            print(f"   测试 {config_name}...")

            model = NNFOwithBayesianJumps(
                input_size=cfg.MODEL.ENCODER.OUT_CHANNELS,
                hidden_size=cfg.MODEL.DISTRIBUTION.LATENT_DIM,
                cfg=cfg
            )

            # 快速前向传播测试
            batch_size = 1
            channels = cfg.MODEL.ENCODER.OUT_CHANNELS
            current_input = torch.randn(batch_size, 1, channels, 16, 16)
            observations = torch.randn(batch_size, 2, channels, 16, 16)
            times = torch.tensor([0.0, 0.5])
            target_times = torch.tensor([1.0])

            with torch.no_grad():
                final_state, loss, predictions = model(
                    times=times,
                    input=current_input,
                    obs=observations,
                    delta_t=0.1,
                    T=target_times
                )

            print(f"     ✅ {config_name} 工作正常")

        except Exception as e:
            print(f"     ❌ {config_name} 失败: {e}")
            return False

    print("✅ 配置测试完成")
    return True


def run_comprehensive_validation():
    """运行综合验证"""
    print("🎯 ODE模块综合验证")
    print("=" * 50)

    results = {}

    # 1. 检查依赖
    results['dependencies'] = check_dependencies()

    # 2. 测试导入
    import_result, modules = test_ode_import()
    results['import'] = import_result

    if not import_result:
        print("\n❌ 导入失败，无法继续测试")
        return results

    NNFOwithBayesianJumps, FuturePredictionODE, create_minimal_ode_config = modules

    # 3. 测试前向传播
    results['forward_pass'] = test_forward_pass(NNFOwithBayesianJumps, create_minimal_ode_config)

    # 4. 测试梯度流
    results['gradient_flow'] = test_gradient_flow(NNFOwithBayesianJumps, create_minimal_ode_config)

    # 5. 测试内存使用
    results['memory_usage'] = test_memory_usage(NNFOwithBayesianJumps, create_minimal_ode_config)

    # 6. 测试不同配置
    results['different_configs'] = test_different_configs(NNFOwithBayesianJumps, create_minimal_ode_config)

    return results


def print_summary(results: Dict[str, bool]):
    """打印总结报告"""
    print("\n" + "=" * 50)
    print("📊 验证结果总结")
    print("=" * 50)

    total_tests = len(results)
    passed_tests = sum(results.values())

    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name:20} {status}")

    print("-" * 50)
    print(f"总计: {passed_tests}/{total_tests} 测试通过")

    if passed_tests == total_tests:
        print("\n🎉 所有验证通过! ODE模块可以正常使用。")
        print("✅ 您可以安全地在项目中集成ODE模块。")
    else:
        print(f"\n⚠️ {total_tests - passed_tests} 个测试失败。")
        print("❌ 请检查失败的测试并解决问题后再使用。")

    print("\n📚 使用指南:")
    print("   - 查看 INTEGRATION_GUIDE.md 了解集成方法")
    print("   - 查看 examples/standalone_example.py 了解使用示例")
    print("   - 查看 configs/README.md 了解配置选项")


def main():
    """主函数"""
    print("🔬 ODE模块验证脚本")
    print("版本: 1.0")
    print("目的: 验证ODE模块在新环境中的兼容性")
    print("")

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        results = run_comprehensive_validation()
        print_summary(results)

        # 返回退出码
        if all(results.values()):
            return 0  # 成功
        else:
            return 1  # 失败

    except KeyboardInterrupt:
        print("\n⏹️ 验证被用户中断")
        return 2

    except Exception as e:
        print(f"\n💥 验证过程中发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)