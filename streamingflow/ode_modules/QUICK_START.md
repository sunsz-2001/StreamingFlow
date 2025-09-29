# ODE模块快速开始指南

## 🚀 3分钟快速上手

### 1. 复制模块到您的项目

```bash
# 将ODE模块复制到您的项目目录
cp -r /path/to/StreamingFlow/streamingflow/ode_modules /your/project/

# 您的项目结构现在应该包含:
your_project/
├── ode_modules/          # ODE模块
├── your_code.py          # 您的代码
└── ...
```

### 2. 最简单的使用示例

```python
import torch
from ode_modules import NNFOwithBayesianJumps
from ode_modules.configs.minimal_ode_config import create_minimal_ode_config

# 1. 创建配置
cfg = create_minimal_ode_config()

# 2. 创建模型
ode_model = NNFOwithBayesianJumps(
    input_size=64,    # 输入特征通道数
    hidden_size=64,   # 隐藏层维度
    cfg=cfg
)

# 3. 准备输入数据 (示例)
batch_size = 2
current_input = torch.randn(batch_size, 1, 64, 32, 32)     # [B, 1, C, H, W]
observations = torch.randn(batch_size, 3, 64, 32, 32)      # [B, T, C, H, W]
times = torch.tensor([0.0, 0.5, 1.0])                      # 观测时间
target_times = torch.tensor([1.5, 2.0, 2.5])              # 预测时间

# 4. 预测未来
with torch.no_grad():
    final_state, loss, predictions = ode_model(
        times=times,
        input=current_input,
        obs=observations,
        delta_t=0.1,
        T=target_times
    )

print(f"预测结果形状: {predictions.shape}")  # [2, 3, 64, 32, 32]
```

### 3. 验证安装

```bash
# 运行验证脚本确保一切正常
cd ode_modules/examples
python validation_script.py
```

## 📋 核心概念

### 数据格式要求

| 参数 | 形状 | 说明 |
|------|------|------|
| `input` | `[B, 1, C, H, W]` | 当前时刻的输入特征 |
| `obs` | `[B, T, C, H, W]` | 历史观测序列 |
| `times` | `[T]` | 观测时间戳 (升序) |
| `T` | `[T_future]` | 目标预测时间戳 |

### 输出说明

| 输出 | 形状 | 说明 |
|------|------|------|
| `final_state` | `[B, C, H, W]` | 最终隐状态 |
| `loss` | `scalar` | ODE辅助损失 |
| `predictions` | `[B, T_future, C, H, W]` | 预测结果 |

## ⚙️ 常用配置

### 轻量级配置 (快速测试)

```python
from ode_modules.configs.minimal_ode_config import create_custom_ode_config

cfg = create_custom_ode_config(
    out_channels=32,          # 较小的特征维度
    latent_dim=32,
    delta_t=0.1              # 较大的时间步
)
```

### 高精度配置 (研究用途)

```python
cfg = create_custom_ode_config(
    out_channels=128,         # 更大的特征维度
    latent_dim=128,
    solver="midpoint",        # 更精确的求解器
    delta_t=0.02,            # 更小的时间步
    use_variable_step=True    # 自适应步长
)
```

## 🔧 集成到现有模型

```python
import torch.nn as nn
from ode_modules import NNFOwithBayesianJumps
from ode_modules.configs.minimal_ode_config import create_minimal_ode_config

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 您的特征提取器
        self.encoder = nn.Conv2d(3, 64, 3, padding=1)

        # ODE模块
        cfg = create_minimal_ode_config()
        self.ode_predictor = NNFOwithBayesianJumps(64, 64, cfg)

        # 您的输出层
        self.decoder = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, video_sequence):
        # 特征提取
        features = self.encoder(video_sequence)

        # ODE预测 (处理您的时序逻辑)
        # ... 准备times, input, obs, target_times

        final_state, loss, predictions = self.ode_predictor(
            times, input, obs, 0.1, target_times
        )

        # 解码输出
        output = self.decoder(predictions)
        return output, loss
```

## 🆘 常见问题

### Q: 形状不匹配错误？
**A:** 检查输入张量形状，确保符合 `[B, T, C, H, W]` 格式。

### Q: 内存不足？
**A:** 使用较小的 `out_channels` 或减少 `batch_size`。

### Q: 导入失败？
**A:** 确保 `ode_modules` 文件夹在正确位置，或运行验证脚本检查。

### Q: 数值不稳定？
**A:** 尝试使用更小的 `delta_t` 或不同的 `solver`。

## 📚 更多资源

- **详细指南**: [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
- **配置文档**: [configs/README.md](configs/README.md)
- **完整示例**: [examples/standalone_example.py](examples/standalone_example.py)
- **测试套件**: [tests/](tests/)

## 🎯 下一步

1. **运行验证脚本**确保模块工作正常
2. **查看完整示例**了解更多用法
3. **阅读配置文档**优化性能
4. **集成到您的项目**中开始使用！

---

**🎉 恭喜！您已经成功集成ODE模块到您的项目中！**