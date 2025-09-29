# ODE模块输入输出修改指南

## 🎯 修改位置总览

### **核心原则: 不修改ODE模块本身，而是在外部进行适配**

```
您的数据 → 输入适配器 → ODE模块 → 输出适配器 → 您的目标格式
    ↓            ↓           ↓          ↓           ↓
  任意格式    标准格式    内部处理    标准格式     任意格式
```

---

## 📍 **1. 修改位置详解**

### **🔧 A. 输入适配 (在调用ODE之前)**

**位置**: `ode_modules/adapters/input_adapters.py`

**目的**: 将您的数据格式转换为ODE模块要求的标准格式

**ODE要求的标准输入格式**:
```python
{
    'input': [B, 1, C, H, W],        # 当前时刻输入
    'obs': [B, T, C, H, W],          # 历史观测序列
    'times': [T],                    # 观测时间戳
    'T': [T_future],                 # 预测时间戳
    'delta_t': float                 # 积分步长
}
```

### **🎯 B. 输出适配 (在ODE输出之后)**

**位置**: `ode_modules/adapters/output_adapters.py`

**目的**: 将ODE的标准输出转换为您需要的格式

**ODE的标准输出格式**:
```python
{
    'final_state': [B, C, H, W],           # 最终状态
    'loss': scalar,                        # 辅助损失
    'predictions': [B, T_future, C, H, W]  # 预测结果
}
```

---

## 🛠️ **2. 具体修改方法**

### **方法1: 使用现有适配器 (推荐)**

```python
from ode_modules.adapters import TimeSeriesODEWrapper

# 创建适配的ODE模型
model = TimeSeriesODEWrapper(
    input_dim=10,      # 您的时间序列维度
    output_dim=10,     # 目标输出维度
    hidden_dim=64      # ODE内部维度
)

# 直接使用您的数据格式
time_series = torch.randn(2, 5, 10)  # [batch, time, features]
result = model(time_series)
predictions = result['predictions']   # [batch, future_time, features]
```

### **方法2: 自定义适配器**

#### **Step 1: 创建输入适配器**

```python
class MyInputAdapter:
    def adapt_input(self, my_data, timestamps=None):
        # 将my_data转换为ODE格式

        # 示例: 处理您的自定义数据
        current_input = self.convert_to_ode_format(my_data)  # [B, 1, C, H, W]
        observations = self.create_observation_sequence(my_data)  # [B, T, C, H, W]
        times = timestamps or self.generate_timestamps(my_data)

        return {
            'current_input': current_input,
            'observations': observations,
            'times': times
        }
```

#### **Step 2: 创建输出适配器**

```python
class MyOutputAdapter:
    def adapt_output(self, ode_predictions, timestamps=None):
        # 将ODE输出转换为您的目标格式

        # ode_predictions: [B, T_future, C, H, W]
        my_format = self.convert_from_ode_format(ode_predictions)

        return {
            'predictions': my_format,
            'timestamps': timestamps,
            'extra_info': self.compute_extra_metrics(my_format)
        }
```

#### **Step 3: 组合使用**

```python
from ode_modules.adapters import CustomODEWrapper

# 创建自定义包装器
wrapper = CustomODEWrapper(
    input_adapter=MyInputAdapter(),
    output_adapter=MyOutputAdapter(),
    out_channels=64,
    latent_dim=64
)

# 使用
result = wrapper(my_custom_data)
```

### **方法3: 直接修改数据格式**

如果您不想使用适配器，可以直接准备符合ODE要求的数据:

```python
from ode_modules import NNFOwithBayesianJumps
from ode_modules.configs.minimal_ode_config import create_minimal_ode_config

# 1. 手动转换数据格式
def convert_my_data_to_ode_format(my_data):
    # 您的转换逻辑
    current_input = ...  # [B, 1, C, H, W]
    observations = ...   # [B, T, C, H, W]
    times = ...         # [T]
    target_times = ...  # [T_future]
    return current_input, observations, times, target_times

# 2. 创建ODE模型
cfg = create_minimal_ode_config()
ode_model = NNFOwithBayesianJumps(64, 64, cfg)

# 3. 转换和预测
current_input, observations, times, target_times = convert_my_data_to_ode_format(my_data)
final_state, loss, predictions = ode_model(times, current_input, observations, 0.1, target_times)

# 4. 转换输出格式
my_result = convert_ode_output_to_my_format(predictions)
```

---

## 📋 **3. 常见数据类型的修改示例**

### **A. 时间序列数据**

**原始数据**: `[B, T, Features]` 1D时间序列

**修改位置**:
```python
# 输入适配 (adapters/input_adapters.py:TimeSeriesAdapter)
features_2d = self.mapper(flat_series)  # 1D → 2D映射
features_2d = features_2d.view(B, T, C, H, W)  # 重塑为2D特征图

# 输出适配 (adapters/output_adapters.py:TimeSeriesOutputAdapter)
time_series = self.inverse_mapper(flat_features)  # 2D → 1D映射
```

### **B. 图像/视频数据**

**原始数据**: `[B, T, C, H, W]` 图像序列

**修改位置**:
```python
# 输入适配 (adapters/input_adapters.py:ImageSequenceAdapter)
features = self.feature_extractor(flat_images)  # 特征提取
features = nn.AdaptiveAvgPool2d(target_size)(features)  # 尺寸调整

# 输出适配 (adapters/output_adapters.py:ImageSequenceOutputAdapter)
upsampled = F.interpolate(features, size=target_size)  # 上采样
decoded = self.decoder(upsampled)  # 解码为图像
```

### **C. 点云数据**

**原始数据**: `List[Tensor]` 点云序列，每个为 `[N, 3]`

**修改位置**:
```python
# 输入适配 (adapters/input_adapters.py:PointCloudAdapter)
bev_grid = self._points_to_bev_grid(points)  # 点云 → BEV网格

# 输出适配 (adapters/output_adapters.py:PointCloudOutputAdapter)
points = self._bev_to_points(bev_features)   # BEV网格 → 点云
```

### **D. 多模态数据**

**原始数据**: 相机 + 激光雷达等多种传感器

**修改位置**:
```python
# 输入适配 (adapters/input_adapters.py:MultiModalAdapter)
fused_features = self.fusion_layer(torch.cat([camera_features, lidar_features]))

# 输出适配: 可以分别解码为各模态或保持融合格式
```

---

## ⚡ **4. 快速开始模板**

### **模板1: 快速适配您的数据**

```python
# 1. 选择最接近的包装器
from ode_modules.adapters import create_ode_wrapper

# 根据您的数据选择:
# - 'timeseries': 时间序列数据
# - 'video': 视频/图像序列
# - 'lidar': 点云数据
# - 'segmentation': 语义分割

wrapper = create_ode_wrapper('timeseries', input_dim=YOUR_DIM, output_dim=YOUR_DIM)

# 2. 直接使用
result = wrapper(your_data)
predictions = result['predictions']
```

### **模板2: 最小修改集成**

```python
# 如果您已有模型，最小改动集成ODE:

class YourExistingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.your_encoder = YourEncoder()

        # 添加ODE模块
        from ode_modules.adapters import VideoODEWrapper
        self.ode_predictor = VideoODEWrapper(feature_channels=64)

        self.your_decoder = YourDecoder()

    def forward(self, input_data):
        # 您的预处理
        features = self.your_encoder(input_data)

        # ODE预测 (自动处理格式转换)
        ode_result = self.ode_predictor(features)
        future_features = ode_result['predictions']

        # 您的后处理
        output = self.your_decoder(future_features)
        return output, ode_result['ode_loss']
```

---

## 🔍 **5. 调试和验证**

### **检查数据流**

```python
# 在适配器中添加调试信息
def adapt_input(self, data, timestamps=None):
    print(f"输入数据形状: {data.shape}")

    adapted = self.convert_data(data)
    print(f"适配后形状: {adapted['current_input'].shape}")

    return adapted
```

### **验证输出正确性**

```python
# 检查输出是否符合预期
result = wrapper(test_data)
assert result['predictions'].shape == expected_shape
assert torch.isfinite(result['predictions']).all()
print("✅ 输出验证通过")
```

---

## 📚 **6. 高级修改**

### **A. 修改ODE内部格式 (不推荐)**

如果必须修改ODE模块内部:

**位置**: `ode_modules/cores/bayesian_ode.py`
```python
def forward(self, times, input, obs, delta_t, T, return_path=True):
    # 在这里修改输入处理逻辑
    # 但强烈建议使用适配器而不是修改这里
```

### **B. 添加新的求解器**

**位置**: `ode_modules/cores/bayesian_ode.py:ode_step`
```python
elif self.solver == "your_new_solver":
    # 添加您的求解器逻辑
    state = your_solver_step(state, input, delta_t)
```

### **C. 修改损失函数**

**位置**: `ode_modules/cores/bayesian_ode.py:forward`
```python
# 在返回前修改损失计算
custom_loss = your_loss_function(predictions, targets)
return state, custom_loss, predictions
```

---

## ✅ **总结**

1. **推荐方法**: 使用现有适配器或创建自定义适配器
2. **修改位置**: `ode_modules/adapters/` 文件夹
3. **不要修改**: ODE模块核心代码 (`cores/`, `cells/`)
4. **测试**: 使用验证脚本确保修改正确

**🎯 记住: 适配器方法让您在不破坏ODE模块的情况下支持任意数据格式！**