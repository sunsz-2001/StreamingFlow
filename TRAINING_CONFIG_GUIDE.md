# 训练配置指南

本文档说明如何配置 `dsec_event_lidar.yaml` 以开始训练异步信息融合模型。

## 📋 快速开始

### 1. 基本训练配置

```bash
# 使用默认配置开始训练
python train.py --config-file streamingflow/configs/dsec_event_lidar.yaml
```

### 2. 关键配置项说明

#### 2.1 异步融合配置（必需）

要启用异步信息融合，需要设置以下参数：

```yaml
DATASET:
  USE_FLOW_DATA: True  # 启用流式数据格式，支持异步多模态数据融合

TIME_RECEPTIVE_FIELD: 1  # 过去帧数
N_FUTURE_FRAMES: 4  # 未来预测帧数（> 0 时启用异步融合）
```

**重要说明：**
- `USE_FLOW_DATA: True` - 启用流式数据格式，将事件序列分割为时间窗口
- `N_FUTURE_FRAMES > 0` - 启用未来预测，`FuturePredictionODE` 会处理异步对齐
- `N_FUTURE_FRAMES = 0` - 仅使用过去帧，不进行未来预测（同步模式）

#### 2.2 ODE 求解器配置

```yaml
MODEL:
  SOLVER: 'euler'  # 'euler' | 'midpoint' | 'dopri5'
  IMPUTE: False  # 是否填充缺失数据
```

**求解器选择：**
- `'euler'` - 欧拉法，快速但精度较低（推荐用于快速训练）
- `'midpoint'` - 中点法，平衡速度和精度
- `'dopri5'` - 自适应步长，最准确但最慢（需要 `USE_VARIABLE_ODE_STEP: True`）

#### 2.3 未来预测配置

```yaml
MODEL:
  FUTURE_PRED:
    N_GRU_BLOCKS: 2  # 空间 GRU 层数
    N_RES_LAYERS: 1  # 残差块层数
    MIXTURE: True  # 使用混合分布建模不确定性
    DELTA_T: 0.05  # ODE 时间步长（秒）
    USE_VARIABLE_ODE_STEP: False  # 自适应步长（需要 SOLVER='dopri5'）
```

**参数建议：**
- `N_GRU_BLOCKS`: 2-4（值越大，模型容量越大，但计算量也越大）
- `DELTA_T`: 0.05-0.1（越小越准确，但计算量越大）
- `MIXTURE: True` - 推荐启用，用于建模多模态不确定性

#### 2.4 分布配置

```yaml
MODEL:
  DISTRIBUTION:
    LATENT_DIM: 64  # 隐藏层维度，需与 TEMPORAL_MODEL.START_OUT_CHANNELS 匹配
    MIN_LOG_SIGMA: -5.0  # 最小对数方差
    MAX_LOG_SIGMA: 5.0  # 最大对数方差
```

**注意：** `LATENT_DIM` 必须与 `TEMPORAL_MODEL.START_OUT_CHANNELS` 相同（通常为 64）。

## 🔧 训练参数配置

### 3.1 基础训练参数

```yaml
GPUS: [1]  # 使用的 GPU 编号列表，例如 [0, 1, 2, 3] 表示使用 4 个 GPU
BATCHSIZE: 16  # 批次大小（根据 GPU 内存调整）
PRECISION: 32  # 16（混合精度，节省内存）或 32（全精度）
EPOCHS: 30  # 训练轮数
N_WORKERS: 4  # 数据加载器工作进程数
```

**内存优化建议：**
- 如果 GPU 内存不足，可以：
  - 减小 `BATCHSIZE`（例如 8 或 4）
  - 使用 `PRECISION: 16`（混合精度训练）
  - 减小 `N_FUTURE_FRAMES`（例如从 4 减到 2）

### 3.2 优化器配置

```yaml
OPTIMIZER:
  LR: 1e-4  # 学习率（建议范围：1e-5 到 1e-3）
  WEIGHT_DECAY: 1e-7  # 权重衰减（L2 正则化）

# 梯度裁剪（可选，在 config.py 中默认值为 5）
# GRAD_NORM_CLIP: 5
```

**学习率建议：**
- 从头训练：`1e-4` 到 `3e-4`
- 微调预训练模型：`1e-5` 到 `5e-5`

### 3.3 数据路径配置

```yaml
DATASET:
  NAME: 'dsec'
  DATAROOT: '/media/switcher/sda/datasets/dsec'  # 数据集根目录
  VERSION: 'trainval'  # 'trainval' | 'test'
```

**确保数据路径正确：**
- 检查 `DATAROOT` 路径是否存在
- 确保数据集中包含事件数据和 LiDAR 数据

## 🚀 训练场景配置

### 场景 1: 快速测试（同步模式）

```yaml
DATASET:
  USE_FLOW_DATA: False  # 同步模式

TIME_RECEPTIVE_FIELD: 1
N_FUTURE_FRAMES: 0  # 不进行未来预测

MODEL:
  SOLVER: 'euler'  # 快速求解器
  FUTURE_PRED:
    DELTA_T: 0.1  # 较大的时间步长
```

### 场景 2: 异步融合训练（推荐）

```yaml
DATASET:
  USE_FLOW_DATA: True  # 启用流式数据

TIME_RECEPTIVE_FIELD: 1
N_FUTURE_FRAMES: 4  # 预测未来 4 帧

MODEL:
  SOLVER: 'euler'  # 或 'midpoint' 用于更好的精度
  FUTURE_PRED:
    N_GRU_BLOCKS: 2
    N_RES_LAYERS: 1
    MIXTURE: True
    DELTA_T: 0.05
```

### 场景 3: 高精度异步融合

```yaml
DATASET:
  USE_FLOW_DATA: True

TIME_RECEPTIVE_FIELD: 1
N_FUTURE_FRAMES: 6  # 预测更多未来帧

MODEL:
  SOLVER: 'midpoint'  # 或 'dopri5' 用于最高精度
  FUTURE_PRED:
    N_GRU_BLOCKS: 3  # 增加模型容量
    N_RES_LAYERS: 2
    MIXTURE: True
    DELTA_T: 0.05
    USE_VARIABLE_ODE_STEP: True  # 需要 SOLVER='dopri5'
```

## 📝 训练命令示例

### 基本训练

```bash
python train.py --config-file streamingflow/configs/dsec_event_lidar.yaml
```

### 使用命令行覆盖配置

```bash
# 修改学习率
python train.py --config-file streamingflow/configs/dsec_event_lidar.yaml \
  OPTIMIZER.LR 5e-5

# 修改批次大小和 GPU
python train.py --config-file streamingflow/configs/dsec_event_lidar.yaml \
  BATCHSIZE 8 GPUS [0,1]

# 启用未来预测
python train.py --config-file streamingflow/configs/dsec_event_lidar.yaml \
  N_FUTURE_FRAMES 4
```

### 从检查点恢复训练

训练脚本会自动检测 `logs/{TAG}/` 目录下的最新检查点并恢复训练。

```bash
# 如果检查点存在，会自动恢复
python train.py --config-file streamingflow/configs/dsec_event_lidar.yaml
```

### 加载预训练权重

```yaml
PRETRAINED:
  LOAD_WEIGHTS: True
  PATH: 'path/to/checkpoint.ckpt'
```

## ⚠️ 常见问题

### 1. 内存不足（OOM）

**解决方案：**
- 减小 `BATCHSIZE`（例如从 16 减到 8）
- 使用 `PRECISION: 16`（混合精度）
- 减小 `N_FUTURE_FRAMES`（例如从 4 减到 2）
- 减小 `N_GRU_BLOCKS` 或 `N_RES_LAYERS`

### 2. 通道数不匹配错误

**原因：** `LATENT_DIM` 与 `START_OUT_CHANNELS` 不匹配

**解决方案：** 确保以下配置一致：
```yaml
MODEL:
  TEMPORAL_MODEL:
    START_OUT_CHANNELS: 64
  DISTRIBUTION:
    LATENT_DIM: 64  # 必须与 START_OUT_CHANNELS 相同
```

### 3. 数据加载错误

**检查项：**
- `DATAROOT` 路径是否正确
- `USE_FLOW_DATA` 是否与数据格式匹配
- 数据集中是否包含所需的事件和 LiDAR 数据

### 4. 训练速度慢

**优化建议：**
- 使用 `SOLVER: 'euler'`（最快的求解器）
- 增加 `DELTA_T`（例如从 0.05 到 0.1）
- 减小 `N_FUTURE_FRAMES`
- 使用多 GPU 训练（`GPUS: [0, 1, 2, 3]`）

## 📊 训练监控

训练日志和检查点保存在：
```
logs/{TAG}/
```

使用 TensorBoard 查看训练曲线：
```bash
tensorboard --logdir logs/{TAG}
```

## 🔍 配置验证清单

在开始训练前，请确认：

- [ ] `DATASET.DATAROOT` 路径正确
- [ ] `USE_FLOW_DATA` 与数据格式匹配
- [ ] `N_FUTURE_FRAMES > 0`（如果启用异步融合）
- [ ] `MODEL.DISTRIBUTION.LATENT_DIM == MODEL.TEMPORAL_MODEL.START_OUT_CHANNELS`
- [ ] `MODEL.MODALITY.USE_EVENT: True` 和 `MODEL.MODALITY.USE_LIDAR: True` 已启用
- [ ] GPU 内存足够（根据 `BATCHSIZE` 调整）
- [ ] 数据加载器可以正常读取数据

## 📚 相关文档

- 模型架构：`STREAMINGFLOW_PIPELINE.txt`
- 配置文件：`streamingflow/configs/dsec_event_lidar.yaml`
- 训练脚本：`train.py`


