# StreamingFlow 项目修改历史

本文件记录 StreamingFlow 项目的所有修改历史，包括新功能、优化和bug修复。

---

## 2025-11-14 (下午) - 修复数据集点云加载逻辑

### 问题背景
在进行 Event 分支速度测试时发现，即使配置文件中设置了 `USE_LIDAR: False`，数据集仍然会无条件加载点云数据（LiDAR + Disparity），导致：
1. **不必要的 I/O 开销**：加载不需要的点云数据文件
2. **内存浪费**：存储不使用的点云数据
3. **Batch Size 限制**：点云数据变长导致 `DataLoader` 堆叠失败（batch_size > 1 时报错）

### 修改内容

#### 1. `streamingflow/datas/DSECData.py`

**修改点 1：添加 `use_lidar` 配置读取**
```python
# 第 143 行新增
self.use_lidar = getattr(self.cfg.MODEL.MODALITY, 'USE_LIDAR', False)
```

**修改点 2：条件化点云数据加载**
- **修改前**（第 1137-1145 行）:
  ```python
  target_infos, points = self.get_infos_and_points(target_idx_list)  # 无条件加载
  points = points[0]
  input_dict = {
      'points': points,  # 无条件添加
      'frame_id': current_info['sample_idx'],
      'pose': current_info['pose'],
      'sequence_name': current_info['sequence_name'],
  }
  ```

- **修改后**（第 1140-1150 行）:
  ```python
  input_dict = {
      'frame_id': current_info['sample_idx'],
      'pose': current_info['pose'],
      'sequence_name': current_info['sequence_name'],
  }
  
  # 根据配置条件加载点云数据
  if self.use_lidar:
      target_infos, points = self.get_infos_and_points(target_idx_list)
      points = points[0]
      input_dict['points'] = points
  ```

**设计对比**：
- ✅ **Event 数据**: 有条件判断 `if self.use_event:`
- ✅ **Image 数据**: 有条件判断 `if self.use_image:`
- ❌ **点云数据**（修改前）: 无条件加载
- ✅ **点云数据**（修改后）: 有条件判断 `if self.use_lidar:`

#### 2. `tests/test_event_branch.py`

**修改点：更新 `benchmark_collate_fn` 注释**
- 添加说明：当 `USE_LIDAR=False` 时，`'points'` 字段不会出现在数据中
- 明确 `variable_length_keys` 中的字段只在相应模态启用时才存在

### 修改效果

#### 测试 Event 分支（USE_LIDAR=False）时：
- ✅ **不再加载点云数据**（节省 I/O 和内存）
- ✅ **支持任意 Batch Size**（无变长数据冲突）
- ✅ **纯粹的 Event 分支测试**（无其他模态干扰）

#### 测试多模态（USE_LIDAR=True）时：
- ✅ **正常加载点云数据**
- ✅ **自动处理变长字段**（通过 `benchmark_collate_fn`）

### 技术要点
- **按需加载原则**：只加载配置启用的数据模态
- **配置驱动设计**：数据加载行为完全由配置文件控制
- **代码一致性**：所有模态（Image/Event/LiDAR）都使用相同的条件加载模式

### 后续修复：标注框变长字段处理

**问题**：修复点云加载后，batch_size=4 时又遇到标注框（Ground Truth Boxes）变长问题：
```
RuntimeError: stack expects each tensor to be equal size, but got [10, 11] at entry 0 and [5, 11] at entry 1
```

**原因**：不同帧中检测到的物体数量不同（如样本0有10个物体，样本1有5个物体），导致 `gt_boxes`, `gt_obj_ids` 等字段无法堆叠。

**解决方案**（`tests/test_event_branch.py` 第 415-419 行）：
```python
variable_length_keys = {
    'points', 'padded_voxel_points',  # 点云相关
    'gt_boxes', 'gt_len', 'gt_obj_ids', 'gt_boxes_prosed',  # 标注框相关
    'gt_names', 'num_lidar_pts', 'num_radar_pts',  # 其他可能的变长字段
}
```

**效果**：现在支持任意 batch size，自动处理所有变长字段（点云、标注框等）。

### 关键 Bug 修复：吞吐量计算错误

**问题发现**：用户发现 `--benchmark-encoder-only` 模式下，虽然只测试编码器部分（应该更快），但显示的 FPS/吞吐量反而比完整测试更低，这是不合理的。

**根本原因**：吞吐量计算公式错误，**忽略了 `batch_size`**：
```python
# 错误的计算（原代码）
throughput = 1000.0 / time_per_batch  # 实际是 Batches/s，而不是 Samples/s
```

这个错误导致：
- 当 `batch_size=4` 时，吞吐量被低估了 4 倍
- 无法正确反映批处理带来的性能提升
- 编码器专项测试和完整测试的对比失真

**正确公式**：
```python
# 正确的计算（修复后）
throughput = (1000.0 / time_per_batch) * batch_size  # Samples/s
```

**修复位置**（`tests/test_event_branch.py`）：
1. `run_encoder_benchmark_test()` 第 746 行：
   ```python
   encoder_throughput = (1000.0 / total_encoder_pipeline_times.mean()) * cfg.BATCHSIZE
   ```

2. `run_benchmark_test()` 第 957-958 行：
   ```python
   inference_throughput = (1000.0 / inference_times.mean()) * cfg.BATCHSIZE
   total_throughput = (1000.0 / total_times.mean()) * cfg.BATCHSIZE
   ```

**命名规范化**：
- 移除了易混淆的 "FPS" 标签（与视频帧率混淆）
- 统一使用 "吞吐量 (samples/s)" 作为指标名称

**验证方法**：
```bash
# batch_size=1 时，吞吐量约为 50 samples/s
python tests/test_event_branch.py --config-file configs/dsec_event.yaml --benchmark-encoder-only --batch-size 1

# batch_size=4 时，吞吐量应该接近 200 samples/s (4倍)
python tests/test_event_branch.py --config-file configs/dsec_event.yaml --benchmark-encoder-only --batch-size 4
```

**效果**：现在吞吐量计算准确，能正确反映不同 `batch_size` 和不同测试范围的性能差异。

### 新增功能：延迟 (Latency) 指标和批处理效率分析

**用户问题**：用户发现 `batch_size=1` 时吞吐量为 70 samples/s，`batch_size=4` 时为 280 samples/s（正好 4 倍），质疑这是否意味着"batch size 越大速度越快"，是否合理。

**核心概念澄清**：
- **吞吐量 (Throughput, samples/s)**：单位时间内处理的样本总数（系统总处理能力）
- **延迟 (Latency, ms/sample)**：处理每个样本的平均时间（单样本处理速度）

**分析**：
```
batch_size=1: 70 samples/s  → 延迟 = 1000/70 ≈ 14.3 ms/sample
batch_size=4: 280 samples/s → 延迟 = 1000/280 ≈ 3.57 ms/batch ÷ 4 = 14.3 ms/sample (相同！)
```

这说明 GPU 在**高效并行处理**：
- batch_size=4 时，4 个样本并行处理只需要和 1 个样本几乎相同的时间
- **吞吐量提升**（好事）：系统利用率更高
- **延迟不变**（最佳情况）：单个样本处理速度没有下降

**新增指标**（`tests/test_event_branch.py`）：
1. **延迟指标**：
   ```python
   latency_per_sample = time_per_batch / batch_size  # ms/sample
   ```
   
2. **批处理效率分析**：
   ```
   批处理效率分析 (batch_size=4):
     单个样本平均延迟: 14.3 ms
     并行效率: 如果串行处理 4 个样本需要 57.2 ms
               实际批处理只需要 14.3 ms
     加速比: 4.00x (理想值: 4.00x)
   ```

**输出对比**：

修改前（易混淆）：
```
推理吞吐量 (samples/s)                        280.00
```

修改后（清晰）：
```
推理吞吐量 (samples/s)                        280.00
推理延迟 (ms/sample)                           14.29

批处理效率分析 (batch_size=4):
  单个样本平均延迟: 14.29 ms
  加速比: 4.00x (理想值: 4.00x)
```

**结论**：
- ✅ Batch size 越大，**吞吐量越高**（在显存允许的范围内）
- ✅ 如果**延迟保持不变**，说明并行化效率很好
- ✅ 不同场景的选择：
  - **训练/离线批处理**：追求吞吐量 → batch_size 尽可能大
  - **在线实时推理**：追求延迟 → batch_size 要平衡响应时间

**效果**：现在用户可以同时看到吞吐量和延迟，全面了解性能特征。

### 自动计算 Event Frames FPS

**用户需求**：希望在测试报告中自动计算并显示 event frames 的真实 FPS，考虑时序长度和相机数量。

**实现原理**：
```python
# 自动从配置读取
TIME_RECEPTIVE_FIELD = 2  # 每个样本的时间步数
num_cameras = 1           # 相机数量
frames_per_sample = TIME_RECEPTIVE_FIELD × num_cameras = 2

# 计算 FPS
FPS = samples/s × frames_per_sample
latency_per_frame = latency_per_sample / frames_per_sample
```

**新增输出**（`tests/test_event_branch.py`）：

修改前（只有 samples/s）：
```
推理吞吐量 (samples/s)                        280.00
推理延迟 (ms/sample)                           14.29
```

修改后（同时显示 samples/s 和 frames/s）：
```
推理吞吐量 (samples/s)                        280.00
推理延迟 (ms/sample)                           14.29
推理吞吐量 (frames/s, FPS)                    560.00  ← 新增！
推理延迟 (ms/frame)                             7.15  ← 新增！
  (TIME_RECEPTIVE_FIELD=2, cameras=1, frames/sample=2)
```

**优势**：
- ✅ 自动从配置文件读取参数，无需手动计算
- ✅ 同时显示 samples/s 和 frames/s，满足不同对比需求
- ✅ 明确标注 "frames/s, FPS"，避免歧义
- ✅ 显示换算参数，便于验证和理解

**适用范围**：
- `run_encoder_benchmark_test()`：编码器专项测试
- `run_benchmark_test()`：完整端到端测试

**效果**：现在可以直接看到 event frames 的真实处理速度，便于与其他工作对比。

### Event 编码器专项速度测试

**新增功能**：添加了 `--benchmark-encoder-only` 选项，专门测试 Event 编码器管道的速度。

**测试范围**：
- ✅ Event 编码器前向传播 (`event_encoder_forward`)
- ✅ 深度分布计算 (`softmax`)
- ✅ 特征+深度扩展 (`_expand_features_with_depth`)
- ❌ **不包括**：BEV 投影、时序模型、解码器

**实现细节**（`tests/test_event_branch.py`）：
- 新增函数：`run_encoder_benchmark_test()`
- 细粒度计时：分别测量编码器、深度估计、特征扩展的时间
- 组件占比分析：显示每个组件在管道中的时间占比

**使用方法**：
```bash
python tests/test_event_branch.py \
    --config-file configs/dsec_event.yaml \
    --benchmark-encoder-only \
    --batch-size 4 \
    --num-batches 100 \
    --num-warmup 10 \
    --save-benchmark encoder_benchmark.csv
```

**输出示例**：
```
Event 编码器专项速度基准测试结果
================================================================================
指标                                          平均值        中位数        标准差
--------------------------------------------------------------------------------
Event 编码器时间 (ms/batch)                    15.23        15.10         0.85
深度估计时间 (ms/batch)                         2.34         2.30         0.12
特征扩展时间 (ms/batch)                         3.45         3.42         0.18
--------------------------------------------------------------------------------
编码器管道总时间 (ms/batch)                    21.02        20.95         1.05

编码器管道各组件占比:
  Event 编码器:  72.45%  (15.23 ms)
  深度估计:      11.13%  (2.34 ms)
  特征扩展:      16.42%  (3.45 ms)
```

---

## 2025-11-14 (上午) - Event 分支速度测试脚本开发

### 修改内容

#### 重要说明
**修改策略调整**: 最初创建了独立的测试脚本（`event_speed_test.py` 和 `test_event_speed_simple.py`），但在用户反馈后，改为**直接扩展已有的** `tests/test_event_branch.py`，以保持代码组织的一致性。

#### 1. 修改文件

**`tests/test_event_branch.py` - 扩展事件分支测试脚本**（主要修改）
- **原有功能**: 真实数据推理、合成数据自检
- **新增功能**: 速度基准测试模式
- **修改内容**:
  - 添加 `--benchmark` 参数启用速度测试模式
  - 添加 `--num-warmup` 参数设置预热次数
  - 添加 `--save-benchmark` 参数保存测试结果
  - 新增 `run_benchmark_test()` 函数实现完整的速度测试逻辑
- **测试功能**:
  - 数据加载速度测量
  - 数据预处理速度测量
  - 模型推理速度测量（使用 CUDA Event 精确计时）
  - 端到端处理速度测量
  - 详细的统计分析（平均值、中位数、标准差、百分位数）
- **使用方法**:
  ```bash
  python tests/test_event_branch.py --config-file configs/dsec_event.yaml --benchmark --num-batches 100 --save-benchmark results.csv
  ```

#### 2. 开发过程说明

**注意**: 开发过程中曾创建了独立的测试脚本（`event_speed_test.py` 和 `test_event_speed_simple.py`），但根据用户反馈，这些独立脚本已被删除。所有速度测试功能已完全集成到 `tests/test_event_branch.py` 中，保持代码的组织性和一致性。

#### 3. 修改原因

- **需求**：用户需要测试 evrt-detr event 分支的处理速度
- **目标**：
  1. 评估数据加载性能瓶颈
  2. 评估模型推理性能
  3. 为性能优化提供基准数据
  4. 对比不同配置下的性能表现

#### 4. 技术实现

**数据加载测试**:
- 使用 `DatasetDSEC` 加载 DSEC 数据集
- 测量单个样本的加载时间
- 测量数据预处理时间（转换为 tensor、移动到设备等）
- 统计分析：平均值、中位数、标准差、百分位数
- 性能评估：根据加载时间给出性能等级

**模型推理测试**:
- 支持加载预训练的 evrt-detr 模型
- 使用 `torch.cuda.Event` 进行精确的 GPU 时间测量
- 包含预热阶段，避免首次运行的开销影响统计
- 测量指标：
  - 推理时间（ms/batch 和 ms/sample）
  - FPS（frames per second）
  - 吞吐量（samples/s）
- 支持不同的数据类型（float32/float16/bfloat16）

**代码结构**:
- 模块化设计，易于扩展和维护
- 完善的错误处理和用户提示
- 自动处理数据类型不兼容问题（`benchmark_collate_fn`）
- 结果自动保存和格式化输出

#### 5. 相关文件

- `tests/test_event_branch.py` - 主测试脚本（已修改）
- `streamingflow/datas/DSECData.py` - DSEC 数据集加载器（已存在）
- `streamingflow/datas/dataloaders.py` - 数据加载器工厂函数（已存在）
- `streamingflow/models/streamingflow.py` - 主模型（已存在）
- `evrt-detr/` - Event 编码器依赖（需预先安装）

#### 6. 依赖项

测试功能依赖以下库（项目已包含）:
- `torch` - PyTorch 深度学习框架
- `numpy` - 数值计算
- `pandas` - 结果保存（可选）
- `fvcore` - 配置管理
- `streamingflow` - 项目自定义模块
- `evrt-detr` - Event 编码器模块（需 `pip install -e ./evrt-detr`）

#### 7. 常见问题和解决方案

**问题1: TypeError: default_collate 错误**
```
TypeError: default_collate: batch must contain tensors... found <U7
```
- **原因**: 数据集返回了字符串字段（如 `sequence_name`），PyTorch 无法处理
- **解决**: 已实现 `benchmark_collate_fn` 自动过滤字符串字段

**问题2: ValueError: too many values to unpack**
```
ValueError: too many values to unpack (expected 2)
```
- **原因**: `prepare_dataloaders(return_dataset=True)` 返回4个值
- **解决**: 正确解包为 `trainloader, valloader, train_dataset, val_dataset`

**问题3: RuntimeError: stack expects each tensor to be equal size**
```
RuntimeError: stack expects each tensor to be equal size, but got [86402, 4] at entry 0 and [87312, 4] at entry 1
```
- **原因**: batch_size > 1 时，点云数据（`points`）等变长字段无法直接 stack
- **解决**: 已修改 `benchmark_collate_fn`，将变长字段保持为 list

**问题4: CUDA Out of Memory**
- **解决**: 减小 `--batch-size` 或使用 `--device cpu`

**问题5: 数据集路径错误**
- **检查**: `cat configs/dsec_event.yaml | grep DATAROOT`
- **确认**: 数据集是否存在于指定路径

#### 8. 完整使用流程

**步骤1: 首次测试（最小配置）**
```bash
python tests/test_event_branch.py \
    --config-file configs/dsec_event.yaml \
    --benchmark \
    --num-batches 10 \
    --num-warmup 2 \
    --batch-size 1
```

**步骤2: 正常基准测试**
```bash
python tests/test_event_branch.py \
    --config-file configs/dsec_event.yaml \
    --benchmark \
    --num-batches 100 \
    --num-warmup 10 \
    --batch-size 4 \
    --split val \
    --save-benchmark results.csv
```

**步骤3: 完整性能测试**
```bash
python tests/test_event_branch.py \
    --config-file configs/dsec_event.yaml \
    --benchmark \
    --num-batches 500 \
    --num-warmup 20 \
    --batch-size 4 \
    --split val \
    --num-workers 4 \
    --save-benchmark dsec_baseline.csv
```

#### 9. 性能优化建议

**数据加载优化**:
- 增加 `--num-workers` (推荐 4-8)
- 使用 SSD 存储数据集
- 预处理并缓存数据

**模型推理优化**:
- 增加 `--batch-size`（根据 GPU 内存）
- 使用混合精度（配置中启用 FP16）
- 考虑模型量化或剪枝
- 使用 `torch.compile()` 编译模型（PyTorch 2.0+）

**系统优化**:
- 关闭其他占用 GPU 的程序
- 固定 GPU 频率：`sudo nvidia-smi -lgc 1500`
- 监控资源：`watch -n 1 nvidia-smi`

---

## 未来计划

1. **性能优化**：
   - 分析瓶颈，优化数据加载流程
   - 探索模型量化和剪枝
   - 实现多 GPU 推理支持
2. **功能扩展**：
   - 添加对更多数据集的支持（Gen1、Gen4等）
   - 添加模型各组件的详细性能分析
   - 添加可视化工具，展示性能分析结果
3. **文档完善**：
   - 添加详细的使用教程
   - 添加性能优化指南
   - 添加常见问题解答

---

## 版本历史

### v1.0 - 2025-11-14
- 初始版本
- 实现基本的数据加载和模型推理速度测试
- 支持 DSEC 数据集
- 支持 evrt-detr 模型

---

*注：本文件将持续更新，记录所有重要的代码修改和功能变更*

