# 技术细节说明

## DSEC数据集标注格式

### 标注文件位置

```
{DATAROOT}/{sequence_name}/{sequence_name}_interpolate_fov_bbox_lidar_check.pkl
```

### 数据结构

pkl文件中每个样本包含：
- `annos`: 序列中所有帧的标注列表
  - `gt_boxes_lidar`: `(N, 9)` 3D边界框 `[x, y, z, dx, dy, dz, heading, vx?, vy?]`
  - `name`: `(N,)` 类别名称（'Vehicle', 'Cyclist', 'Pedestrian'）
  - `obj_ids`: `(N,)` 实例ID

### 数据加载

- 数据加载：`streamingflow/datas/DSECData.py`
- 标注提取：`__getitem__` 方法中从pkl文件提取
- BEV标签生成：从3D边界框投影生成（如需要）

## 几何维度调整

### 问题

事件编码器输出尺寸与geometry维度不匹配，需要插值调整。

### 解决方案

在 `projection_to_birds_eye_view` 中使用双线性插值调整 `geometry_b` 的空间维度。

### 原因

- Geometry基于 `IMAGE.FINAL_DIM / encoder_downsample`
- 事件编码器输出尺寸由架构决定，可能与相机分支不同

### 建议

统一事件和相机的分辨率配置，从根本上消除维度不匹配。

## 事件深度头

### 实现

- `EventEncoderEvRT` 新增 `EventDepthHead`
- 预测stride-8下的事件深度logits
- 配置项：`USE_DEPTH_HEAD`, `DEPTH_BINS`, `DEPTH_HEAD_CHANNELS`

### 待完成

- 事件深度监督（结合LiDAR投影）
- 异步更新策略
- 可视化与诊断

