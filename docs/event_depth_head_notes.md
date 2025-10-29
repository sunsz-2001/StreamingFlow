# 事件深度分支集成记录

本文用于记录 EvRT 事件编码器 + 轻量化深度头（无 TES）在线上联调的阶段性成果与后续计划。

---

## 当前实现概览

1. **EvRT 事件编码器输出扩展**  
   - `streamingflow/models/event_encoder_evrt.py` 在保持原有特征输出的基础上新增 `EventDepthHead`，直接预测 stride-8 下的事件深度 logits。  
   - 深度头结构为浅层 Conv-BN-ReLU 堆叠，可通过 `MODEL.EVENT.DEPTH_HEAD_CHANNELS` 调整宽度，`MODEL.EVENT.DEPTH_BINS` 控制离散深度数量。

2. **StreamingFlow 主干调整**  
   - `streamingflow/models/streamingflow.py` 通过 `event_encoder_forward` 同时获得事件特征与深度，自动插值到 LSS 需要的深度 bin。  
   - 事件分支具备完整的 lifting→BEV→Temporal 流水线，可在 `USE_CAMERA=False` 时独立运行。  
   - 当相机与事件同时启用时，默认以 `MODEL.EVENT.BEV_FUSION`（`sum`/`avg`）在 BEV 空间融合两路特征，也可将 `FUSION_TYPE` 设为 `concat`/`residual` 继续沿用旧的前融合路径。  
   - 前向输出新增 `event_depth_prediction`，便于训练或异步更新时复用。

3. **配置项**  
   - `MODEL.EVENT.USE_DEPTH_HEAD`（默认启用）控制是否构建深度头。  
   - `MODEL.EVENT.DEPTH_BINS` 与 `MODEL.EVENT.DEPTH_HEAD_CHANNELS` 支持独立调节事件深度分辨率与头部宽度。  
   - `MODEL.EVENT.FUSION_TYPE` 默认 `independent` 表示事件分支独立输出；`MODEL.EVENT.BEV_FUSION` 控制与相机 BEV 的融合方式。

---

## 待完成事项

1. **事件深度监督**  
   - 结合 LiDAR 投影或相机深度生成伪标签，对 `event_depth_prediction` 加入显式损失。  
   - 评估与相机深度的 KL/交叉熵联合训练策略。

2. **异步更新策略**  
   - 设计事件输入触发逻辑（时间窗或事件量），缓存最新 `event_depth_prediction` 供相机帧缺失时复用。  
   - 调研事件分支独立刷新 BEV 状态对整体规划/预测任务的影响。

3. **可视化与诊断**  
   - 增加事件深度分布可视化脚本（深度投影、熵统计），方便快速排查异常序列。  
   - 结合 `tests/test_event_branch.py` 在真实数据上校验体素化输出形状与数值范围。

记录人：*
