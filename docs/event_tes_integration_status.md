# TES 集成进度记录

本文件用于记录 EvRT-DETR + TES + StreamingFlow 联调的阶段性成果与后续计划，方便未来继续推进。

---

## 已完成工作

1. **事件单目分支基础完善**  
   - `streamingflow/models/event_encoder_evrt.py` 复用 evrt-detr PresNet + HybridEncoder，保证与 LSS 输出形状对齐。  
   - `tests/test_event_branch.py` 验证伪帧 → 编码 → BEV LSS 流程可运行。

2. **TES 相关新模块**  
   - `streamingflow/models/event_stereo.py`：实现 `EventStereoBuffer` 及轻量版 `SimpleTESHead`（历史平均 + 整数视差成本体 + 深度分布）。  
   - `streamingflow/models/event_branch.py`：共享权重的双目事件编码器与规范化工具函数。  
   - `streamingflow/models/streamingflow.py`：支持 `MODEL.EVENT_STEREO` 配置，加入 stereo 缓冲、override 接口以及 BEV 聚合逻辑。

3. **测试脚本与文档**  
   - `tests/test_event_stereo_head.py`：对 TES 头做纯张量单元测试。  
   - `tests/test_event_stereo_desc.py`：探索 DSEC 序列的 `voxel/`、`raw_events/` 目录结构，打印关键统计。  
   - `tests/EVENT_BRANCH_TESTS.md`：整理所有事件分支相关的测试命令和值得关注的输出。  
   - `tools/dsec_inspect.py`：辅助查看目录结构、事件/voxel/视差基本信息。

4. **配置扩展**  
   - `MODEL.EVENT_STEREO` 增加历史长度、最大视差、深度 bin、深度范围、基线、调试开关，便于后续调参。

---

## 待确认 / 待完成事项

1. **左右事件数据来源**  
   - 当前 `voxel/*.npz` 仅包含单路堆叠（shape `(5, 480, 640)`）；需确认真正的左/右事件文件结构（如 `num5voxel0/1`、`events_left/right` 等）。  
   - 若只有 `raw_events/*.npy`，需按 DSEC 官方规则解码并区分左右摄像头。

2. **标定参数**  
   - 待找到或确认序列对应的 `fx` 与 `baseline`（可从 DSEC 官方校准文件或 TemporalEventStereo loader 获取）。

3. **数据加载与对齐**  
   - 在 StreamingFlow 数据层输出 `event_left` / `event_right`（或等价结构），并提供时间戳/索引信息。  
   - 完整实现环形缓冲触发逻辑（固定时间或事件量），确保 TES 更新与 RGB/LiDAR 异步时的稳定性。

4. **真实数据测试**  
   - 补全 `tests/test_event_stereo_desc.py`，真正加载左/右事件张量并运行 `EventStereoEncoder + SimpleTESHead`。  
   - 在小批量 DSEC 序列上检验深度分布输出是否符合预期（形状、归一化、随视差变化的合理性）。

5. **训练与融合**  
   - 若准备训练 TES 头：搭建阶段 A（只训 TES）和阶段 B（StreamingFlow 联训）脚本，指定损失与冻结策略。  
   - 完成 LSS 输入的深度 override 后，观察 BEV 输出质量，必要时调节融合方式。

---

## 建议的下一步

1. **核实 DSEC 左/右事件文件**，确认路径与命名规则；若缺失，需要从原始 `.h5` 重新导出。  
2. **确定标定数据**，填入 `MODEL.EVENT_STEREO.BASELINE` 及动态 `fx` 计算逻辑。  
3. **更新测试脚本** 以加载真实左右事件并跑 TES 头；随后在 StreamingFlow `forward` 中传入正确数据进行端到端测试。  
4. **整理任意额外的调试输出**（如成本体熵、深度分布可视化），便于后续梯度检查与调参。

记录人：*

