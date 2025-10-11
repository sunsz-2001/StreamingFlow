# Camera Multisweep (High‑Frequency Camera Observations) — Change Plan (方案 A‑2)

目的
- 在不改变监督时间点（仍用 2Hz 关键帧标签）的前提下，引入高频相机观测（≈12Hz 的非关键帧 sample_data）作为 ODE 的异步观测，提升时序状态估计的时域分辨率与稳定性。
- 改动尽量小、开关可控、默认关闭，评测优先，训练可后续打开。

关键思路（方案 A‑2）
- 相机高频观测只作为 ODE 的观测，不改变接收域长度 `TIME_RECEPTIVE_FIELD`、也不改变损失计算的时间戳（仍然是 2Hz 的 `target_timestamp`）。
- 在前向中：对新增的高频相机帧做 lifting 到 BEV，得到单帧 BEV 切片特征，连同其时间戳作为观测事件注入 ODE（与现有 camera_states、lidar_states 一起按时间排序融合）。

---

一、配置层改动

文件：`streamingflow/config.py`
- 在 `DATASET` 节新增开关（默认关闭）：
  - `DATASET.CAMERA_MULTISWEEP: False`  是否启用高频相机观测
  - `DATASET.CAMERA_SWEEP_STRIDE: 2`    相机中间帧抽样步长（1≈全取≈12Hz；2≈6Hz）
  - `DATASET.CAMERA_MAX_SWEEPS: 6`      接收域内最多取多少张中间帧（控制显存/带宽）
  - 可选：`DATASET.CAMERA_WINDOW_SEC: null`  接收域窗口长度（秒）。为空时，复用 `TIME_RECEPTIVE_FIELD` 推导窗口 = `(R-1)*0.5s`
  - 可选：`DATASET.CAMERA_SET_HI: null`  仅对哪些相机启用高频（如 `['CAM_FRONT']`）；为空表示按 `IMAGE.NAMES` 全部相机。

YAML（示例，可按需写入你的配置文件）
```yaml
DATASET:
  CAMERA_MULTISWEEP: true
  CAMERA_SWEEP_STRIDE: 2   # ~6Hz
  CAMERA_MAX_SWEEPS: 6
  CAMERA_WINDOW_SEC: null
  CAMERA_SET_HI: ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
```

---

二、数据集改动（NuScenes）

文件：`streamingflow/datas/NuscenesData.py`

1) 新增高频相机观测的采集函数（伪代码）
- 位置建议：紧邻 `get_points_from_multisweeps()` 之后，或放在类内任意位置。
- 函数签名（示例）：
```python
def get_camera_multisweeps(self, rec_ref, window_sec, stride, max_sweeps, camera_set=None):
    """
    返回：
      images_hi: List[Tensor (N,3,H,W)]  长度 S_cam
      intrinsics_hi: List[Tensor (N,3,3)]
      extrinsics_hi: List[Tensor (N,4,4)]  (sensor_to_lidar)
      timestamps_hi: np.ndarray (S_cam,)   # 绝对时间戳(微秒)
    说明：只取 is_key_frame=False 的 sample_data，沿 prev 链回溯，限定于 [t_ref - window_sec, t_ref]。
    """
    # 1) 参考时刻（关键帧 lidar 当前帧）
    ref_sd_rec = self.nusc.get('sample_data', rec_ref['data']['LIDAR_TOP'])
    t_ref = ref_sd_rec['timestamp']
    t_min = t_ref - int(window_sec * 1e6)

    cameras = camera_set or self.cfg.IMAGE.NAMES
    buckets = []  # 每个元素是一帧的 {cam->(img,intri,extr)}，之后在相机维度上拼接

    for cam in cameras:
        sd = self.nusc.get('sample_data', rec_ref['data'][cam])  # 关键帧 sample_data
        # 从关键帧向 prev 方向遍历中间帧（is_key_frame=False）
        step = 0
        cur = sd
        while True:
            cur = self.nusc.get('sample_data', cur['prev']) if cur['prev'] != "" else None
            if cur is None: break
            if cur['timestamp'] < t_min: break
            if cur['is_key_frame']: break  # 遇到上一个关键帧则停止
            if step % stride != 0:  # 抽样步长
                step += 1
                continue
            # 装载图像 + 标定，复用 get_input_data 的流程生成 (img, intri, sensor_to_lidar)
            # 建议抽象出一个小工具函数：load_one_camera_frame(cur, lidar_to_world, augmentation_parameters)
            # 将该相机的此帧加入 buckets 相应时间戳的聚合中（按 cur['timestamp'] 归类）
            step += 1

    # 将 buckets 中同一 timestamp 的多个相机拼成 (N,3,H,W)/(N,3,3)/(N,4,4)
    # 取最近的 max_sweeps 个时间戳，按时间升序返回 4 组列表（或张量）
    return images_hi, intrinsics_hi, extrinsics_hi, timestamps_hi
```

2) 在 `__getitem__` 中生成并写入 batch（默认关闭，开关打开才执行）
- 参考位置：`__getitem__` 末尾、已有 `data['camera_timestamp']` 处理附近（`streamingflow/datas/NuscenesData.py:913` 之后）。
- 逻辑：
  - 如果 `cfg.DATASET.CAMERA_MULTISWEEP` 为真：
    - 计算窗口长度：`window_sec = cfg.DATASET.CAMERA_WINDOW_SEC or 0.5*(self.receptive_field-1)`
    - 调 `get_camera_multisweeps(rec_ref, window_sec, stride, max_sweeps, camera_set)`
    - 将返回的列表沿“时间维”堆叠为张量：
      - `data['image_hi'] -> torch.stack(list, dim=0)`，形状 `[S_cam, N, 3, H, W]`
      - `data['intrinsics_hi'] -> torch.stack(..., dim=0)`，形状 `[S_cam, N, 3, 3]`
      - `data['extrinsics_hi'] -> torch.stack(..., dim=0)`，形状 `[S_cam, N, 4, 4]`
      - `data['camera_timestamp_hi'] -> np.array(S_cam,)`
    - 统一为相对秒：
      - `data['camera_timestamp_hi'] = (data['camera_timestamp_hi'] - current_time) / 1e6`
  - 若关闭开关，则不写入上述键。

数据形状小结
- 现有键：
  - `image: [T, N, 3, H, W]`（T=TIME_RECEPTIVE_FIELD，关键帧）
  - `camera_timestamp: [T]`（关键帧秒）
- 新增键（高频，仅观测）：
  - `image_hi: [S_cam, N, 3, H, W]`
  - `intrinsics_hi: [S_cam, N, 3, 3]`
  - `extrinsics_hi: [S_cam, N, 4, 4]`
  - `camera_timestamp_hi: [S_cam]`（秒）

---

三、模型前向改动（最小侵入）

文件：`streamingflow/models/streamingflow.py`

1) 扩展 `forward` 签名（全部可选，默认 None 保持兼容）
```python
def forward(self,
    image, intrinsics, extrinsics, future_egomotion,
    padded_voxel_points=None, camera_timestamp=None,
    points=None, lidar_timestamp=None, target_timestamp=None,
    image_hi=None, intrinsics_hi=None, extrinsics_hi=None, camera_timestamp_hi=None,
):
    ...
```

2) 生成高频相机 BEV 切片（不走 TemporalModel，相当于单帧 lifting）
- 伪代码：
```python
hi_states = None
if self.use_camera and image_hi is not None:
    # 逐帧 lifting，egomotion 置零（单帧特征）
    # image_hi: [S_cam, N, 3, H, W] → 增 batch 维： [1, S_cam, N, ...]
    b = 1
    x_hi, depth_hi, _ = self.calculate_birds_eye_view_features(
        image_hi.unsqueeze(0), intrinsics_hi.unsqueeze(0), extrinsics_hi.unsqueeze(0),
        future_egomotion=torch.zeros((b, image_hi.shape[0], 6), device=image_hi.device, dtype=image_hi.dtype)
    )  # 返回形如 [1, S_cam, C, H_bev, W_bev]
    hi_states = x_hi.squeeze(0)  # [S_cam, C, H, W]
```

3) 将高频相机观测传给 ODE（与原 camera_states/lidar_states 一起）
- 需要把 `camera_timestamp_hi: [S_cam]` 扩展到 batch：`camera_timestamp_hi.unsqueeze(0)` → `[1, S_cam]`
- 在调用 ODE 时追加两个参数：`camera_states_hi` 与 `camera_timestamp_hi`。

---

四、ODE 融合改动（支持额外观测）

文件：`streamingflow/ode_modules/cores/future_predictor.py`

1) 扩展 `forward` 签名
```python
def forward(self,
    future_prediction_input,
    camera_states, lidar_states, camera_timestamp, lidar_timestamp, target_timestamp,
    camera_states_hi=None, camera_timestamp_hi=None,
):
    ...
```

2) 构建观测表时合并高频相机观测（按时间戳聚合）
```python
obs_feature_with_time = {}
if camera_states is not None:
    for i in range(camera_timestamp.shape[1]):
        obs_feature_with_time[camera_timestamp[bs, i]] = camera_states[bs, i].unsqueeze(0)
if lidar_states is not None:
    for i in range(lidar_timestamp.shape[1]):
        obs_feature_with_time[lidar_timestamp[bs, i]] = lidar_states[bs, i].unsqueeze(0)
if camera_states_hi is not None:
    for i in range(camera_timestamp_hi.shape[1]):
        obs_feature_with_time[camera_timestamp_hi[bs, i]] = camera_states_hi[bs, i].unsqueeze(0)
# 后续按时间排序 → times, observations → self.gru_ode(...)
```

3) 其余 ODE 主循环、变步长、插补逻辑不变。

---

五、Trainer / Evaluate 透传改动

- `streamingflow/trainer.py`
  - 在 `shared_step()` 中，若 `cfg.DATASET.CAMERA_MULTISWEEP` 为真且 batch 含有 `image_hi` 等键，则从 batch 取出并在 `self.model(...)` 调用时透传这 4 个新参数。
  - 训练损失与监督时间点不变（只在 2Hz 的 `target_timestamp` 上计算）。

- `evaluate.py` / `evaluate_datastream.py`
  - 与 `trainer.py` 同理，判断有则透传，无则保持 `None`。

---

六、开关、回退与兼容

- 新增逻辑全部由 `cfg.DATASET.CAMERA_MULTISWEEP` 控制；默认 False，完全不改变现有行为。
- 当为 True 但因数据缺失没有取到中间帧时，应优雅回退（`image_hi` 等为 None，不影响主流程）。

---

七、验证与上线建议

1) 验证顺序
- 仅评测路径先打通：`CAMERA_MULTISWEEP=True, CAMERA_SWEEP_STRIDE=2, CAMERA_MAX_SWEEPS=6`，仅前向注入，不改训练。
- 观察推理耗时与显存占用；必要时仅对 `['CAM_FRONT']` 开启。

2) 质量确认
- 指标不下降（vehicle_iou、panoptic 等）；关注时间一致性的可视化效果是否更稳定。

3) 后续扩展（可选）
- 训练也打开 `CAMERA_MULTISWEEP`：仍只在 2Hz 关键帧上监督；高频观测仅作为 ODE 观测。
- 若收益显著，再考虑“方案 B”（相机接收域改高频，与 LiDAR 接收域解耦）。

---

八、注意事项与坑位

- 时间戳单位统一为秒（相对参考时刻），请与 `camera_timestamp`/`lidar_timestamp` 保持一致。
- lifting 到 BEV 时，单帧 egomotion 置零；若希望更精确，也可使用相邻姿态估计补偿，但会增加复杂度。
- 多相机拼接顺序与现有 `IMAGE.NAMES` 保持一致，避免维度错位。
- 显存与 IO：高频相机会显著增加带宽消耗。优先做小规模启用（少量相机、较大的 stride、受限的 `CAMERA_MAX_SWEEPS`）。

---

九、最小改动清单（按文件）

- `streamingflow/config.py`
  - 新增 `DATASET.CAMERA_MULTISWEEP`、`DATASET.CAMERA_SWEEP_STRIDE`、`DATASET.CAMERA_MAX_SWEEPS`、`DATASET.CAMERA_WINDOW_SEC`、`DATASET.CAMERA_SET_HI` 字段（含默认值）。

- `streamingflow/datas/NuscenesData.py`
  - 新增：`get_camera_multisweeps(...)`，收集非关键帧相机观测（按 stride 与窗口）。
  - `__getitem__`：在开关为真时，写入 `image_hi`、`intrinsics_hi`、`extrinsics_hi`、`camera_timestamp_hi`；单位标准化为秒。

- `streamingflow/models/streamingflow.py`
  - `forward(...)` 扩展 4 个可选入参；
  - 若传入 `image_hi` 等，则生成 `hi_states`（单帧 lifting 到 BEV 的切片），与 `camera_timestamp_hi` 一起传给 ODE。

- `streamingflow/ode_modules/cores/future_predictor.py`
  - `forward(...)` 扩展 2 个可选入参；
  - 构建观测表时合并 `camera_states_hi@camera_timestamp_hi`。

- `streamingflow/trainer.py` / `evaluate.py` / `evaluate_datastream.py`
  - 如 batch 提供 `*_hi` 键则透传；否则传 `None` 保持兼容。

---

十、里程碑

- M1：评测路径支持高频相机观测（方案 A‑2），默认关闭，打开后推理可运行。
- M2：小范围训练打开开关（不改监督栅格），观察是否带来收益。
- M3：根据收益与资源，评估是否推进方案 B。

---

如需，我可以基于此清单提交一版最小实现（仅评测路径）。默认行为不变，开关关闭时零影响。

