# StreamingFlow ODE Modules Pipeline

```mermaid
flowchart TD
    A[Input\nfuture_prediction_input,\ncamera_states,\nlidar_states,\ncamera_timestamp,\nlidar_timestamp,\ntarget_timestamp] --> B[Merge Observations\nper batch\nsort by timestamp]
    B --> C[FuturePredictionODE\ncore forward pass]
    C --> D[NNFOwithBayesianJumps]
    D --> D1[SRVP Encode\nSmallEncoder\n(+ optional skip conn)]
    D1 --> D2[Neural ODE Integration\nDualGRUODECell (continuous dynamics)\nGRUObservationCell (jump updates)\nSolver: euler | midpoint]
    D2 --> D3[Latent State Inference\nConvNet produces q(y)\nrsample_normal]
    D3 --> D4[SRVP Decode\nSmallDecoder -> future feature volumes]
    D4 --> E[Stack predicted\nfuture steps\nfor each target time]
    E --> F[SpatialGRU stack\nn_gru_blocks × SpatialGRU]
    F --> G[Residual refinement\n(Block × n_res_layers)\nor\nDeepLabHead]
    G --> H[Outputs\nfuture feature sequence\n(auxiliary loss)]
```

**Data flow summary**

- **Inputs**: fused present state (`future_prediction_input`), modality-specific states (`camera_states`, `lidar_states`) and their timestamps, plus desired prediction timestamps (`target_timestamp`).
- **Observation assembly**: FuturePredictionODE aligns and merges camera/LiDAR trajectories into a single time-ordered observation tensor.
- **Continuous-time modelling**: `NNFOwithBayesianJumps` encodes observations, integrates dynamics with `DualGRUODECell`, applies jump updates via `GRUObservationCell`, samples latent predictions, and decodes to spatial feature maps.
- **Spatial-temporal refinement**: The predicted sequence is refined by stacked `SpatialGRU` blocks followed by convolutional residual heads (`Block` or `DeepLabHead`), yielding the final future feature volume and auxiliary outputs used by downstream decoders.
