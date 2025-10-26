import numpy as np
import torch


class EventTensorizer:
    """Convert raw event streams into dense tensor frames aligned with camera preprocessing."""

    def __init__(self, cfg):
        image_cfg = cfg.IMAGE
        event_cfg = cfg.MODEL.EVENT

        self.bins = event_cfg.BINS
        self.channels = 2 * self.bins
        in_channels = getattr(event_cfg, "IN_CHANNELS", 0)
        if in_channels > 0 and in_channels != self.channels:
            raise ValueError(
                "Online事件累积要求 MODEL.EVENT.IN_CHANNELS == 2 * MODEL.EVENT.BINS。"
            )

        self.final_height, self.final_width = image_cfg.FINAL_DIM
        self.scale_width = float(image_cfg.RESIZE_SCALE)
        self.scale_height = float(image_cfg.RESIZE_SCALE)
        resized_width = int(image_cfg.ORIGINAL_WIDTH * self.scale_width)
        self.crop_w = int(max(0, (resized_width - self.final_width) / 2))
        self.crop_h = int(image_cfg.TOP_CROP)
        self.normalize = getattr(event_cfg, "NORMALIZE", False)

    def prepare_frames(self, event_input, device, max_seq_len=None):
        """Return Tensor[B, S, N, C, H, W] ready for the encoder."""
        if isinstance(event_input, torch.Tensor):
            frames = event_input.to(device=device, dtype=torch.float32)
            frames = self._ensure_batch_shape(frames)
            if max_seq_len is not None and frames.shape[1] > max_seq_len:
                frames = frames[:, :max_seq_len]
            return frames

        if isinstance(event_input, dict):
            if "frames" in event_input:
                frames = event_input["frames"]
                frames = frames.to(device=device, dtype=torch.float32)
                frames = self._ensure_batch_shape(frames)
                if max_seq_len is not None and frames.shape[1] > max_seq_len:
                    frames = frames[:, :max_seq_len]
                return frames

            if "events" in event_input:
                raw = event_input["events"]
                counts = event_input.get("counts")
                return self._from_events_array(raw, counts, device, max_seq_len=max_seq_len)

            if "raw" in event_input:
                raw = event_input["raw"]
                return self._from_nested(raw, device, max_seq_len=max_seq_len)

        if isinstance(event_input, (list, tuple)):
            return self._from_nested(event_input, device, max_seq_len=max_seq_len)

        raise TypeError("不支持的事件输入格式，需提供Tensor、dict或嵌套list。")

    def _ensure_batch_shape(self, frames):
        if frames.dim() == 5:
            frames = frames.unsqueeze(0)
        if frames.dim() != 6:
            raise ValueError("事件帧Tensor必须是[B, S, N, C, H, W]或[S, N, C, H, W]。")
        return frames

    def _from_nested(self, nested, device, max_seq_len=None):
        if not isinstance(nested, (list, tuple)):
            raise TypeError("原始事件输入需要是嵌套list/tuple。")
        if len(nested) == 0:
            raise ValueError("原始事件输入为空。")

        batch = len(nested)
        seq = len(nested[0])
        cams = len(nested[0][0])
        if max_seq_len is not None:
            seq = min(seq, max_seq_len)

        frames = torch.zeros(
            (batch, seq, cams, self.channels, self.final_height, self.final_width),
            device=device,
            dtype=torch.float32,
        )

        for b in range(batch):
            if len(nested[b]) < seq:
                raise ValueError("原始事件序列长度不足以覆盖所需时间步。")
            for s in range(seq):
                if len(nested[b][s]) != cams:
                    raise ValueError("每个时间步的相机数量应一致。")
                for n in range(cams):
                    events = nested[b][s][n]
                    frame = self.events_to_frame(events, device)
                    frames[b, s, n] = frame
        return frames

    def _from_events_array(self, events_tensor, counts, device, max_seq_len=None):
        if not torch.is_tensor(events_tensor):
            events_tensor = torch.as_tensor(events_tensor)
        events_tensor = events_tensor.to(device=device, dtype=torch.float32)
        if events_tensor.dim() != 5 or events_tensor.size(-1) < 4:
            raise ValueError("events_tensor 需为[B, S, N, E, 4+]。")

        batch, seq, cams, events_per, _ = events_tensor.shape
        if max_seq_len is not None:
            seq = min(seq, max_seq_len)
            events_tensor = events_tensor[:, :seq]

        if counts is not None:
            counts = torch.as_tensor(counts, device=device, dtype=torch.long)
            if counts.shape[:3] != (batch, seq, cams):
                raise ValueError("counts 形状需与[B, S, N]匹配。")
        else:
            counts = torch.full((batch, seq, cams), events_per, device=device, dtype=torch.long)

        frames = torch.zeros(
            (batch, seq, cams, self.channels, self.final_height, self.final_width),
            device=device,
            dtype=torch.float32,
        )

        for b in range(batch):
            for s in range(seq):
                for n in range(cams):
                    n_events = counts[b, s, n].item()
                    if n_events <= 0:
                        continue
                    n_events = min(n_events, events_per)
                    events = events_tensor[b, s, n, :n_events]
                    frame = self.events_to_frame(events, device)
                    frames[b, s, n] = frame
        return frames

    def events_to_frame(self, events, device):
        if events is None:
            return torch.zeros(
                (self.channels, self.final_height, self.final_width),
                device=device,
                dtype=torch.float32,
            )

        if isinstance(events, np.ndarray):
            events = torch.from_numpy(events)
        elif isinstance(events, list):
            events = torch.as_tensor(events)

        if not torch.is_tensor(events):
            raise TypeError("事件需为Tensor / ndarray / list。")

        events = events.to(device=device, dtype=torch.float32)
        if events.numel() == 0:
            return torch.zeros(
                (self.channels, self.final_height, self.final_width),
                device=device,
                dtype=torch.float32,
            )
        if events.dim() != 2 or events.size(-1) < 4:
            raise ValueError("每条事件应至少包含[x, y, t, polarity]四个字段。")

        x = events[:, 0]
        y = events[:, 1]
        t = events[:, 2]
        p = events[:, 3]

        x = x * self.scale_width - self.crop_w
        y = y * self.scale_height - self.crop_h

        valid = (x >= 0) & (x < self.final_width) & (y >= 0) & (y < self.final_height)
        if valid.sum() == 0:
            return torch.zeros(
                (self.channels, self.final_height, self.final_width),
                device=device,
                dtype=torch.float32,
            )

        x = x[valid].long()
        y = y[valid].long()
        t = t[valid]
        p = p[valid]

        if t.numel() == 0:
            return torch.zeros(
                (self.channels, self.final_height, self.final_width),
                device=device,
                dtype=torch.float32,
            )

        t_min = t.min()
        t_max = t.max()
        if t_max == t_min:
            bin_idx = torch.zeros_like(t, dtype=torch.long)
        else:
            norm_t = (t - t_min) / (t_max - t_min)
            bin_idx = torch.clamp((norm_t * self.bins).floor().long(), 0, self.bins - 1)

        polarity = torch.where(p > 0, torch.ones_like(p, dtype=torch.long), torch.zeros_like(p, dtype=torch.long))
        channel_idx = bin_idx + polarity * self.bins
        spatial_idx = y * self.final_width + x

        frame = torch.zeros(
            (self.channels, self.final_height * self.final_width),
            device=device,
            dtype=torch.float32,
        )
        ones = torch.ones_like(channel_idx, dtype=frame.dtype)
        frame.index_put_((channel_idx, spatial_idx), ones, accumulate=True)
        frame = frame.view(self.channels, self.final_height, self.final_width)

        if self.normalize and ones.numel() > 0:
            frame = frame / ones.numel()

        return frame
