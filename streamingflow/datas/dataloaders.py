import torch
import torch.utils.data
from nuscenes.nuscenes import NuScenes
from streamingflow.datas.NuscenesData import FuturePredictionDataset
from lyft_dataset_sdk.lyftdataset import LyftDataset
from streamingflow.datas.LyftData import FuturePredictionDatasetLyft
from streamingflow.datas.DSECData import DatasetDSEC

import os
import numpy as np


def dsec_collate_fn(batch):
    """
    自定义 collate 函数，用于 DSEC 数据集。
    处理变长字段（如点云数据、事件网格等），这些字段在不同样本间大小可能不同。
    """
    from torch.utils.data.dataloader import default_collate
    
    # 变长字段列表（这些字段在不同样本间大小可能不同，保持为 list）
    # - 点云数据：points 的形状是 [N, 4]，其中 N 在不同样本间不同
    # - 事件数据：events_grid, evs_stmp 是列表
    # - 标注框数据：不同帧中物体数量不同（gt_boxes, gt_obj_ids 等）
    variable_length_keys = {
        'points', 'padded_voxel_points',  # 点云相关
        'events_grid', 'evs_stmp', 'events',  # 事件相关
        'gt_boxes', 'gt_len', 'gt_obj_ids', 'gt_boxes_prosed',  # 标注框相关
        'gt_bboxes_3d', 'gt_labels_3d',  # 检测标签（变长，每个样本物体数量不同）
        'gt_names', 'num_lidar_pts', 'num_radar_pts',  # 其他可能的变长字段
        'flow_data',  # 流式数据
        'metas',  # metas包含类型对象（box_type_3d），需要特殊处理
    }
    
    # 需要保留的字符串字段（用于评估时的样本ID）
    preserved_string_keys = {'sequence_name', 'frame_id'}
    
    # 过滤掉字符串字段，分离变长字段
    filtered_batch = []
    variable_length_data = {key: [] for key in variable_length_keys}
    preserved_string_data = {key: [] for key in preserved_string_keys}
    
    for sample in batch:
        if isinstance(sample, dict):
            filtered_sample = {}
            for key, value in sample.items():
                # 保留用于评估的字符串字段
                if key in preserved_string_keys:
                    preserved_string_data[key].append(value)
                    continue
                
                # 跳过其他字符串字段
                if isinstance(value, str):
                    continue
                # 跳过包含字符串的 numpy 数组（但保留 frame_id 如果是 numpy）
                if isinstance(value, np.ndarray) and value.dtype.kind in ('U', 'S', 'O'):
                    continue
                
                # 变长字段单独处理
                if key in variable_length_keys:
                    variable_length_data[key].append(value)
                else:
                    filtered_sample[key] = value
            
            filtered_batch.append(filtered_sample)
        else:
            filtered_batch.append(sample)
    
    # 使用默认 collate 处理固定长度字段
    try:
        collated = default_collate(filtered_batch) if filtered_batch else {}
        
        # 将变长字段作为 list 添加回去
        for key, values in variable_length_data.items():
            if values and any(v is not None for v in values):
                # 对于gt_labels_3d，确保转换为torch.Tensor
                if key == 'gt_labels_3d':
                    converted_values = []
                    for v in values:
                        if v is None:
                            converted_values.append(torch.tensor([], dtype=torch.long))
                        elif isinstance(v, np.ndarray):
                            converted_values.append(torch.from_numpy(v).long())
                        elif isinstance(v, torch.Tensor):
                            converted_values.append(v.long())
                        else:
                            converted_values.append(torch.tensor(v, dtype=torch.long))
                    collated[key] = converted_values
                elif key == 'metas':
                    # metas 需要展平：从 [[meta1], [meta2], ...] 转换为 [meta1, meta2, ...]
                    # 因为每个样本返回的是 [meta]，collate 后变成嵌套的 list
                    flattened_metas = []
                    for v in values:
                        # 递归展平，直到找到 dict
                        while isinstance(v, (list, tuple)) and len(v) > 0:
                            v = v[0]
                        # 确保最终是 dict
                        if isinstance(v, dict):
                            flattened_metas.append(v)
                        else:
                            # 如果仍然不是 dict，尝试包装或使用默认值
                            flattened_metas.append(v if isinstance(v, dict) else {})
                    collated[key] = flattened_metas
                else:
                    collated[key] = values  # 保持为 list，不 stack
        
        # 添加保留的字符串字段
        for key, values in preserved_string_data.items():
            if values:
                collated[key] = values  # 保持为 list
        
        # 输出数据尺寸信息
        if collated:
            # print("[DataLoader] Batch sizes:")
            # 输出 event frames 尺寸
            if 'event' in collated and isinstance(collated['event'], dict):
                if 'frames' in collated['event']:
                    event_frames = collated['event']['frames']
                    if torch.is_tensor(event_frames):
                        print(f"  event.frames: {event_frames.shape}")
            
            # 输出 images 尺寸
            if 'images' in collated:
                images = collated['images']
                if isinstance(images, dict):
                    for cam_name, img_list in images.items():
                        if isinstance(img_list, (list, tuple)) and len(img_list) > 0:
                            if torch.is_tensor(img_list[0]):
                                print(f"  images[{cam_name}]: list of {len(img_list)} tensors, each {img_list[0].shape}")
                            elif isinstance(img_list[0], np.ndarray):
                                print(f"  images[{cam_name}]: list of {len(img_list)} arrays, each {img_list[0].shape}")
                elif torch.is_tensor(images):
                    print(f"  images: {images.shape}")
            
            # 输出 points 信息（如果存在）
            if 'points' in collated:
                points = collated['points']
                if isinstance(points, list) and len(points) > 0:
                    if torch.is_tensor(points[0]):
                        print(f"  points: list of {len(points)} tensors, first shape: {points[0].shape}")
                    elif isinstance(points[0], np.ndarray):
                        print(f"  points: list of {len(points)} arrays, first shape: {points[0].shape}")
                elif torch.is_tensor(points):
                    print(f"  points: {points.shape}")
            
            # 输出其他关键张量字段的尺寸
            key_fields = ['intrinsics', 'extrinsics', 'future_egomotion']
            for key in key_fields:
                if key in collated:
                    value = collated[key]
                    if torch.is_tensor(value):
                        print(f"  {key}: {value.shape}")
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if torch.is_tensor(sub_value):
                                print(f"  {key}[{sub_key}]: {sub_value.shape}")
        
        return collated
    except Exception as e:
        print(f"[Warn] DSEC Collate 失败: {e}")
        if filtered_batch:
            print(f"[Debug] Batch keys: {filtered_batch[0].keys() if isinstance(filtered_batch[0], dict) else 'N/A'}")
            # 打印形状信息帮助调试
            if isinstance(filtered_batch[0], dict):
                for key, value in filtered_batch[0].items():
                    if torch.is_tensor(value) or isinstance(value, np.ndarray):
                        print(f"  {key}: {type(value).__name__} {getattr(value, 'shape', 'N/A')}")
        raise

def prepare_dataloaders(cfg, return_dataset=False):
    if cfg.DATASET.NAME == 'nuscenes':
        # 28130 train and 6019 val
        dataroot = cfg.DATASET.DATAROOT
        nusc = NuScenes(version='v1.0-{}'.format(cfg.DATASET.VERSION), dataroot=dataroot, verbose=True)
        traindata = FuturePredictionDataset(nusc, 0, cfg)
        valdata = FuturePredictionDataset(nusc, 1, cfg)

        if cfg.DATASET.VERSION == 'mini':
            traindata.indices = traindata.indices[:10]
            # valdata.indices = valdata.indices[:10]

        nworkers = cfg.N_WORKERS
        trainloader = torch.utils.data.DataLoader(
            traindata, batch_size=cfg.BATCHSIZE, shuffle=True, num_workers=nworkers, pin_memory=True, drop_last=True
        )
        valloader = torch.utils.data.DataLoader(
            valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=nworkers, pin_memory=True, drop_last=False)
    elif cfg.DATASET.NAME == 'nuscenesmultisweep':
        # 28130 train and 6019 val
        dataroot = cfg.DATASET.DATAROOT
        nusc = NuScenes(version='v1.0-{}'.format(cfg.DATASET.VERSION), dataroot=dataroot, verbose=True)
        traindata = FuturePredictionDatasetMultiSweep(nusc, 0, cfg)
        valdata = FuturePredictionDatasetMultiSweep(nusc, 1, cfg)

        if cfg.DATASET.VERSION == 'mini':
            traindata.indices = traindata.indices[:10]
            # valdata.indices = valdata.indices[:10]

        nworkers = cfg.N_WORKERS
        trainloader = torch.utils.data.DataLoader(
            traindata, batch_size=cfg.BATCHSIZE, shuffle=True, num_workers=nworkers, pin_memory=True, drop_last=True
        )
        valloader = torch.utils.data.DataLoader(
            valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=nworkers, pin_memory=True, drop_last=False)
    elif cfg.DATASET.NAME == 'lyft':
        # train contains 22680 samples
        # we split in 16506 6174
        # dataroot = os.path.join(cfg.DATASET.DATAROOT, 'trainval')
        dataroot = cfg.DATASET.DATAROOT
        nusc = LyftDataset(data_path=dataroot,
                           json_path=os.path.join(dataroot, 'train_data'),
                           verbose=True)
        traindata = FuturePredictionDatasetLyft(nusc, 1, cfg)
        valdata = FuturePredictionDatasetLyft(nusc, 0, cfg)

        if cfg.DATASET.VERSION == 'mini':
            traindata.indices = traindata.indices[:10]
            # valdata.indices = valdata.indices[:10]

        nworkers = cfg.N_WORKERS
        trainloader = torch.utils.data.DataLoader(
            traindata, batch_size=cfg.BATCHSIZE, shuffle=True, num_workers=nworkers, pin_memory=True, drop_last=True
        )
        valloader = torch.utils.data.DataLoader(
            valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=nworkers, pin_memory=True, drop_last=False)
    elif cfg.DATASET.NAME == 'dsec':
        traindata = DatasetDSEC(cfg, cfg, is_train=True)
        valdata = DatasetDSEC(cfg, cfg, is_train=False)

        nworkers = cfg.N_WORKERS
        trainloader = torch.utils.data.DataLoader(
            traindata, batch_size=cfg.BATCHSIZE, shuffle=True, num_workers=nworkers, 
            pin_memory=True, drop_last=True, collate_fn=dsec_collate_fn
        )
        valloader = torch.utils.data.DataLoader(
            valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=nworkers, 
            pin_memory=True, drop_last=False, collate_fn=dsec_collate_fn)

    else:
        raise NotImplementedError

    if return_dataset:
        return trainloader, valloader, traindata, valdata
    else:
        return trainloader, valloader
