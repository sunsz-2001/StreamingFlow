import os
import warnings
from PIL import Image

import numpy as np
import cv2
import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision

from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.utils.geometry_utils import transform_matrix
from functools import reduce
from nuscenes.utils.data_classes import RadarPointCloud

from streamingflow.utils.tools import ( gen_dx_bx, get_nusc_maps)

from streamingflow.utils.geometry import (
    resize_and_crop_image,
    update_intrinsics,
    calculate_birds_eye_view_parameters,
    convert_egopose_to_matrix_numpy,
    pose_vec2mat,
    mat2pose_vec,
    invert_matrix_egopose_numpy,
    get_global_pose
)
from streamingflow.utils.instance import convert_instance_mask_to_center_and_offset_label
import streamingflow.utils.sampler as trajectory_sampler

from streamingflow.utils.data_classes import LidarPointCloud
from streamingflow.utils.data_utils import voxelize_occupy, calc_displace_vector, point_in_hull_fast
import yaml


def locate_message(utimes, utime):
    i = np.searchsorted(utimes, utime)
    if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
        i -= 1
    return i

class FuturePredictionDataset(torch.utils.data.Dataset):
    SAMPLE_INTERVAL = 0.5 #SECOND
    def __init__(self, nusc, is_train, cfg):
        self.nusc = nusc
        self.dataroot = self.nusc.dataroot
        self.nusc_exp = NuScenesExplorer(nusc)
        try:
            self.nusc_can = NuScenesCanBus(dataroot=self.dataroot)
            self._has_can_bus = True
        except Exception as exc:
            warnings.warn(
                f"NuScenes CAN bus data unavailable at {self.dataroot}. Proceeding without CAN features. ({exc})"
            )
            self.nusc_can = None
            self._has_can_bus = False
        self.is_train = is_train
        self.cfg = cfg

        if self.is_train == 0:
            self.mode = 'train'
        elif self.is_train == 1:
            self.mode = 'val'
        elif self.is_train == 2:
            self.mode = 'test'
        else:
            raise NotImplementedError

        self.sequence_length = cfg.TIME_RECEPTIVE_FIELD + cfg.N_FUTURE_FRAMES
        self.receptive_field = cfg.TIME_RECEPTIVE_FIELD

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()
        self.indices = self.get_indices()

        # Image resizing and cropping
        self.augmentation_parameters = self.get_resizing_and_cropping_parameters()

        # Normalising input images
        self.normalise_image = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Bird's-eye view parameters
        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND
        )
        self.bev_resolution, self.bev_start_position, self.bev_dimension = (
            bev_resolution.numpy(), bev_start_position.numpy(), bev_dimension.numpy()
        )

        # Spatial extent in bird's-eye view, in meters
        self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])

        # The number of sampled trajectories
        self.n_samples = self.cfg.PLANNING.SAMPLE_NUM

        # HD-map feature extractor
        if self.cfg.SEMANTIC_SEG.HDMAP.ENABLED:
            self.nusc_maps = get_nusc_maps(self.cfg.DATASET.MAP_FOLDER)
        else:
            self.nusc_maps = {}
        self.scene2map = {}
        for sce in self.nusc.scene:
            log = self.nusc.get('log', sce['log_token'])
            self.scene2map[sce['name']] = log['location']
        self.save_dir = cfg.DATASET.SAVE_DIR

        # rv_config_filename = 'streamingflow/configs/nuscenes/data_preparing.yaml'
        # if yaml.__version__ >= '5.1':
        #     self.rv_config = yaml.load(open(rv_config_filename), Loader=yaml.FullLoader)
        # else:
        #     self.rv_config = yaml.load(open(rv_config_filename))

    def get_camera_multisweeps(self, rec_ref, window_sec, stride, max_sweeps, camera_set=None):
        """Collect non-keyframe camera sample_data as high-frequency observations within a time window.

        Returns lists (time-ascending):
            images_hi: List[Tensor (N,3,H,W)]
            intrinsics_hi: List[Tensor (N,3,3)]
            extrinsics_hi: List[Tensor (N,4,4)] (sensor_to_lidar)
            timestamps_hi: List[int] (absolute microseconds)
        """
        try:
            ref_sd_rec = self.nusc.get('sample_data', rec_ref['data']['LIDAR_TOP'])
        except Exception:
            return [], [], [], []

        t_ref = ref_sd_rec['timestamp']
        if window_sec is None:
            # default to match receptive field window
            window_sec = max(0.0, 0.5 * (self.receptive_field - 1))
        t_min = int(t_ref - window_sec * 1e6)

        # Build lidar_to_world at reference time (same as in get_input_data)
        lidar_pose = self.nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        yaw = Quaternion(lidar_pose['rotation']).yaw_pitch_roll[0]
        lidar_rotation = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)])
        lidar_translation = np.array(lidar_pose['translation'])[:, None]
        lidar_to_world = np.vstack([
            np.hstack((lidar_rotation.rotation_matrix, lidar_translation)),
            np.array([0, 0, 0, 1])
        ])

        # Use specified cameras or default to front camera only (to avoid cross-camera timestamp alignment complexity)
        cameras = camera_set if (camera_set is not None and len(camera_set) > 0) else ['CAM_FRONT']

        # Gather frames as (timestamp -> list of per-camera tuples)
        buckets = {}

        for cam in cameras:
            sd = self.nusc.get('sample_data', rec_ref['data'][cam])
            step = 0
            cur = sd
            # Walk prev chain to collect in-window, non-keyframe samples
            while True:
                if cur is None or cur['prev'] == "":
                    break
                cur = self.nusc.get('sample_data', cur['prev'])
                if cur is None:
                    break
                if cur['timestamp'] < t_min:
                    break
                if cur['is_key_frame']:
                    break
                if (step % max(1, int(stride))) != 0:
                    step += 1
                    continue

                # Load calibration and ego pose for this camera frame
                car_egopose = self.nusc.get('ego_pose', cur['ego_pose_token'])
                egopose_rotation = Quaternion(car_egopose['rotation']).inverse
                egopose_translation = -np.array(car_egopose['translation'])[:, None]
                world_to_car_egopose = np.vstack([
                    np.hstack((egopose_rotation.rotation_matrix, egopose_rotation.rotation_matrix @ egopose_translation)),
                    np.array([0, 0, 0, 1])
                ])

                sensor_sample = self.nusc.get('calibrated_sensor', cur['calibrated_sensor_token'])
                intrinsic = torch.Tensor(sensor_sample['camera_intrinsic'])
                sensor_rotation = Quaternion(sensor_sample['rotation'])
                sensor_translation = np.array(sensor_sample['translation'])[:, None]
                car_egopose_to_sensor = np.vstack([
                    np.hstack((sensor_rotation.rotation_matrix, sensor_translation)),
                    np.array([0, 0, 0, 1])
                ])
                car_egopose_to_sensor = np.linalg.inv(car_egopose_to_sensor)

                lidar_to_sensor = car_egopose_to_sensor @ world_to_car_egopose @ lidar_to_world
                sensor_to_lidar = torch.from_numpy(np.linalg.inv(lidar_to_sensor)).float()

                # Load and preprocess image
                image_filename = os.path.join(self.dataroot, cur['filename'])
                img = Image.open(image_filename)
                # Resize and crop
                img = resize_and_crop_image(
                    img, resize_dims=self.augmentation_parameters['resize_dims'], crop=self.augmentation_parameters['crop']
                )
                # Normalise image
                normalised_img = self.normalise_image(img)

                # Adjust intrinsics for resize/crop
                top_crop = self.augmentation_parameters['crop'][1]
                left_crop = self.augmentation_parameters['crop'][0]
                intrinsic_upd = update_intrinsics(
                    intrinsic, top_crop, left_crop,
                    scale_width=self.augmentation_parameters['scale_width'],
                    scale_height=self.augmentation_parameters['scale_height']
                )

                ts = cur['timestamp']
                if ts not in buckets:
                    buckets[ts] = {'images': [], 'intrinsics': [], 'extrinsics': []}
                buckets[ts]['images'].append(normalised_img.unsqueeze(0))  # (1,3,H,W)
                buckets[ts]['intrinsics'].append(intrinsic_upd.unsqueeze(0))  # (1,3,3)
                buckets[ts]['extrinsics'].append(sensor_to_lidar.unsqueeze(0))  # (1,4,4)

                step += 1

        if len(buckets) == 0:
            return [], [], [], []

        # Sort timestamps ascending and cap to max_sweeps most recent
        timestamps_sorted = sorted(buckets.keys())
        if max_sweeps is not None and max_sweeps > 0:
            timestamps_sorted = timestamps_sorted[-int(max_sweeps):]

        images_hi, intrinsics_hi, extrinsics_hi, timestamps_hi = [], [], [], []
        for ts in timestamps_sorted:
            pack = buckets[ts]
            # Stack along camera dimension N
            images_hi.append(torch.cat(pack['images'], dim=0).unsqueeze(0))       # (1,N,3,H,W)
            intrinsics_hi.append(torch.cat(pack['intrinsics'], dim=0).unsqueeze(0))  # (1,N,3,3)
            extrinsics_hi.append(torch.cat(pack['extrinsics'], dim=0).unsqueeze(0))  # (1,N,4,4)
            timestamps_hi.append(ts)

        # Concatenate along time dimension S_cam
        # Final shapes: [S_cam, N, ...]
        images_hi = [t.squeeze(0) for t in images_hi]
        intrinsics_hi = [t.squeeze(0) for t in intrinsics_hi]
        extrinsics_hi = [t.squeeze(0) for t in extrinsics_hi]

        return images_hi, intrinsics_hi, extrinsics_hi, timestamps_hi

    def get_scenes(self):
        # filter by scene split
        split = {'v1.0-trainval': {0: 'train', 1: 'val', 2: 'test'},
                 'v1.0-mini': {0: 'mini_train', 1: 'mini_val'},}[
            self.nusc.version
        ][self.is_train]

        can_blacklist = self.nusc_can.can_blacklist if self._has_can_bus else []
        blacklist = [419] + can_blacklist  # # scene-0419 does not have vehicle monitor data
        blacklist = ['scene-' + str(scene_no).zfill(4) for scene_no in blacklist]

        scenes = create_splits_scenes()[split][:]
        for scene_no in blacklist:
            if scene_no in scenes:
                scenes.remove(scene_no)

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples

    def get_indices(self):
        indices = []
        for index in range(len(self.ixes)):
            is_valid_data = True
            previous_rec = None
            current_indices = []
            for t in range(self.sequence_length):
                index_t = index + t
                # Going over the dataset size limit.
                if index_t >= len(self.ixes):
                    is_valid_data = False
                    break
                rec = self.ixes[index_t]
                # Check if scene is the same
                if (previous_rec is not None) and (rec['scene_token'] != previous_rec['scene_token']):
                    is_valid_data = False
                    break

                current_indices.append(index_t)
                previous_rec = rec

            if is_valid_data:
                indices.append(current_indices)

        return np.asarray(indices)

    def get_resizing_and_cropping_parameters(self):
        original_height, original_width = self.cfg.IMAGE.ORIGINAL_HEIGHT, self.cfg.IMAGE.ORIGINAL_WIDTH
        final_height, final_width = self.cfg.IMAGE.FINAL_DIM

        resize_scale = self.cfg.IMAGE.RESIZE_SCALE
        resize_dims = (int(original_width * resize_scale), int(original_height * resize_scale))
        resized_width, resized_height = resize_dims

        crop_h = self.cfg.IMAGE.TOP_CROP
        crop_w = int(max(0, (resized_width - final_width) / 2))
        # Left, top, right, bottom crops.
        crop = (crop_w, crop_h, crop_w + final_width, crop_h + final_height)

        if resized_width != final_width:
            print('Zero padding left and right parts of the image.')
        if crop_h + final_height != resized_height:
            print('Zero padding bottom part of the image.')

        return {'scale_width': resize_scale,
                'scale_height': resize_scale,
                'resize_dims': resize_dims,
                'crop': crop,
                }


    def get_input_data(self, rec):
        """
        Parameters
        ----------
            rec: nuscenes identifier for a given timestamp

        Returns
        -------
            images: torch.Tensor<float> (N, 3, H, W)
            intrinsics: torch.Tensor<float> (3, 3)
            extrinsics: torch.Tensor(N, 4, 4)
        """
        images = []
        intrinsics = []
        extrinsics = []
        depths = []
        cameras = self.cfg.IMAGE.NAMES
        events = None

        # The extrinsics we want are from the camera sensor to "flat egopose" as defined
        # https://github.com/nutonomy/nuscenes-devkit/blob/9b492f76df22943daf1dc991358d3d606314af27/python-sdk/nuscenes/nuscenes.py#L279
        # which corresponds to the position of the lidar.
        # This is because the labels are generated by projecting the 3D bounding box in this lidar's reference frame.

        # From lidar egopose to world.
        lidar_sample = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        lidar_pose = self.nusc.get('ego_pose', lidar_sample['ego_pose_token'])
        yaw = Quaternion(lidar_pose['rotation']).yaw_pitch_roll[0]
        lidar_rotation = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)])
        lidar_translation = np.array(lidar_pose['translation'])[:, None]
        lidar_to_world = np.vstack([
            np.hstack((lidar_rotation.rotation_matrix, lidar_translation)),
            np.array([0, 0, 0, 1])
        ])

        for cam in cameras:
            camera_sample = self.nusc.get('sample_data', rec['data'][cam])
            # Transformation from world to egopose
            car_egopose = self.nusc.get('ego_pose', camera_sample['ego_pose_token'])
            egopose_rotation = Quaternion(car_egopose['rotation']).inverse
            egopose_translation = -np.array(car_egopose['translation'])[:, None]
            world_to_car_egopose = np.vstack([
                np.hstack((egopose_rotation.rotation_matrix, egopose_rotation.rotation_matrix @ egopose_translation)),
                np.array([0, 0, 0, 1])
            ])

            # From egopose to sensor
            sensor_sample = self.nusc.get('calibrated_sensor', camera_sample['calibrated_sensor_token'])
            intrinsic = torch.Tensor(sensor_sample['camera_intrinsic'])
            sensor_rotation = Quaternion(sensor_sample['rotation'])
            sensor_translation = np.array(sensor_sample['translation'])[:, None]
            car_egopose_to_sensor = np.vstack([
                np.hstack((sensor_rotation.rotation_matrix, sensor_translation)),
                np.array([0, 0, 0, 1])
            ])
            car_egopose_to_sensor = np.linalg.inv(car_egopose_to_sensor)

            # Combine all the transformation.
            # From sensor to lidar.
            lidar_to_sensor = car_egopose_to_sensor @ world_to_car_egopose @ lidar_to_world
            sensor_to_lidar = torch.from_numpy(np.linalg.inv(lidar_to_sensor)).float()

            # Load image
            image_filename = os.path.join(self.dataroot, camera_sample['filename'])
            img = Image.open(image_filename)
            orig_img_size = img.size
            # Resize and crop
            img = resize_and_crop_image(
                img, resize_dims=self.augmentation_parameters['resize_dims'], crop=self.augmentation_parameters['crop']
            )
            # Normalise image
            normalised_img = self.normalise_image(img)

            # Combine resize/cropping in the intrinsics
            top_crop = self.augmentation_parameters['crop'][1]
            left_crop = self.augmentation_parameters['crop'][0]
            intrinsic = update_intrinsics(
                intrinsic, top_crop, left_crop,
                scale_width=self.augmentation_parameters['scale_width'],
                scale_height=self.augmentation_parameters['scale_height']
            )

            # Get Depth
            # Depth data should under the dataroot path 
            if self.cfg.LIFT.GT_DEPTH:
                if self.cfg.GEN.GEN_DEPTH:
                    depth = self.get_depth_from_lidar(lidar_sample,camera_sample)   # online without preprocessing
                else:

                    cam_depth = np.fromfile(os.path.join(self.dataroot,'depth_gt', os.path.split(camera_sample['filename'])[-1]+'.bin'), dtype=np.float32, count=-1).reshape(-1, 3)
                    coords = cam_depth[:, :2].astype(np.int16)

                    depth = np.ones([orig_img_size[1], orig_img_size[0]]) * (-1)
                    depth[coords[:, 1],coords[:, 0]] = cam_depth[:,2]

                depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
                depth = F.interpolate(depth, scale_factor=self.cfg.IMAGE.RESIZE_SCALE, mode='bilinear')
                depth = depth.squeeze()
                crop = self.augmentation_parameters['crop']
                depth = depth[crop[1]:crop[3], crop[0]:crop[2]]
                depth = torch.round(depth)
                depths.append(depth.unsqueeze(0).unsqueeze(0))

            images.append(normalised_img.unsqueeze(0).unsqueeze(0))
            intrinsics.append(intrinsic.unsqueeze(0).unsqueeze(0))
            extrinsics.append(sensor_to_lidar.unsqueeze(0).unsqueeze(0))

        images, intrinsics, extrinsics = (torch.cat(images, dim=1),
                                          torch.cat(intrinsics, dim=1),
                                          torch.cat(extrinsics, dim=1)
                                          )
        if len(depths) > 0:
            depths = torch.cat(depths, dim=1)

        if self.cfg.MODEL.MODALITY.USE_EVENT:
            # Load event observations independently of the RGB pipeline.
            events = self.get_event_data(rec)

        return images, intrinsics, extrinsics, depths, events

    def get_event_data(self, rec):
        """Load event observations for the given sample record."""
        if not self.cfg.MODEL.MODALITY.USE_EVENT:
            return None

        event_cameras = getattr(self.cfg.DATASET, 'EVENT_CAMERAS', None)
        if not event_cameras:
            event_cameras = self.cfg.IMAGE.NAMES

        event_tensors = []
        for cam in event_cameras:
            if cam not in rec['data']:
                raise KeyError(f"Camera '{cam}' not found in sample data; cannot load event tensor.")
            camera_sample = self.nusc.get('sample_data', rec['data'][cam])
            event_tensor = self._load_event_tensor(camera_sample)
            event_tensors.append(event_tensor.unsqueeze(0).unsqueeze(0))

        if len(event_tensors) == 0:
            raise RuntimeError("No event tensors were loaded for the current sample.")

        # Concatenate along the camera dimension to match downstream expectations: [1, N_evt, C, H, W]
        return torch.cat(event_tensors, dim=1)

    def _load_event_tensor(self, camera_sample):
        """加载与某个相机样本对齐的事件帧，并做与图像一致的预处理。

        Args:
            camera_sample (dict): nuScenes 的相机 `sample_data` 记录。

        Returns:
            torch.Tensor: 形状为 [C, H, W] 的事件张量，已完成：
                - 路径映射（由相机文件名映射到事件文件名）
                - npz 读取与通道维度归一化
                - 归一化（按 `DATASET.EVENTS_NORMALIZATION`）
                - resize 与裁剪（与图像保持一致）

        说明：当找不到事件文件且允许回退时，返回全零张量，通道数来源于配置；否则抛出异常。
        """

        # 事件根目录：若未配置且允许回退，则返回全零事件张量；否则报错。
        events_root = getattr(self.cfg.DATASET, 'EVENTS_ROOT', '')
        if events_root == '':
            if getattr(self.cfg.DATASET, 'EVENTS_FALLBACK_ZERO', True):
                channels = getattr(self.cfg.MODEL.EVENT, 'IN_CHANNELS', 0)
                if channels <= 0:
                    channels = 2 * getattr(self.cfg.MODEL.EVENT, 'BINS', 10)
                h, w = self.cfg.IMAGE.FINAL_DIM
                return torch.zeros(channels, h, w)
            raise ValueError("USE_EVENT is True but DATASET.EVENTS_ROOT is not specified.")

        # 路径映射：复用图像的相对路径，替换后缀得到事件文件路径。
        rel_path = camera_sample['filename']
        suffix = getattr(self.cfg.DATASET, 'EVENTS_FILE_SUFFIX', '.npz')
        event_rel = os.path.splitext(rel_path)[0] + suffix
        event_path = os.path.join(events_root, event_rel)

        if not os.path.exists(event_path):
            if getattr(self.cfg.DATASET, 'EVENTS_FALLBACK_ZERO', True):
                channels = getattr(self.cfg.MODEL.EVENT, 'IN_CHANNELS', 0)
                if channels <= 0:
                    channels = 2 * getattr(self.cfg.MODEL.EVENT, 'BINS', 10)
                h, w = self.cfg.IMAGE.FINAL_DIM
                return torch.zeros(channels, h, w)
            raise FileNotFoundError(f"Event frame not found at {event_path}")

        # 读取 npz：优先使用 `frame` 键；否则取第一个数组。
        data = np.load(event_path)
        if 'frame' in data:
            frame = data['frame']
        else:
            frame = data[data.files[0]]

        # 维度规范：
        # - 常见格式为 [2, T, H, W]（正负极性两路），这里将其摊平为 [C, H, W]
        # - 若已为 [C, H, W] 则直接使用
        if frame.ndim == 4 and frame.shape[0] == 2:
            frame = frame.reshape(-1, frame.shape[-2], frame.shape[-1])
        elif frame.ndim == 3:
            pass
        else:
            raise ValueError(f"Unexpected event frame shape {frame.shape} in {event_path}")

        # 转 tensor 并按配置做强度归一化
        frame = torch.from_numpy(frame.astype(np.float32))
        norm = float(getattr(self.cfg.DATASET, 'EVENTS_NORMALIZATION', 255.0))
        if norm > 0:
            frame = frame / norm

        # 与图像相同的几何增强：先 resize 再 crop，保持与视觉模态对齐
        frame = frame.unsqueeze(0)  # [1, C, H, W]
        resize_dims = self.augmentation_parameters['resize_dims']
        frame = F.interpolate(frame, size=(resize_dims[1], resize_dims[0]), mode='bilinear', align_corners=False)
        crop = self.augmentation_parameters['crop']
        frame = frame[:, :, crop[1]:crop[3], crop[0]:crop[2]]
        frame = frame.squeeze(0)  # [C, H, W]

        return frame

    def _get_top_lidar_pose(self, rec):
        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0]
        rot = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse
        return trans, rot

    def get_depth_from_lidar(self, lidar_sample, cam_sample):
  
        points, coloring, im = self.nusc_exp.map_pointcloud_to_image(lidar_sample['token'], cam_sample['token'])
        cam_file_name = os.path.split(cam_sample['filename'])[-1]
        tmp_cam = np.zeros((self.cfg.IMAGE.ORIGINAL_HEIGHT, self.cfg.IMAGE.ORIGINAL_WIDTH))
        points = points.astype(int)
        tmp_cam[points[1, :], points[0,:]] = coloring

        return tmp_cam


    def get_birds_eye_view_label_multisweep(self, rec, instance_map, in_pred):
        translation, rotation = self._get_top_lidar_pose(rec)

        nsweeps = 10
        segmentation = np.zeros((nsweeps, self.bev_dimension[0], self.bev_dimension[1]))
        pedestrian = np.zeros((nsweeps, self.bev_dimension[0], self.bev_dimension[1]))
        # Background is ID 0
        instance = np.zeros((nsweeps, self.bev_dimension[0], self.bev_dimension[1]))

        sample_data_token = rec['data']['LIDAR_TOP']
        curr_sample_data = self.nusc.get('sample_data', sample_data_token)

        multi_frames_instance_boxes = []
        annotations = []
        for ann_token in rec['anns']:
            # Filter out all non vehicle instances

            annotation = self.nusc.get('sample_annotation', ann_token)

            if self.cfg.DATASET.FILTER_INVISIBLE_VEHICLES and int(annotation['visibility_token']) == 1 and in_pred is False:
                continue
            if in_pred is True and annotation['instance_token'] not in instance_map:
                continue
            ann_rec = self.nusc.get('sample_annotation', ann_token)
            category_name = ann_rec['category_name']
            instance_token = ann_rec['instance_token']                
            instance_boxes, instance_all_times, _, _ = LidarPointCloud. \
                get_instance_boxes_multisweep_sample_data(self.nusc, curr_sample_data,
                                                        instance_token,
                                                        nsweeps_back=0,
                                                        nsweeps_forward=nsweeps)
            multi_frames_instance_boxes.append(instance_boxes)
            annotations.append(annotation)

        for time in range(nsweeps):           
            instance_boxes = [x[time] for x in multi_frames_instance_boxes]
            for box_idx in range(len(instance_boxes)):
                instance_box = instance_boxes[box_idx]
                annotation = annotations[box_idx]

                if instance_box is not None:
                    # NuScenes filter
                    if 'vehicle' in annotation['category_name']:
                        if annotation['instance_token'] not in instance_map:
                            instance_map[annotation['instance_token']] = len(instance_map) + 1
                        instance_id = instance_map[annotation['instance_token']]
                        poly_region, z = self._get_poly_region_in_image_box_input(instance_box, translation, rotation)
                        cv2.fillPoly(instance[time], [poly_region], instance_id)
                        cv2.fillPoly(segmentation[time], [poly_region], 1.0)
                    elif 'human' in annotation['category_name']:
                        if annotation['instance_token'] not in instance_map:
                            instance_map[annotation['instance_token']] = len(instance_map) + 1
                        poly_region, z = self._get_poly_region_in_image_box_input(instance_box, translation, rotation)
                        cv2.fillPoly(pedestrian[time], [poly_region], 1.0)

        return segmentation, instance, pedestrian, instance_map, instance_all_times


    def _get_poly_region_in_image_box_input(self, instance_box, ego_translation, ego_rotation):

        box = instance_box
        # box.translate(ego_translation)
        # box.rotate(ego_rotation)

        pts = box.bottom_corners()[:2].T
        pts = np.round((pts - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0) / self.bev_resolution[:2]).astype(np.int32)
        pts[:, [1, 0]] = pts[:, [0, 1]]

        z = box.bottom_corners()[2, 0]
        return pts, z

    def get_birds_eye_view_label(self, rec, instance_map, in_pred):
        translation, rotation = self._get_top_lidar_pose(rec)
        segmentation = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
        pedestrian = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
        # Background is ID 0
        instance = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))

        for annotation_token in rec['anns']:
            # Filter out all non vehicle instances
            annotation = self.nusc.get('sample_annotation', annotation_token)

            if self.cfg.DATASET.FILTER_INVISIBLE_VEHICLES and int(annotation['visibility_token']) == 1 and in_pred is False:
                continue
            if in_pred is True and annotation['instance_token'] not in instance_map:
                continue

            # NuScenes filter
            if 'vehicle' in annotation['category_name']:
                if annotation['instance_token'] not in instance_map:
                    instance_map[annotation['instance_token']] = len(instance_map) + 1
                instance_id = instance_map[annotation['instance_token']]
                poly_region, z = self._get_poly_region_in_image(annotation, translation, rotation)
                cv2.fillPoly(instance, [poly_region], instance_id)
                cv2.fillPoly(segmentation, [poly_region], 1.0)
            elif 'human' in annotation['category_name']:
                if annotation['instance_token'] not in instance_map:
                    instance_map[annotation['instance_token']] = len(instance_map) + 1
                poly_region, z = self._get_poly_region_in_image(annotation, translation, rotation)
                cv2.fillPoly(pedestrian, [poly_region], 1.0)


        return segmentation, instance, pedestrian, instance_map

    def _get_poly_region_in_image(self, instance_annotation, ego_translation, ego_rotation):
        box = Box(
            instance_annotation['translation'], instance_annotation['size'], Quaternion(instance_annotation['rotation'])
        )
        box.translate(ego_translation)
        box.rotate(ego_rotation)

        pts = box.bottom_corners()[:2].T
        pts = np.round((pts - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0) / self.bev_resolution[:2]).astype(np.int32)
        pts[:, [1, 0]] = pts[:, [0, 1]]

        z = box.bottom_corners()[2, 0]
        return pts, z

    def get_label(self, rec, instance_map, in_pred):
        segmentation_np, instance_np, pedestrian_np, instance_map = \
            self.get_birds_eye_view_label(rec, instance_map, in_pred)
        segmentation = torch.from_numpy(segmentation_np).long().unsqueeze(0).unsqueeze(0)
        instance = torch.from_numpy(instance_np).long().unsqueeze(0)
        pedestrian = torch.from_numpy(pedestrian_np).long().unsqueeze(0).unsqueeze(0)

        return segmentation, instance, pedestrian, instance_map

    def get_label_multi_sweep(self, rec, instance_map, in_pred):
        segmentation_np, instance_np, pedestrian_np, instance_map, timestamp = \
            self.get_birds_eye_view_label_multisweep(rec, instance_map, in_pred)
        segmentation = torch.from_numpy(segmentation_np).long().unsqueeze(0).unsqueeze(0)
        instance = torch.from_numpy(instance_np).long().unsqueeze(0)
        pedestrian = torch.from_numpy(pedestrian_np).long().unsqueeze(0).unsqueeze(0)

        return segmentation, instance, pedestrian, instance_map, timestamp


    def get_future_egomotion(self, rec, index):
        rec_t0 = rec

        # Identity
        future_egomotion = np.eye(4, dtype=np.float32)

        if index < len(self.ixes) - 1:
            rec_t1 = self.ixes[index + 1]

            if rec_t0['scene_token'] == rec_t1['scene_token']:
                egopose_t0 = self.nusc.get(
                    'ego_pose', self.nusc.get('sample_data', rec_t0['data']['LIDAR_TOP'])['ego_pose_token']
                )
                egopose_t1 = self.nusc.get(
                    'ego_pose', self.nusc.get('sample_data', rec_t1['data']['LIDAR_TOP'])['ego_pose_token']
                )

                egopose_t0 = convert_egopose_to_matrix_numpy(egopose_t0)
                egopose_t1 = convert_egopose_to_matrix_numpy(egopose_t1)

                future_egomotion = invert_matrix_egopose_numpy(egopose_t1).dot(egopose_t0)
                future_egomotion[3, :3] = 0.0
                future_egomotion[3, 3] = 1.0

        future_egomotion = torch.Tensor(future_egomotion).float()

        # Convert to 6DoF vector
        future_egomotion = mat2pose_vec(future_egomotion)
        return future_egomotion.unsqueeze(0)

    def get_static_future_egomotion(self):
        # Identity
        future_egomotion = np.eye(4, dtype=np.float32)

        future_egomotion[3, :3] = 0.0
        future_egomotion[3, 3] = 1.0

        future_egomotion = torch.Tensor(future_egomotion).float()

        # Convert to 6DoF vector
        future_egomotion = mat2pose_vec(future_egomotion)
        return future_egomotion.unsqueeze(0)

    def get_trajectory_sampling(self, rec=None, sample_indice=None):
        if rec is None and sample_indice is None:
            raise ValueError("No valid input rec or token")
        if rec is None and sample_indice is not None:
            rec = self.ixes[sample_indice]

        if self._has_can_bus:
            ref_scene = self.nusc.get("scene", rec['scene_token'])

            pose_msgs = self.nusc_can.get_messages(ref_scene['name'],'pose')
            pose_uts = [msg['utime'] for msg in pose_msgs]
            steer_msgs = self.nusc_can.get_messages(ref_scene['name'], 'steeranglefeedback')
            steer_uts = [msg['utime'] for msg in steer_msgs]

            ref_utime = rec['timestamp']
            pose_index = locate_message(pose_uts, ref_utime)
            pose_data = pose_msgs[pose_index]
            steer_index = locate_message(steer_uts, ref_utime)
            steer_data = steer_msgs[steer_index]

            v0 = pose_data["vel"][0]  # [0] means longitudinal velocity  m/s
            steering = steer_data["value"]

            location = self.scene2map[ref_scene['name']]
            flip_flag = True if location.startswith('singapore') else False
            if flip_flag:
                steering *= -1
            Kappa = 2 * steering / 2.588
        else:
            v0 = 0.0
            Kappa = 0.0

        # initial state
        T0 = np.array([0.0, 1.0])  # define front
        N0 = np.array([1.0, 0.0]) if Kappa <= 0 else np.array([-1.0, 0.0])  # define side

        t_start = 0  # second
        t_end = self.cfg.N_FUTURE_FRAMES * self.SAMPLE_INTERVAL  # second
        t_interval = self.SAMPLE_INTERVAL / 10
        tt = np.arange(t_start, t_end + t_interval, t_interval)
        sampled_trajectories_fine = trajectory_sampler.sample(v0, Kappa, T0, N0, tt, self.n_samples)
        sampled_trajectories = sampled_trajectories_fine[:, ::10]
        return sampled_trajectories

    def voxelize_hd_map(self, rec):
        dx, bx, _ = gen_dx_bx(self.cfg.LIFT.X_BOUND, self.cfg.LIFT.Y_BOUND, self.cfg.LIFT.Z_BOUND)
        stretch = [self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1]]
        dx, bx = dx[:2].numpy(), bx[:2].numpy()

        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        map_name = self.scene2map[self.nusc.get('scene', rec['scene_token'])['name']]

        rot = Quaternion(egopose['rotation']).rotation_matrix
        rot = np.arctan2(rot[1,0], rot[0,0]) # in radian
        center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])

        box_coords = (
            center[0],
            center[1],
            stretch[0]*2,
            stretch[1]*2
        ) # (x_center, y_center, width, height)
        canvas_size = (
                int(self.cfg.LIFT.X_BOUND[1] * 2 / self.cfg.LIFT.X_BOUND[2]),
                int(self.cfg.LIFT.Y_BOUND[1] * 2 / self.cfg.LIFT.Y_BOUND[2])
        )

        elements = self.cfg.SEMANTIC_SEG.HDMAP.ELEMENTS
        hd_features = self.nusc_maps[map_name].get_map_mask(box_coords, rot * 180 / np.pi , elements, canvas_size=canvas_size)
        #traffic = self.hd_traffic_light(map_name, center, stretch, dx, bx, canvas_size)
        #return torch.from_numpy(np.concatenate((hd_features, traffic), axis=0)[None]).float()
        hd_features = torch.from_numpy(hd_features[None]).float()
        hd_features = torch.transpose(hd_features,-2,-1) # (y,x) replace horizontal and vertical coordinates
        return hd_features

    def hd_traffic_light(self, map_name, center, stretch, dx, bx, canvas_size):

        roads = np.zeros(canvas_size)
        my_patch = (
            center[0] - stretch[0],
            center[1] - stretch[1],
            center[0] + stretch[0],
            center[1] + stretch[1],
        )
        tl_token = self.nusc_maps[map_name].get_records_in_patch(my_patch, ['traffic_light'], mode='intersect')['traffic_light']
        polys = []
        for token in tl_token:
            road_token =self.nusc_maps[map_name].get('traffic_light', token)['from_road_block_token']
            pt = self.nusc_maps[map_name].get('road_block', road_token)['polygon_token']
            polygon = self.nusc_maps[map_name].extract_polygon(pt)
            polys.append(np.array(polygon.exterior.xy).T)

        def get_rot(h):
            return torch.Tensor([
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ])
        # convert to local coordinates in place
        rot = get_rot(np.arctan2(center[3], center[2])).T
        for rowi in range(len(polys)):
            polys[rowi] -= center[:2]
            polys[rowi] = np.dot(polys[rowi], rot)

        for la in polys:
            pts = (la - bx) / dx
            pts = np.int32(np.around(pts))
            cv2.fillPoly(roads, [pts], 1)

        return roads[None]

    def get_gt_trajectory(self, rec, ref_index):
        n_output = self.cfg.N_FUTURE_FRAMES
        gt_trajectory = np.zeros((n_output+1, 3), np.float64)

        egopose_cur = get_global_pose(rec, self.nusc, inverse=True)

        for i in range(n_output+1):
            index = ref_index + i
            if index < len(self.ixes):
                rec_future = self.ixes[index]

                egopose_future = get_global_pose(rec_future, self.nusc, inverse=False)

                egopose_future = egopose_cur.dot(egopose_future)
                theta = quaternion_yaw(Quaternion(matrix=egopose_future))

                origin = np.array(egopose_future[:3, 3])

                gt_trajectory[i, :] = [origin[0], origin[1], theta]

        if gt_trajectory[-1][0] >= 2:
            command = 'RIGHT'
        elif gt_trajectory[-1][0] <= -2:
            command = 'LEFT'
        else:
            command = 'FORWARD'

        return gt_trajectory, command

    def get_routed_map(self, gt_points):
        dx, bx, _ = gen_dx_bx(self.cfg.LIFT.X_BOUND, self.cfg.LIFT.Y_BOUND, self.cfg.LIFT.Z_BOUND)
        dx, bx = dx[:2].numpy(), bx[:2].numpy()

        canvas_size = (
            int(self.cfg.LIFT.X_BOUND[1] * 2 / self.cfg.LIFT.X_BOUND[2]),
            int(self.cfg.LIFT.Y_BOUND[1] * 2 / self.cfg.LIFT.Y_BOUND[2])
        )

        roads = np.zeros(canvas_size)
        W = 1.85
        pts = np.array([
            [-4.084 / 2. + 0.5, W / 2.],
            [4.084 / 2. + 0.5, W / 2.],
            [4.084 / 2. + 0.5, -W / 2.],
            [-4.084 / 2. + 0.5, -W / 2.],
        ])
        pts = (pts - bx) / dx
        pts[:, [0, 1]] = pts[:, [1, 0]]

        pts = np.int32(np.around(pts))
        cv2.fillPoly(roads, [pts], 1)

        gt_points = gt_points[:-1].numpy()
        # 坐标原点在左上角
        target = pts.copy()
        target[:,0] = pts[:,0] + gt_points[0] / dx[0]
        target[:,1] = pts[:,1] - gt_points[1] / dx[1]
        target = np.int32(np.around(target))
        cv2.fillPoly(roads, [target], 1)
        return roads

    def __len__(self):
        return len(self.indices)

    def get_points_from_multisweeps(self, index):
        rec_curr_token = self.ixes[self.indices[index][self.receptive_field - 1]]

        curr_sample_data = self.nusc.get('sample_data', rec_curr_token['data']['LIDAR_TOP'])

        nsweeps_back = int((self.receptive_field - 1) * 0.5 / 0.05) 
        # frame_skip = int(( nsweeps_back / 5 ))
        frame_skip = self.cfg.DATASET.FRAME_SKIP

        num_sweeps = nsweeps_back
        # Get the synchronized point clouds
        all_pc, all_times = LidarPointCloud.from_file_multisweep_bf_sample_data(self.nusc, curr_sample_data,
                                                                                nsweeps_back=nsweeps_back,nsweeps_forward=0)
        # Store point cloud of each sweep
        pc = all_pc.points
        pc = np.concatenate([pc,all_times.reshape(1,-1)],axis=0)
        _, sort_idx = np.unique(all_times, return_index=True)
        unique_times = all_times[np.sort(sort_idx)]  # Preserve the item order in unique_times
        num_sweeps = len(unique_times)
        
        pc_list = []
        
        for tid in range(num_sweeps):
            _time = unique_times[tid]
            points_idx = np.where(all_times == _time)[0]
            _pc = pc[:, points_idx]
            pc_list.append(_pc.T)
            
        self.aggregate_lidar_sweeps = True
        selected_times = unique_times[0:nsweeps_back:frame_skip]
        # Reorder the pc, and skip sample frames if wanted
        # tmp_pc_list_1 = pc_list[0:nsweeps_back:frame_skip]
        if not self.aggregate_lidar_sweeps:
            tmp_pc_list_1 = pc_list[::frame_skip]
        else:
            tmp_pc_list_1 = []
            for i in range(0, len(pc_list), frame_skip):
                # 选取当前区间的数组
                arrays_to_concat = pc_list[i:i+frame_skip]
                # 沿着第一个维度（行）拼接数组
                concatenated_array = np.concatenate(arrays_to_concat, axis=0)
                # 将拼接后的数组添加到新列表中
                tmp_pc_list_1.append(concatenated_array)
        
        selected_times = unique_times[::frame_skip]

        selected_times = selected_times[::-1]
        tmp_pc_list_1 = tmp_pc_list_1[::-1]

        point_dim = pc.shape[0]
        filled_pc = []
        for pc_array in tmp_pc_list_1:
            if pc_array.shape[0] == 0:
                pc_array = np.zeros((1, point_dim), dtype=np.float32)
            else:
                pc_array = pc_array.astype(np.float32, copy=False)
            filled_pc.append(pc_array)

        lidar_timestamps = (
            curr_sample_data['timestamp'] - np.array(selected_times) * 1e6
        ).astype(np.int64)

        return filled_pc, lidar_timestamps

    def __getitem__(self, index):
        """
        Returns
        -------
            data: dict with the following keys:
                image: torch.Tensor<float> (T, N, 3, H, W)
                    normalised cameras images with T the sequence length, and N the number of cameras.
                intrinsics: torch.Tensor<float> (T, N, 3, 3)
                    intrinsics containing resizing and cropping parameters.
                extrinsics: torch.Tensor<float> (T, N, 4, 4)
                    6 DoF pose from world coordinates to camera coordinates.
                segmentation: torch.Tensor<int64> (T, 1, H_bev, W_bev)
                    (H_bev, W_bev) are the pixel dimensions in bird's-eye view.
                instance: torch.Tensor<int64> (T, 1, H_bev, W_bev)
                centerness: torch.Tensor<float> (T, 1, H_bev, W_bev)
                offset: torch.Tensor<float> (T, 2, H_bev, W_bev)
                flow: torch.Tensor<float> (T, 2, H_bev, W_bev)
                future_egomotion: torch.Tensor<float> (T, 6)
                    6 DoF egomotion t -> t+1

        """
        data = {}
        keys = ['image', 'intrinsics', 'extrinsics', 'depths',
                'segmentation', 'instance', 'centerness', 'offset', 'flow', 'pedestrian',
                'future_egomotion', 'hdmap', 'gt_trajectory', 'indices' , 'camera_timestamp', 'target_timestamp', 'lidar_timestamp', 'points'
                ]
        if self.cfg.MODEL.MODALITY.USE_EVENT:
            keys.append('event')
        for key in keys:
            data[key] = []
        # if self.cfg.MODEL.MODALITY.USE_RADAR:
        #     data['radar_pointclouds']=[]
        # if self.cfg.MODEL.MODALITY.USE_LIDAR:
        #     if self.cfg.MODEL.LIDAR.USE_RANGE:
        #         data['range_clouds']=[]    
        #     if self.cfg.MODEL.LIDAR.USE_STPN or self.cfg.MODEL.LIDAR.USE_BESTI: 
        #         data['padded_voxel_points']=[]
        #         data['lidar_timestamp']=[]
        #     else:
        #         data['points'] = []

        instance_map = {}
        

        rec_ref = self.ixes[self.indices[index][self.receptive_field - 1]]
        ref_sd_rec = self.nusc.get('sample_data',rec_ref['data']['LIDAR_TOP'])
        current_time = ref_sd_rec['timestamp']
        ref_pose_rec = self.nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = self.nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Homogeneous transformation matrix from global to _current_ ego car frame.
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                            inverse=True)   

       
        # Loop over all the frames in the sequence.
        data['status'] = 'valid'

        for i, index_t in enumerate(self.indices[index]):
            if i >= self.receptive_field:
                in_pred = True
            else:
                in_pred = False
            rec = self.ixes[index_t]

            if i < self.receptive_field:

                images, intrinsics, extrinsics, depths, events = self.get_input_data(rec)
                data['image'].append(images)
                data['intrinsics'].append(intrinsics)
                data['extrinsics'].append(extrinsics)
                data['depths'].append(depths)
                if self.cfg.MODEL.MODALITY.USE_EVENT:
                    # Keep the event tensor for each history step so the model receives the full temporal window
                    data['event'].append(events)
                data['camera_timestamp'].append(rec['timestamp'])

            
            # hd_map_feature = self.voxelize_hd_map(rec)

            future_egomotion = self.get_future_egomotion(rec, index_t)                

            if not self.cfg.DATASET.USE_MULTISWEEP:
                segmentation, instance, pedestrian, instance_map = self.get_label(rec, instance_map, in_pred)
                data['target_timestamp'].append(rec['timestamp'])
                data['segmentation'].append(segmentation)
                data['instance'].append(instance)
                data['pedestrian'].append(pedestrian)
                data['future_egomotion'].append(future_egomotion)
                # data['hdmap'].append(hd_map_feature)
                data['indices'].append(index_t)
            else:
                if i >= self.receptive_field - 1 and i< self.sequence_length - 1:
                    try:
                        segmentation_ms, instance_ms, pedestrian_ms, instance_map_ms, timestamp_ms = self.get_label_multi_sweep(rec, instance_map, in_pred)
                        nsweeps = segmentation_ms.shape[2]
                        for sweep in range(nsweeps):
                            data['segmentation'].append(segmentation_ms[:,:,sweep,:,:])
                            data['instance'].append(instance_ms[:,sweep,:,:])
                            data['pedestrian'].append(pedestrian_ms[:,:,sweep,:,:])
                            if sweep==nsweeps - 1:
                                data['future_egomotion'].append(future_egomotion)
                            else:
                                data['future_egomotion'].append(self.get_static_future_egomotion())
                            data['target_timestamp'].append(rec['timestamp'] - 1e6 * timestamp_ms[sweep])                  
                    except:
                        data['status'] = 'invalid'
                else:
                    segmentation, instance, pedestrian, instance_map = self.get_label(rec, instance_map, in_pred)
                    data['segmentation'].append(segmentation)
                    data['instance'].append(instance)
                    data['pedestrian'].append(pedestrian)
                    data['future_egomotion'].append(future_egomotion)         
                    data['target_timestamp'].append(rec['timestamp'])

            if self.cfg.MODEL.MODALITY.USE_RADAR:
                radar_pointcloud = self.get_radar_data(rec,nsweeps=1,min_distance=2.2)
                data['radar_pointclouds'].append(radar_pointcloud)    
            if self.cfg.MODEL.LIDAR.USE_RANGE:
                lidar_range_cloud = self.get_lidar_range_data(rec,nsweeps=1,min_distance=2.2)
                data['range_clouds'].append(lidar_range_cloud)
            if i == self.cfg.TIME_RECEPTIVE_FIELD-1:
                gt_trajectory, command = self.get_gt_trajectory(rec, index_t)
                data['gt_trajectory'] = torch.from_numpy(gt_trajectory).float()
                data['command'] = command
                trajs = self.get_trajectory_sampling(rec)
                data['sample_trajectory'] = torch.from_numpy(trajs).float()
        



        points, lidar_times = self.get_points_from_multisweeps(index)
        data['lidar_timestamp'] = lidar_times

        for point in points:
            assert point.shape[0] <= 350000
            pad_rows = np.zeros((350000 - point.shape[0], point.shape[1]))  
            point = np.append(point, pad_rows, axis=0)
            data['points'].append(point)

        for key, value in data.items():
            if key in ['image', 'intrinsics', 'extrinsics', 'depths', 'segmentation', 'instance', 'future_egomotion', 'pedestrian', 'event']:
                if key == 'depths' and self.cfg.LIFT.GT_DEPTH is False:
                    continue
                data[key] = torch.cat(value, dim=0)

        if self.cfg.MODEL.LIDAR.USE_STPN or self.cfg.MODEL.LIDAR.USE_BESTI: 
            data['padded_voxel_points'] = torch.cat(data['padded_voxel_points'], dim=0)
        if self.cfg.MODEL.LIDAR.USE_RANGE:
            data['range_clouds'] = torch.cat(data['range_clouds'], dim=0)   
        if self.cfg.MODEL.MODALITY.USE_RADAR:
            data['radar_pointclouds'] = torch.cat(data['radar_pointclouds'], dim=0)       
        
        # Optional: high-frequency camera multisweep (asynchronous camera observations)
        if getattr(self.cfg.DATASET, 'CAMERA_MULTISWEEP', False):
            window_sec = self.cfg.DATASET.CAMERA_WINDOW_SEC
            stride = self.cfg.DATASET.CAMERA_SWEEP_STRIDE
            max_sweeps = self.cfg.DATASET.CAMERA_MAX_SWEEPS
            camera_set = self.cfg.DATASET.CAMERA_SET_HI if hasattr(self.cfg.DATASET, 'CAMERA_SET_HI') else None
            images_hi, intrinsics_hi, extrinsics_hi, timestamps_hi = self.get_camera_multisweeps(
                rec_ref, window_sec, stride, max_sweeps, camera_set
            )
            if len(timestamps_hi) > 0:
                # Stack along time dimension
                data['image_hi'] = torch.stack(images_hi, dim=0)            # [S_cam, N, 3, H, W]
                data['intrinsics_hi'] = torch.stack(intrinsics_hi, dim=0)   # [S_cam, N, 3, 3]
                data['extrinsics_hi'] = torch.stack(extrinsics_hi, dim=0)   # [S_cam, N, 4, 4]
                data['camera_timestamp_hi'] = np.array(timestamps_hi)

        data['target_point'] = torch.tensor([0., 0.])
        instance_centerness, instance_offset, instance_flow = convert_instance_mask_to_center_and_offset_label(
            data['instance'], data['future_egomotion'],
            num_instances=len(instance_map), ignore_index=self.cfg.DATASET.IGNORE_INDEX, subtract_egomotion=True,
            spatial_extent=self.spatial_extent,
        )
        data['centerness'] = instance_centerness
        data['offset'] = instance_offset
        data['flow'] = instance_flow
        
        data['camera_timestamp'] = np.array(data['camera_timestamp'])
        data['camera_timestamp'] = (data['camera_timestamp'] - current_time) / 1e6

        data['lidar_timestamp'] = np.array(data['lidar_timestamp'])
        data['lidar_timestamp'] = (data['lidar_timestamp'] - current_time) / 1e6

        data['target_timestamp'] = np.array(data['target_timestamp'])
        data['target_timestamp'] = (data['target_timestamp'] - current_time) / 1e6

        # Normalise hi-frequency camera timestamps if present
        if 'camera_timestamp_hi' in data:
            data['camera_timestamp_hi'] = (data['camera_timestamp_hi'] - current_time) / 1e6

        return data
