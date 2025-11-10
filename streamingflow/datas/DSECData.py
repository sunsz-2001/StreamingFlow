import os
from PIL import Image
import pickle 
import io
from collections import defaultdict
import copy

import numpy as np
import cv2
import torch
import torchvision
import torch.nn.functional as F
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from lyft_dataset_sdk.lyftdataset import LyftDataset
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.data_classes import PointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.data_classes import Box
from functools import reduce
from streamingflow.utils.geometry import (
    update_intrinsics,
    calculate_birds_eye_view_parameters,
    convert_egopose_to_matrix_numpy,
    pose_vec2mat,
    mat2pose_vec,
    invert_matrix_egopose_numpy,
)
from streamingflow.utils.instance import convert_instance_mask_to_center_and_offset_label
from streamingflow.utils.data_classes import LidarPointCloud
from streamingflow.utils.data_utils import voxelize_occupy, calc_displace_vector, point_in_hull_fast
import yaml

from streamingflow.utils import box_utils, common_utils
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.core.bbox import get_box_type

def resize_and_crop_image(img, resize_dims, crop):
    # Bilinear resizing followed by cropping
    # OpenCV 默认使用双线性插值进行缩放
    img_resized = cv2.resize(img, resize_dims, interpolation=cv2.INTER_LINEAR)
    
    # Crop the image (crop is a tuple of (x, y, w, h))
    x, y, w, h = crop
    img_cropped = img_resized[y:y+h, x:x+w]
    
    return img_cropped

def range_projection(current_vertex, proj_H=64, proj_W=900, fov_up=3.0, fov_down=-25.0, max_range=50, min_range=2):
  """ Project a pointcloud into a spherical projection, range image.
    Args:
      current_vertex: raw point clouds
    Returns:
      proj_vertex: each pixel contains the corresponding point (x, y, z, depth)
  """
  # laser parameters
  fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
  fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
  fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians
  
  # get depth of all points

  depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
  current_vertex = current_vertex[(depth > min_range) & (depth < max_range)]  # get rid of [0, 0, 0] points
  depth = depth[(depth > min_range) & (depth < max_range)]
  
  # get scan components
  scan_x = current_vertex[:, 0]
  scan_y = current_vertex[:, 1]
  scan_z = current_vertex[:, 2]
  intensity = current_vertex[:, 3]
  
  # get angles of all points
  yaw = -np.arctan2(scan_y, scan_x)
  pitch = np.arcsin(scan_z / depth)
  
  # get projections in image coords
  proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
  proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]
  
  # scale to image size using angular resolution
  proj_x *= proj_W  # in [0.0, W]
  proj_y *= proj_H  # in [0.0, H]
  
  # round and clamp for use as index
  proj_x = np.floor(proj_x)
  proj_x = np.minimum(proj_W - 1, proj_x)
  proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
  proj_x_orig = np.copy(proj_x)
  
  proj_y = np.floor(proj_y)
  proj_y = np.minimum(proj_H - 1, proj_y)
  proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
  proj_y_orig = np.copy(proj_y)
  
  # order in decreasing depth
  order = np.argsort(depth)[::-1]
  depth = depth[order]
  intensity = intensity[order]
  proj_y = proj_y[order]
  proj_x = proj_x[order]
  
  scan_x = scan_x[order]
  scan_y = scan_y[order]
  scan_z = scan_z[order]
  
  indices = np.arange(depth.shape[0])
  indices = indices[order]
  
  proj_range = np.full((proj_H, proj_W), -1,
                       dtype=np.float32)  # [H,W] range (-1 is no data)
  proj_vertex = np.full((proj_H, proj_W, 4), -1,
                        dtype=np.float32)  # [H,W] index (-1 is no data)
  proj_idx = np.full((proj_H, proj_W), -1,
                     dtype=np.int32)  # [H,W] index (-1 is no data)
  proj_intensity = np.full((proj_H, proj_W), -1,
                           dtype=np.float32)  # [H,W] index (-1 is no data)
  
  proj_range[proj_y, proj_x] = depth
  proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z, depth]).T
  proj_idx[proj_y, proj_x] = indices
  proj_intensity[proj_y, proj_x] = intensity
  
  return proj_vertex

def locate_message(utimes, utime):
    i = np.searchsorted(utimes, utime)
    if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
        i -= 1
    return i

class DatasetDSEC(torch.utils.data.Dataset):
    def __init__(self, data_cfg,  cfg, is_train=True):
        self.data_cfg = data_cfg
        self.is_train = is_train
        self.cfg = cfg
        self.dataroot = self.data_cfg.DATASET.DATAROOT
        self.use_image = self.cfg.MODEL.MODALITY.USE_CAMERA
        self.use_event = getattr(self.cfg.MODEL.MODALITY, 'USE_EVENT', False)

        self.mode = 'train' if self.is_train else 'val'
        self.box_type_3d, self.box_mode_3d = get_box_type('LiDAR')
        self.sequence_length = cfg.TIME_RECEPTIVE_FIELD + cfg.N_FUTURE_FRAMES
        self.receptive_field = cfg.TIME_RECEPTIVE_FIELD
        self.n_future_frames = cfg.N_FUTURE_FRAMES
        self.scenes = self.get_scenes()
        self.voxel_size = cfg.VOXEL.VOXEL_SIZE
        self.area_extents = np.array(cfg.VOXEL.AREA_EXTENTS)
        self.event_scale = .5
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
        self.class_names = ['Vehicle', 'Pedestrian']



    def get_scenes(self):

        split_dir = os.path.join(self.dataroot, 'detection_' + self.mode + '_sample_new.txt')
        sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]

        self.infos = []
        self.init_infos(sample_sequence_list)
        return sample_sequence_list
    
    def init_infos(self, sample_sequence_list):
        
        waymo_infos = []
        num_skipped_infos = 0
        for k in range(len(sample_sequence_list)):
        # for k in range(10):
            sequence_name = os.path.splitext(sample_sequence_list[k])[0]
            if self.mode == 'train':
                info_path = os.path.join(self.dataroot, sequence_name, ('%s.pkl' % sequence_name)).replace('.pkl', '_interpolate_fov_bbox_lidar_check.pkl')
            else:
                info_path = os.path.join(self.dataroot, sequence_name, ('%s.pkl' % sequence_name)).replace('.pkl', '_interpolate_fov_bbox_lidar_check.pkl')    
                
            if not os.path.exists(info_path):
                num_skipped_infos += 1
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                
            ######## 100 FPS ########
            if self.mode == 'train':
                new_infos = []
                infos = infos[:-1] # TODO 
                # infos = infos[:1]
                
                for i in range(len(infos)):
                    new_info = copy.deepcopy(infos[i])
                    event_grid_paths = []
                    event_paths = []
                    img_infos = infos[i]['image']
                    lidar_path = os.path.join(self.dataroot, infos[i]['lidar_path'])
                    event_init_path = os.path.join(self.dataroot, 
                                                   img_infos['image_0_path'].split('/')[0], 
                                                   'events_split',
                                                   )
                    event_grid_init_path = os.path.join(self.dataroot,
                                                   img_infos['image_0_path'].split('/')[0], 
                                                   'voxel',
                                                   )
                    event_file_number = int(img_infos['event_0_path'].split('/')[-1][:6]) + 2
                    event_file_name = str(event_file_number).zfill(6)
                    
                    for j in range(100):
                        event_paths.append(os.path.join(event_init_path, event_file_name + '_' + str(j) + '.npz'))
                    event_paths = np.array(event_paths)
                    new_info['event_paths'] = event_paths
                    for j in range(10):
                        event_grid_paths.append(os.path.join(event_grid_init_path, event_file_name + '_' + str(j+1) + '.npz'))
                    event_grid_paths = np.array(event_grid_paths)
                    new_info['event_grid_paths'] = event_grid_paths
                    
                    new_info['seq_annos'] = np.array(infos[i]['annos'])
                    
                    new_infos.append(new_info)
                infos = new_infos
            else:
                new_infos = []
                infos = infos[:-1] # TODO 

                for i in range(len(infos)):
                    new_info = copy.deepcopy(infos[i])
                    event_paths = []
                    event_grid_paths = []
                    # new_info['annos'] = infos[i]['annos'][j]
                    img_infos = infos[i]['image']
                    lidar_path = os.path.join(self.dataroot, infos[i]['lidar_path'])
                    event_init_path = os.path.join(self.dataroot,
                                                   img_infos['image_0_path'].split('/')[0], 
                                                   'events_split',
                                                   )
                    event_grid_init_path = os.path.join(self.dataroot,
                                                   img_infos['image_0_path'].split('/')[0], 
                                                   'voxel',
                                                   )
                    event_file_number = int(img_infos['event_0_path'].split('/')[-1][:6]) + 2
                    event_file_name = str(event_file_number).zfill(6)
                    
                    for j in range(100):
                        event_paths.append(os.path.join(event_init_path, event_file_name + '_' + str(j) + '.npz'))
                    event_paths = np.array(event_paths)
                    new_info['event_paths'] = event_paths

                    for j in range(10):
                        event_grid_paths.append(os.path.join(event_grid_init_path, event_file_name + '_' + str(j+1) + '.npz'))
                    event_grid_paths = np.array(event_grid_paths)
                    new_info['event_grid_paths'] = event_grid_paths
                    
                    new_info['seq_annos'] = np.array(infos[i]['annos'])
                    new_infos.append(new_info)
                infos = new_infos
            waymo_infos.extend(infos)
        
        self.infos.extend(waymo_infos[:])
        
        
    def check_sequence_name_with_all_version(self, seq_file):
        if '_with_camera_labels' not in seq_file and not os.path.exists(seq_file):
            seq_file = seq_file[:-9] + '_with_camera_labels.tfrecord'
        if '_with_camera_labels' in seq_file and not os.path.exists(seq_file):
            seq_file = seq_file.replace('_with_camera_labels', '')

        return seq_file


    def get_resizing_and_cropping_parameters(self):
        original_height, original_width = self.cfg.IMAGE.ORIGINAL_HEIGHT, self.cfg.IMAGE.ORIGINAL_WIDTH
        final_height, final_width = self.cfg.IMAGE.FINAL_DIM

        resize_scale = self.cfg.IMAGE.RESIZE_SCALE
        resize_dims = (int(original_width * resize_scale), int(original_height * resize_scale))
        resized_width, resized_height = resize_dims

        crop_h = int(max(0, (resized_height - final_height) / 2))
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

    def get_lidar_range_data(self, sample_rec, nsweeps, min_distance):
        """
        Returns at most nsweeps of lidar in the ego frame.
        Returned tensor is 5(x, y, z, reflectance, dt, ring_index) x N
        Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
        """


        if self.cfg.GEN.GEN_RANGE:
            # points = np.zeros((5, 0))
            points = np.zeros((5, 0))
            V = 35000 * nsweeps
            # Get reference pose and timestamp.
            ref_sd_token = sample_rec['data']['LIDAR_TOP']
            ref_sd_rec = self.nusc.get('sample_data', ref_sd_token)
            ref_pose_rec = self.nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
            ref_cs_rec = self.nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
            ref_time = 1e-6 * ref_sd_rec['timestamp']

            # Homogeneous transformation matrix from global to _current_ ego car frame.
            car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                                inverse=True)

            # Aggregate current and previous sweeps.
            sample_data_token = sample_rec['data']['LIDAR_TOP']
            current_sd_rec = self.nusc.get('sample_data', sample_data_token)
            sample_sd_rec = current_sd_rec
            for _ in range(nsweeps):
                # Load up the pointcloud and remove points close to the sensor.
                current_pc = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, current_sd_rec['filename']))
                current_pc.remove_close(min_distance)

                # Get past pose.
                current_pose_rec = self.nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
                global_from_car = transform_matrix(current_pose_rec['translation'],
                                                    Quaternion(current_pose_rec['rotation']), inverse=False)

                # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                current_cs_rec = self.nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
                car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                    inverse=False)

                # Fuse four transformation matrices into one and perform transform.
                trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
                current_pc.transform(trans_matrix)

                # Add time vector which can be used as a temporal feature.
                time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
                times = time_lag * np.ones((1, current_pc.nbr_points()))

                new_points = np.concatenate((current_pc.points, times), 0)
                points = np.concatenate((points, new_points), 1)

                if current_sd_rec['prev'] == '':
                    break
                else:
                    current_sd_rec = self.nusc.get('sample_data', current_sd_rec['prev'])
                    
                    
                    
            if points.shape[1] > V:
                    # # print('lidar_data.shape[0]', lidar_data.shape[0])
                    # np.random.shuffle(lidar_data)
                points = points[:,:V]
            elif points.shape[0] < V:
                points = np.pad(points,[(0,0),(0, V-points.shape[1])],mode='constant')  
                # Abort if there are no previous sweeps.
            range_view = range_projection(points.transpose(1,0),  
                                        self.rv_config['range_image']['height'], self.rv_config['range_image']['width'],
                                        self.rv_config['range_image']['fov_up'], self.rv_config['range_image']['fov_down'],
                                        self.rv_config['range_image']['max_range'], self.rv_config['range_image']['min_range'])

            rv_file_name = os.path.split(sample_sd_rec['filename'])[-1] 
            # os.makedirs(os.path.join(self.dataroot,'range_nusc',sample_sd_rec['channel']), exist_ok=True)                           
            # np.save(os.path.join(self.dataroot,'range_nusc', sample_sd_rec['channel'], rv_file_name+'.npy'),range_view)
        else:
            sample_data_token = sample_rec['data']['LIDAR_TOP']
            sample_sd_rec = self.nusc.get('sample_data', sample_data_token)
            range_view = np.load(os.path.join(self.dataroot,'range_nusc',sample_sd_rec['channel'], os.path.split(sample_sd_rec['filename'])[-1]+'.npy'))                           
        return torch.from_numpy(range_view).unsqueeze(0).to(torch.float32)
    
    def get_depth_from_lidar(self, lidar_sample, cam_sample):
  
        points, coloring, im = self.nusc_exp.map_pointcloud_to_image(lidar_sample['token'], cam_sample['token'])
        cam_file_name = os.path.split(cam_sample['filename'])[-1]
        tmp_cam = np.zeros((self.cfg.IMAGE.ORIGINAL_HEIGHT, self.cfg.IMAGE.ORIGINAL_WIDTH))
        points = points.astype(int)
        tmp_cam[points[1, :], points[0,:]] = coloring

        return tmp_cam

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
            # if self.cfg.LIFT.GT_DEPTH:
            #     if self.cfg.GEN.GEN_DEPTH:
            #         depth = self.get_depth_from_lidar(lidar_sample,camera_sample)   # online without preprocessing
            #     else:
            #         import time
            #         t1 = time.time()
            #         depth = np.load(os.path.join(self.dataroot,'depth_nusc',camera_sample['channel'], os.path.split(camera_sample['filename'])[-1]+'.npy'))
            #         t2 = time.time()
            #         print('get one depth image:', t2-t1)
                
            #     depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
            #     depth = F.interpolate(depth, scale_factor=self.cfg.IMAGE.RESIZE_SCALE, mode='bilinear')
            #     depth = depth.squeeze()
            #     crop = self.augmentation_parameters['crop']
            #     depth = depth[crop[1]:crop[3], crop[0]:crop[2]]
            #     depth = torch.round(depth)
            #     depths.append(depth.unsqueeze(0).unsqueeze(0))

            images.append(normalised_img.unsqueeze(0).unsqueeze(0))
            intrinsics.append(intrinsic.unsqueeze(0).unsqueeze(0))
            extrinsics.append(sensor_to_lidar.unsqueeze(0).unsqueeze(0))

        images, intrinsics, extrinsics = (torch.cat(images, dim=1),
                                          torch.cat(intrinsics, dim=1),
                                          torch.cat(extrinsics, dim=1)
                                          )
        # if len(depths) > 0:
        #     depths = torch.cat(depths, dim=1)

        return images, intrinsics, extrinsics
    def get_lidar_data(self, sample_rec, nsweeps, min_distance):
        """
        Returns at most nsweeps of lidar in the ego frame.
        Returned tensor is 5(x, y, z, reflectance, dt, ring_index) x N
        Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
        """
        # points = np.zeros((5, 0))
        points = np.zeros((5, 0))
        V = 35000 * nsweeps
        # Get reference pose and timestamp.
        ref_sd_token = sample_rec['data']['LIDAR_TOP']
        ref_sd_rec = self.nusc.get('sample_data', ref_sd_token)
        ref_pose_rec = self.nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = self.nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Homogeneous transformation matrix from global to _current_ ego car frame.
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                            inverse=True)

        # Aggregate current and previous sweeps.
        sample_data_token = sample_rec['data']['LIDAR_TOP']
        current_sd_rec = self.nusc.get('sample_data', sample_data_token)
        for _ in range(nsweeps):
            # Load up the pointcloud and remove points close to the sensor.
            current_pc = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, current_sd_rec['filename']))
            current_pc.remove_close(min_distance)

            # Get past pose.
            current_pose_rec = self.nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(current_pose_rec['translation'],
                                                Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = self.nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
            current_pc.transform(trans_matrix)

            # Add time vector which can be used as a temporal feature.
            time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
            times = time_lag * np.ones((1, current_pc.nbr_points()))

            new_points = np.concatenate((current_pc.points, times), 0)
            points = np.concatenate((points, new_points), 1)

            if current_sd_rec['prev'] == '':
                break
            else:
                current_sd_rec = self.nusc.get('sample_data', current_sd_rec['prev'])
                
                
                
        if points.shape[1] > V:
                # # print('lidar_data.shape[0]', lidar_data.shape[0])
                # np.random.shuffle(lidar_data)
            points = points[:,:V]
        elif points.shape[0] < V:
            points = np.pad(points,[(0,0),(0, V-points.shape[1])],mode='constant')  
            # Abort if there are no previous sweeps.

        return points
        # return torch.from_numpy(range_projection(points.transpose(1,0),  
        #                             self.rv_config['range_image']['height'], self.rv_config['range_image']['width'],
        #                             self.rv_config['range_image']['fov_up'], self.rv_config['range_image']['fov_down'],
        #                             self.rv_config['range_image']['max_range'], self.rv_config['range_image']['min_range'])).unsqueeze(0).to(torch.float32)
    
    
    def _get_top_lidar_pose(self, rec):
        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        # ref_cs_rec = self.nusc.get('calibrated_sensor', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['calibrated_sensor_token'])
        # yaw = Quaternion(ref_cs_rec['rotation']).yaw_pitch_roll[0]
        # rot = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse

        trans = -np.array(egopose['translation'])
        yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0]
        rot = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse
        return trans, rot

    def get_birds_eye_view_label(self, rec, instance_map):
        translation, rotation = self._get_top_lidar_pose(rec)
        segmentation = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
        # pedestrian = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
        # Background is ID 0
        instance = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))

        for annotation_token in rec['anns']:
            # Filter out all non vehicle instances
            annotation = self.nusc.get('sample_annotation', annotation_token)
            category_name = annotation['category_name']
            if not self.is_lyft:
                # NuScenes filter
                # if 'vehicle' not in annotation['category_name']:
                #     continue
                if self.cfg.DATASET.FILTER_INVISIBLE_VEHICLES and int(annotation['visibility_token']) == 1:
                    continue
            else:
                # Lyft filter
                if annotation['category_name'] not in ['bus', 'car', 'construction_vehicle', 'trailer', 'truck']:
                    continue


            if annotation['instance_token'] not in instance_map:
                instance_map[annotation['instance_token']] = len(instance_map) + 1
            instance_id = instance_map[annotation['instance_token']]

            if not self.is_lyft:
                instance_attribute = int(annotation['visibility_token'])
            else:
                instance_attribute = 0

            poly_region, z = self._get_poly_region_in_image(annotation, translation, rotation)
            cv2.fillPoly(instance, [poly_region], instance_id)
            
            cv2.fillPoly(segmentation, [poly_region], 1.0)
            
            # cv2.fillPoly(z_position, [poly_region], z)
            # cv2.fillPoly(attribute_label, [poly_region], instance_attribute)

        return segmentation, instance, instance_map

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

    def get_infos_and_points(self, idx_list):
        infos, points = [], []
        for i in idx_list:
            lidar_path = self.infos[i]["lidar_path"]
            #################### For disparity input #################
            lidar_path = os.path.join(self.dataroot, self.infos[i]['lidar_path'])
            disparity_path = os.path.join(self.dataroot, self.infos[i]['disparity_path'])

            disp_point = np.load(disparity_path)
            zeros = np.zeros((disp_point.shape[0], 1))
            disp_point = np.concatenate((disp_point, zeros), axis=1)

            current_point = np.load(lidar_path)
            ones = np.ones((current_point.shape[0], 1))
            current_point = np.concatenate((current_point, ones), axis=1)
            
            current_point = np.concatenate((disp_point, current_point), axis=0)
                    
            infos.append(self.infos[i])
            points.append(current_point)

        return infos, points
    
    # def get_label(self, rec, instance_map, in_pred):
    #     segmentation_np, instance_np, instance_map = \
    #         self.get_birds_eye_view_label(rec, instance_map)
    #     segmentation = torch.from_numpy(segmentation_np).long().unsqueeze(0).unsqueeze(0)
    #     instance = torch.from_numpy(instance_np).long().unsqueeze(0)

    #     return segmentation, instance, instance_map


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

    def __len__(self):
        return len(self.infos)

    def get_temporal_voxels(self,index):
        if self.cfg.GEN.GEN_VOXELS:

            
            rec_curr_token = self.ixes[self.indices[index][self.receptive_field - 1]]
            curr_sample_data = self.nusc.get('sample_data', rec_curr_token['data']['LIDAR_TOP'])
            nsweeps_back = 5
            frame_skip = 1
            num_sweeps = nsweeps_back
            # Get the synchronized point clouds
            all_pc, all_times = LidarPointCloud.from_file_multisweep_bf_sample_data(self.nusc, curr_sample_data,
                                                                                    nsweeps_back=nsweeps_back,nsweeps_forward=0)
            # Store point cloud of each sweep
            pc = all_pc.points
            _, sort_idx = np.unique(all_times, return_index=True)
            unique_times = all_times[np.sort(sort_idx)]  # Preserve the item order in unique_times
            num_sweeps = len(unique_times)
           
            pc_list = []
            


            for tid in range(num_sweeps):
                _time = unique_times[tid]
                points_idx = np.where(all_times == _time)[0]
                _pc = pc[:, points_idx]
                pc_list.append(_pc.T)
                

            selected_times = unique_times
            # Reorder the pc, and skip sample frames if wanted
            tmp_pc_list_1 = pc_list
            # selected_times = -selected_times[::-1] + 0.5 * (self.receptive_field - 1)
         
            tmp_pc_list_1 = tmp_pc_list_1[::-1]


            num_past_pcs = len(tmp_pc_list_1)

            # print('rec_field: ', self.receptive_field)
            # print('rec_samples: ', num_past_pcs)
            assert num_past_pcs == 5
            
            # latest_points = tmp_pc_list_1[0][:,:2].T
            # import cv2 as cv
            # os.makedirs('./output_lidar', exist_ok=True)
            # img = np.zeros((256, 256, 3), np.uint8)
            # point_size = 1
            # point_color = (0, 0, 255) # BGR
            # thickness = 4 # 可以为 0 、4、8
            # cv2.circle(img, latest_points.astype(int), point_size, point_color, thickness)
            # curr_sample_token = data['sample_token'][self.cfg.TIME_RECEPTIVE_FIELD - 1]       
            # cv2.imwrite(os.path.join('./output_lidar',curr_sample_token ) + '.png',img)

            # Voxelize the input point clouds, and compute the ground truth displacement vectors
            padded_voxel_points_list = list()  # This contains the compact representation of voxelization, as in the paper
        
            
            for i in range(num_past_pcs):
                vox = voxelize_occupy(pc_list[i], voxel_size=self.cfg.VOXEL.VOXEL_SIZE, extents=np.array(self.cfg.VOXEL.AREA_EXTENTS))
                padded_voxel_points_list.append(vox)

            # Compile the batch of voxels, so that they can be fed into the network
            padded_voxel_points = np.stack(padded_voxel_points_list, 0).astype(np.float32)
        
            # os.makedirs(os.path.join(self.dataroot,'voxels_nusc',curr_sample_data['channel']), exist_ok=True)                           
            # np.save(os.path.join(self.dataroot,'voxels_nusc',curr_sample_data['channel'], os.path.split(curr_sample_data['filename'])[-1]+'.npy'),padded_voxel_points)
        else:
            rec_curr_token = self.ixes[self.indices[index][self.receptive_field - 1]]
            curr_sample_data = self.nusc.get('sample_data', rec_curr_token['data']['LIDAR_TOP'])
            padded_voxel_points = np.load(os.path.join(self.dataroot,'voxels_nusc',curr_sample_data['channel'], os.path.split(curr_sample_data['filename'])[-1]+'.npy'))   

        padded_voxel_points = torch.from_numpy(padded_voxel_points)
        return padded_voxel_points, selected_times


    def __getitem_DEP__(self, index):
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
                'future_egomotion', 'hdmap', 'gt_trajectory', 'indices' , 'camera_timestamp' ,
                ]
        for key in keys:
            data[key] = []
        if self.cfg.MODEL.MODALITY.USE_RADAR:
            data['radar_pointclouds']=[]
        if self.cfg.MODEL.MODALITY.USE_LIDAR:
            if self.cfg.MODEL.LIDAR.USE_RANGE:
                data['range_clouds']=[]    
            if self.cfg.MODEL.LIDAR.USE_STPN or self.cfg.MODEL.LIDAR.USE_BESTI: 
                data['padded_voxel_points']=[]
                data['lidar_timestamp']=[]

        instance_map = {}
        # Loop over all the frames in the sequence.
        for i, index_t in enumerate(self.indices[index]):
            if i >= self.receptive_field:
                in_pred = True
            else:
                in_pred = False
            rec = self.ixes[index_t]
            data['camera_timestamp'].append(rec['timestamp'])
            if i < self.receptive_field:
                images, intrinsics, extrinsics = self.get_input_data(rec)
                data['image'].append(images)
                data['intrinsics'].append(intrinsics)
                data['extrinsics'].append(extrinsics)
            segmentation, instance, instance_map = self.get_label(rec, instance_map, in_pred)

            future_egomotion = self.get_future_egomotion(rec, index_t)
            # hd_map_feature = self.voxelize_hd_map(rec)
            
            data['segmentation'].append(segmentation)
            data['instance'].append(instance)
            data['future_egomotion'].append(future_egomotion)
            # data['hdmap'].append(hd_map_feature)
            data['indices'].append(index_t)

            if self.cfg.MODEL.MODALITY.USE_RADAR:
                radar_pointcloud = self.get_radar_data(rec,nsweeps=1,min_distance=2.2)
                data['radar_pointclouds'].append(radar_pointcloud)    
            if self.cfg.MODEL.LIDAR.USE_RANGE:
                lidar_range_cloud = self.get_lidar_range_data(rec,nsweeps=1,min_distance=2.2)
                data['range_clouds'].append(lidar_range_cloud)
            # if i == self.cfg.TIME_RECEPTIVE_FIELD-1:
            #     gt_trajectory, command = self.get_gt_trajectory(rec, index_t)
            #     data['gt_trajectory'] = torch.from_numpy(gt_trajectory).float()
            #     data['command'] = command
            #     trajs = self.get_trajectory_sampling(rec)
            #     data['sample_trajectory'] = torch.from_numpy(trajs).float()
        

        data['camera_timestamp'] = np.array(data['camera_timestamp'])
        data['camera_timestamp'] = (data['camera_timestamp'] - data['camera_timestamp'][0]) / 1e6
        
        if self.cfg.MODEL.LIDAR.USE_STPN or self.cfg.MODEL.LIDAR.USE_BESTI: 
            padded_voxel_points, lidar_voxel_times = self.get_temporal_voxels(index)
            data['lidar_timestamp']  = lidar_voxel_times
            data['padded_voxel_points'].append(padded_voxel_points)        
        for key, value in data.items():
            if key in ['image', 'intrinsics', 'extrinsics', 'segmentation', 'instance', 'future_egomotion']:
                data[key] = torch.cat(value, dim=0)

        if self.cfg.MODEL.LIDAR.USE_STPN or self.cfg.MODEL.LIDAR.USE_BESTI: 
            data['padded_voxel_points'] = torch.cat(data['padded_voxel_points'], dim=0)
        if self.cfg.MODEL.LIDAR.USE_RANGE:
            data['range_clouds'] = torch.cat(data['range_clouds'], dim=0)   
        if self.cfg.MODEL.MODALITY.USE_RADAR:
            data['radar_pointclouds'] = torch.cat(data['radar_pointclouds'], dim=0)       
        
        data['target_point'] = torch.tensor([0., 0.])
        instance_centerness, instance_offset, instance_flow = convert_instance_mask_to_center_and_offset_label(
            data['instance'], data['future_egomotion'],
            num_instances=len(instance_map), ignore_index=self.cfg.DATASET.IGNORE_INDEX, subtract_egomotion=True,
            spatial_extent=self.spatial_extent,
        )
        data['centerness'] = instance_centerness
        data['offset'] = instance_offset
        data['flow'] = instance_flow
        return data
    
    @staticmethod
    def get_sweep_idxs(current_info, sweep_count=[0, 0], current_idx=0):
        """
        TODO: Docstrings

        """

        assert type(sweep_count) is list and len(sweep_count) == 2, "Please give the upper and lower range of frames you want to process!"

        current_sample_idx = current_info["sample_idx"]
        current_seq_len = current_info["sequence_len"]

        target_sweep_list = np.array(list(range(sweep_count[0], sweep_count[1]+1)))
        target_sample_list = current_sample_idx + target_sweep_list
        # set the high and low thresh to extract multi frames in current sequence
        target_sample_list = [i if i >= 0 else 0 for i in target_sample_list]
        target_sample_list = [i if i < current_seq_len else current_seq_len-1 for i in target_sample_list]
        # get the index of target frames in the waymo info list
        target_idx_res = np.array(target_sample_list) - current_sample_idx
        target_idx_list = current_idx + target_idx_res

        return target_idx_list

    def get_events(self, current_idx, idx_list):
        
        events_list = []
        for i in idx_list:
            event_tmp = {}
            if i != current_idx: continue
            # pdb.set_trace()
            event_paths = self.infos[i]['event_paths']
            for i in range(len(event_paths)):
                event_path = event_paths[i]
                events = np.load(event_path, allow_pickle=True)
                event_tmp['evs_norm'] = torch.from_numpy(events['evs_norm'])
                event_tmp['ev_loc'] = torch.from_numpy(events['ev_loc'])
                events_list.append(event_tmp)
        return events_list
    
    def get_events_grid(self, current_idx, idx_list):
        if not self.use_event:
            return {}

        event_frames = []
        event_shape = None

        for i in idx_list:
            if i != current_idx:
                continue
            event_paths = self.infos[i].get('event_grid_paths', [])
            for event_path in event_paths:
                if not os.path.exists(event_path):
                    continue
                voxel = np.load(event_path)['voxel'].astype(np.float32)
                voxel = torch.from_numpy(voxel)
                h, w = voxel.shape[-2:]
                if self.event_scale != 1:
                    target_hw = (int(h * self.event_scale), int(w * self.event_scale))
                    voxel = F.interpolate(
                        voxel.unsqueeze(0), size=target_hw, mode='bilinear', align_corners=False
                    ).squeeze(0)
                    h, w = target_hw
                event_shape = (h, w)
                event_frames.append(voxel)
            break

        if not event_frames:
            return {'event': None, 'event_shape': event_shape}

        event_tensor = torch.cat(event_frames, dim=0)  # [C_evt, H, W]
        event_tensor = event_tensor.unsqueeze(0).unsqueeze(0).contiguous()
        return {
            'event': event_tensor.to(torch.float32),
            'event_shape': event_shape,
        }

    def _build_dummy_event(self):
        channels = getattr(self.cfg.MODEL.EVENT, 'IN_CHANNELS', 0)
        if channels <= 0:
            channels = 2 * getattr(self.cfg.MODEL.EVENT, 'BINS', 10)
        h, w = self.cfg.IMAGE.FINAL_DIM
        dummy = torch.zeros(1, 1, channels, h, w, dtype=torch.float32)
        return dummy
    
    def inverse_T(self, T):
        assert T.shape == (4, 4)
        R = T[:3, :3]
        R_inv = np.linalg.inv(R)
        t = T[:-1, -1]
        T_inv = np.eye(4)
        T_inv[:3, :3] = R_inv
        T_inv[:-1, -1] = -R_inv @ t
        return T_inv

    def get_images_and_params(self, current_idx, idx_list):
        imgs_dict = {
            'images': {},
            'extrinsic': {},
            'intrinsic': {},
            'image_shape': {}
        }

        for i in idx_list:
            if i != current_idx: continue
            
            img_infos = self.infos[i]['image']
            for key in img_infos.keys():
                if 'path' not in key or 'image' not in key: continue
                img_path = img_infos[key]
                for j in range(1):
                    img_path = img_path.replace(img_path.split('/')[-2], 'image_%d' % j)
                    img_path = os.path.join(self.dataroot, img_path)

                    img = cv2.imread(img_path)
                    img = resize_and_crop_image(
                        img, resize_dims=self.augmentation_parameters['resize_dims'], 
                        crop=self.augmentation_parameters['crop'])
                    # normalize images
                    img = img.astype(np.float32)
                    img /= 255.0
                    cam_name = 'camera_%s' % str(j)
                    
                    ### resize image
                    if cam_name not in imgs_dict['images']:
                        imgs_dict['images'][cam_name] = []
                    imgs_dict['images'][cam_name].append(img)
            top_crop = self.augmentation_parameters['crop'][1]
            left_crop = self.augmentation_parameters['crop'][0]
            img_infos['image_%d_intrinsic' % j] = torch.from_numpy(img_infos['image_%d_intrinsic' % j])
            img_infos['image_%d_intrinsic' % j] = update_intrinsics(
                img_infos['image_%d_intrinsic' % j], top_crop, left_crop,
                scale_width=self.augmentation_parameters['scale_width'],
                scale_height=self.augmentation_parameters['scale_height']
            )
            

        for j in range(1):
            cam_name = 'camera_%s' % str(j)
            # new_ex_param = np.linalg.inv(img_infos['image_%d_extrinsic' % j])
            new_ex_param = self.inverse_T(img_infos['image_%d_extrinsic' % j])
            imgs_dict['extrinsic'][cam_name] = torch.from_numpy(new_ex_param)
            imgs_dict['intrinsic'][cam_name] = img_infos['image_%d_intrinsic' % j]
            imgs_dict['image_shape'][cam_name] = img_infos['image_shape_%d' % j]
        return imgs_dict
    
    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.mode == 'train':
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)
            # data_dict_ = copy.deepcopy(data_dict)

        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            data_dict['gt_obj_ids'] = data_dict['gt_obj_ids'][selected]
            
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            
            gt_obj_ids = np.array([n for n in data_dict['gt_obj_ids']])
            
            
            
            obj_index_mat = common_utils.IndexDict()
            num_ids = []
            for id in gt_obj_ids:
                num_ids.append(obj_index_mat.get(id))
            gt_obj_ids = np.array(num_ids)
            
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_obj_ids.reshape(-1, 1), gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            
            
            data_dict['gt_boxes'] = gt_boxes
            

        gt_labels_3d = []
        for cat in data_dict['gt_names']:
            if cat in self.class_names:
                gt_labels_3d.append(self.class_names.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # haotian: this is an important change: from 0.5, 0.5, 0.5 -> 0.5, 0.5, 0
        gt_bboxes_3d = LiDARInstance3DBoxes(
            data_dict['gt_boxes'], box_dim=data_dict['gt_boxes'].shape[-1], origin=(0.5, 0.5, 0)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=data_dict['gt_names'],
        )

        gt_boxes_lidar_list = []
        gt_len = data_dict['gt_len']
        for i in range(len(gt_len)):
            if i ==0:
                gt_data = data_dict['gt_boxes'][:gt_len[0]] 
                gt_boxes_lidar_list.append(gt_data)
            else:
                gt_data = data_dict['gt_boxes'][gt_len[i-1]:gt_len[i]]
                gt_boxes_lidar_list.append(gt_data)
            # if gt_data.shape[0] == 0:
            #     pdb.set_trace()
        data_dict['gt_boxes_prosed'] = gt_boxes_lidar_list
        
        return data_dict

    

    def __getitem__(self, index):
        current_info = copy.deepcopy(self.infos[index])
        target_idx_list = self.get_sweep_idxs(current_info, current_idx=index)
        target_infos, points = self.get_infos_and_points(target_idx_list)
        points = points[0]
        input_dict = {
            'points': points,
            'frame_id': current_info['sample_idx'],
            'pose': current_info['pose'],
            'sequence_name': current_info['sequence_name'],
        }
        # breakpoint()
        if self.use_image:
            img_dict = self.get_images_and_params(index, target_idx_list)
            input_dict.update(img_dict)

        if self.use_event:
            ev_grid_dict = self.get_events_grid(index, target_idx_list)
            event_tensor = ev_grid_dict.get('event')
            if event_tensor is None:
                event_tensor = self._build_dummy_event()
            input_dict['event'] = event_tensor
            
        if 'seq_annos' in current_info:

            gt_boxes_lidar = []
            gt_len = []
            gt_len_index = 0
            anno_names = []
            gt_obj_ids = []

            for i in range(len(current_info['seq_annos'])):
                annos = current_info['seq_annos'][i]
                
                gt_boxes_lidar.append(annos['gt_boxes_lidar'])
                gt_len_index += len(annos['gt_boxes_lidar'])
                gt_len.append(gt_len_index)
                anno_names.append(annos['name'])
                gt_obj_ids.append(annos['obj_ids'])
                    
                    
            gt_boxes_lidar = np.concatenate(gt_boxes_lidar)
            gt_names = np.concatenate(anno_names)
            gt_len = np.array(gt_len)
            gt_obj_ids = np.concatenate(gt_obj_ids)
                
                

            input_dict.update({
                # 'gt_names': annos['name'],
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar,
                'gt_len': gt_len,
                'gt_obj_ids': gt_obj_ids
            })
        data_dict = self.prepare_data(data_dict=copy.deepcopy(input_dict))
        
        return data_dict


def list_oss_dir(oss_path, client, with_info=False):
    """
    Loading files from OSS
    """
    s3_dir = fix_path(oss_path)
    files_iter = client.get_file_iterator(s3_dir)
    if with_info:
        file_list = {p: k for p, k in files_iter}
    else:
        file_list = [p for p, k in files_iter]
    return file_list

def fix_path(path_str):
    try:
        st_ = str(path_str)
        if "s3://" in st_:
            return  st_
        if "s3:/" in st_:
            st_ = "s3://" + st_.strip('s3:/')
            return st_
        else:
            st_ = "s3://" + st_
            return st_
    except:
        raise TypeError

def oss_exist(data_path, file_path, oss_data_list, refresh=False):
    if data_path is None:
        raise IndexError("No initialized path set!")
    if refresh:
        oss_data_list = list_oss_dir(data_path, with_info=False)
    pure_name = fix_path(file_path).strip("s3://")
    if pure_name in oss_data_list:
        return True
    else:
        return False

if __name__ == '__main__':
    from fvcore.common.config import CfgNode as _CfgNode


    def convert_to_dict(cfg_node, key_list=[]):
        """Convert a config node to dictionary."""
        _VALID_TYPES = {tuple, list, str, int, float, bool}
        if not isinstance(cfg_node, _CfgNode):
            if type(cfg_node) not in _VALID_TYPES:
                print(
                    'Key {} with value {} is not a valid type; valid types: {}'.format(
                        '.'.join(key_list), type(cfg_node), _VALID_TYPES
                    ),
                )
            return cfg_node
        else:
            cfg_dict = dict(cfg_node)
            for k, v in cfg_dict.items():
                cfg_dict[k] = convert_to_dict(v, key_list + [k])
            return cfg_dict
    class CfgNode(_CfgNode):
        """Remove once https://github.com/rbgirshick/yacs/issues/19 is merged."""

        def convert_to_dict(self):
            return convert_to_dict(self)
    CN = CfgNode
    data_cfg = CN()
    data_cfg.DATASET = CN()
    data_cfg.DATASET.DATAROOT = '/media/switcher/sda/datasets/dsec/'
    data_cfg.DATASET.IGNORE_INDEX = 255  # Ignore index when creating flow/offset labels
    data_cfg.DATASET.FILTER_INVISIBLE_VEHICLES = True  # Filter vehicles that are not visible from the cameras
    data_cfg.DATASET.SAVE_DIR = 'datas'
    cfg = CN()
    cfg.TIME_RECEPTIVE_FIELD = 3  # how many frames of temporal context (1 for single timeframe)
    cfg.N_FUTURE_FRAMES = 4  # how many time steps into the future to predict
    cfg.VOXEL = CN()  # Lidar pointcloud voxelization
    cfg.VOXEL.VOXEL_SIZE = (0.8, 0.8, 0.4)
    cfg.VOXEL.AREA_EXTENTS = [[-51.2, 51.2], [-51.2, 51.2], [-3, 2]]

    cfg.LIFT = CN()  # image to BEV lifting
    cfg.LIFT.GT_DEPTH = True
    cfg.LIFT.GEN_DEPTH = False
    cfg.LIFT.DISCOUNT = 0.5
    cfg.LIFT.X_BOUND = [-51.2, 51.2, 0.8]  # Forward
    cfg.LIFT.Y_BOUND = [-51.2, 51.2, 0.8]  # Sides
    cfg.LIFT.Z_BOUND = [-10.0, 10.0, 20.0]  # Height
    cfg.LIFT.D_BOUND = [2.0, 50.0, 1.0]
    cfg.LIFT.RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    cfg.GEN = CN()
    cfg.GEN.GEN_DEPTH = True
    cfg.GEN.GEN_RANGE = True
    cfg.GEN.GEN_VOXELS = True

    cfg.IMAGE = CN()
    cfg.IMAGE.FINAL_DIM = (240, 360)
    cfg.IMAGE.RESIZE_SCALE = 0.5
    cfg.IMAGE.TOP_CROP = 0
    cfg.IMAGE.ORIGINAL_HEIGHT = 640  # Original input RGB camera height
    cfg.IMAGE.ORIGINAL_WIDTH = 480  # Original input RGB camera width
    cfg.IMAGE.NAMES = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    
    cfg.MODEL = CN()
    cfg.MODEL.MODALITY = CN()
    cfg.MODEL.MODALITY.USE_LIDAR = True
    cfg.MODEL.MODALITY.USE_CAMERA = True

    dsec = DatasetDSEC(data_cfg, cfg, is_train=True)
    dat = dsec[0]
    pass
