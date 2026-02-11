import os
import numpy as np
from tqdm import tqdm
import pickle as pkl

def save_pkl(data, pkl_path):
    with open(pkl_path, "wb") as f:
        pkl.dump(data, f)
    print(f"Saved {pkl_path}")
    
def load_pkl(pkl_path):
    """
    Load a pickle (.pkl) file.

    Args:
        pkl_path (str): path to the pkl file

    Returns:
        object: deserialized Python object
    """
    with open(pkl_path, "rb") as f:
        data = pkl.load(f)
    return data

def gen_flow_split(root_dir):
    # 初始化全局最值
    global_min = np.array([np.inf, np.inf, np.inf])
    global_max = np.array([-np.inf, -np.inf, -np.inf])

    total_frames = 0

    # 遍历第一级目录
    for name in tqdm(os.listdir(root_dir)):
        first_level_path = os.path.join(root_dir, name)

        # 判断是否为 zurich* 文件夹
        if not (os.path.isdir(first_level_path) and name.startswith("zurich")):
            continue


        # 遍历点云文件
        flow_fps = 2
        base_fps = 10
        for file in os.listdir(first_level_path):
            if not file.endswith("interpolate_fov_bbox_lidar_check.pkl"):
                continue

            file_path = os.path.join(first_level_path, file)
            try:
                dat = load_pkl(file_path)
                new_dat = []
                for line in dat:
                    flow_list = []
                    for i in range(flow_fps):
                        flow_index = base_fps//flow_fps*(i+1)-1
                        flow_list.append(flow_index)
                    line['flow_list'] = flow_list
                    new_dat.append(line)
            except Exception as e:
                print(f"[Error] Failed to load {file_path}: {e}")
                continue
            file_path = file_path.replace("interpolate_fov_bbox_lidar_check.pkl", "interpolate_fov_bbox_lidar_check_flow.pkl")
            save_pkl(new_dat, file_path)

if __name__ == '__main__':
    root_dir = "/media/switcher/sda/datasets/dsec/"

    gen_flow_split(root_dir)
