import os
import glob
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from utils import parse_grasp_file , build_targets_from_grasps

IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class JacquardDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        folders = [os.path.join(root_dir, f) for f in os.listdir(
            root_dir) if os.path.isdir(os.path.join(root_dir, f))]

        for folder in folders:
            files = glob.glob(os.path.join(folder, '*_RGB.*'))
            for rgb_path in files:
                prefix = rgb_path.split('_RGB')[0]
                depth_path = prefix + '_perfect_depth.tiff'
                grasp_path = prefix + '_grasps.txt'

                if os.path.exists(depth_path) and os.path.exists(grasp_path):
                    self.samples.append((rgb_path, depth_path, grasp_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, depth_path, grasp_path = self.samples[idx]

        # Load RGB
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            raise ValueError(f"Invalid RGB image: {rgb_path}")
        orig_h, orig_w = rgb.shape[:2]
        rgb_resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

        # Load depth
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Invalid depth image: {depth_path}")
        depth_resized = cv2.resize(depth, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
        depth_resized = (depth_resized / 1000.0)[..., None]  

        # Parse grasps
        grasps = parse_grasp_file(grasp_path)  # list of (x,y,theta_deg,w_px,h_px)

        scale_x = IMG_SIZE / float(orig_w)
        scale_y = IMG_SIZE / float(orig_h)

        grasps_scaled = []
        for (x, y, theta_deg, w_px, h_px) in grasps:
            x_s = x * scale_x
            y_s = y * scale_y
            w_s = w_px * (IMG_SIZE / float(max(orig_h, orig_w)))
            h_s = h_px * (IMG_SIZE / float(max(orig_h, orig_w)))
            grasps_scaled.append((x_s, y_s, theta_deg, w_s, h_s))

        target = build_targets_from_grasps(grasps_scaled, (IMG_SIZE, IMG_SIZE), sigma=5)

        rgbd = np.concatenate([rgb_resized, depth_resized], axis=2)  # (H, W, 4)

        inp = torch.tensor(rgbd).permute(2, 0, 1).float()
        targ = torch.tensor(target).float()

        return inp, targ


