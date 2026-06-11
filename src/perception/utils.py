import numpy as np
import cv2


def parse_grasp_file(path):
    """
    Parse jacquard grasp file lines of format:
      x; y; theta_deg; width_px; height_px
    Returns a list: [(x,y,theta_deg,w,h), ...]
    """
    grasps = []
    with open(path, 'r') as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(';') if p.strip() != '']
            if len(parts) >= 5:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    theta = float(parts[2])  # degrees
                    w = float(parts[3])
                    h = float(parts[4])
                    grasps.append((x, y, theta, w, h))
                except ValueError:
                    continue
    return grasps



def build_targets_from_grasps(grasps, img_size, sigma=5):
    """
    Output channels:
      0: Q (grasp heatmap)
      1: sin(2θ)
      2: cos(2θ)
      3: width 
      4: height 
    """
    H, W = img_size

    Q = np.zeros((H, W), dtype=np.float32)
    sin2 = np.zeros((H, W), dtype=np.float32)
    cos2 = np.zeros((H, W), dtype=np.float32)
    W_map = np.zeros((H, W), dtype=np.float32)
    H_map = np.zeros((H, W), dtype=np.float32)

    for (x, y, theta_deg, w_px, h_px) in grasps:
        x_i, y_i = int(round(x)), int(round(y))
        if not (0 <= x_i < W and 0 <= y_i < H):
            continue

        tmp = np.zeros((H, W), dtype=np.float32)
        tmp[y_i, x_i] = 1.0
        tmp = cv2.GaussianBlur(tmp, (0, 0), sigma)

        Q = np.maximum(Q, tmp)

        theta = np.deg2rad(theta_deg)
        weight = tmp

        sin2 += weight * np.sin(2 * theta)
        cos2 += weight * np.cos(2 * theta)
        W_map += weight * (w_px / W)
        H_map += weight * (h_px / H)

    if Q.max() > 0:
        Q /= Q.max()

    eps = 1e-6
    sin2 /= (Q + eps)
    cos2 /= (Q + eps)
    W_map /= (Q + eps)
    H_map /= (Q + eps)

    target = np.stack([Q, sin2, cos2, W_map, H_map], axis=0)
    return target

