import torch
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from models.Unet import UNet

IMG_SIZE = 256 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

image_folder = "testdata/images/"
depth_folder = "testdata/depth/"
grasp_folder = "testdata/grasp/"

for fname in os.listdir(image_folder):
    if not fname.endswith(".png"): continue

    rgb_path = os.path.join(image_folder, fname)
    depth_path = os.path.join(depth_folder, fname.replace("_RGB.png", "_perfect_depth.tiff"))

    rgb_orig = cv2.imread(rgb_path)
    if rgb_orig is None:
        raise ValueError(f"Invalid RGB image: {rgb_path}")
    orig_h, orig_w = rgb_orig.shape[:2]


    rgb_resized = cv2.resize(rgb_orig, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Invalid depth image: {depth_path}")
    depth_resized = cv2.resize(depth, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    depth_resized = (depth_resized / 1000.0)[..., None]

    rgbd = np.concatenate([rgb_resized, depth_resized], axis=2)
    inp = torch.tensor(rgbd).permute(2,0,1).unsqueeze(0).float().to(device)

    model = UNet(in_channels=4, out_channels=5).to(device)
    model.load_state_dict(torch.load("model_unet_epoch50.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        out = model(inp)

    out_np = out[0].cpu().numpy()
    heat_logits, sin_map, cos_map, w_map, h_map = out_np
    heat = 1 / (1 + np.exp(-heat_logits))

    y, x = np.unravel_index(np.argmax(heat), heat.shape)
    score = heat[ y, x]

    theta = math.atan2(sin_map[y, x], cos_map[y, x])
    angle_deg = math.degrees(theta)
    width = w_map[y, x] * IMG_SIZE
    height = h_map[y, x] * IMG_SIZE

    scale_x = orig_w / IMG_SIZE
    scale_y = orig_h / IMG_SIZE

    x_orig = x * scale_x
    y_orig = y * scale_y
    width_orig = width * scale_x
    height_orig = height * scale_y

    center = (float(x_orig) , float(y_orig))
    size = (float(width_orig), float(height_orig))
    rotated_rect = (center, size, float(angle_deg))
    box_pts = cv2.boxPoints(rotated_rect).astype(np.int32)

    overlay = rgb_orig.copy()
    cv2.drawContours(overlay, [box_pts], 0, (0,255,0), 2)  
    cv2.circle(overlay, (int(x_orig), int(y_orig)), 4, (0,255,0), -1)  

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # plot

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1); plt.title("Input"); plt.imshow(cv2.cvtColor(rgb_orig, cv2.COLOR_BGR2RGB)); plt.axis('off')
    plt.subplot(1,2,2); plt.title("Grasp Rectangle"); plt.imshow(overlay_rgb); plt.axis('off')
    plt.show()

    print("Grasp Prediction:")
    print(f" Center (x,y): ({x_orig:.1f}, {y_orig:.1f})")
    print(f" Angle (deg): {angle_deg:.2f}")
    print(f" Width (px): {width_orig:.1f}")
    print(f" Height (px): {height_orig:.1f}")
