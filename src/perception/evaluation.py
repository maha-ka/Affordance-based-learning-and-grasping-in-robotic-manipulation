import torch
import cv2
import numpy as np
import math

IMG_SIZE = 256

def angle_error(rad_pred, rad_true):
    diff = abs(rad_pred - rad_true)
    diff = min(diff, abs(diff - math.pi))
    return abs(math.degrees(diff))

def infer_grasp_from_target(output_tensor): 

    if isinstance(output_tensor, torch.Tensor):
        out = output_tensor.detach().cpu().numpy()

    else:
        out = output_tensor
    heat_logits = out[0]
    sin_map = out[1]
    cos_map = out[2]
    w_map = out[3]
    h_map = out[4]

    heat = 1.0 / (1.0 + np.exp(-heat_logits))  
    idx = np.unravel_index(np.argmax(heat), heat.shape)
    y, x = int(idx[0]), int(idx[1])

    sin_v = float(sin_map[y, x])
    cos_v = float(cos_map[y, x])
    theta = 0.5 * math.atan2(sin_v, cos_v)

    max_dim = float(max(IMG_SIZE, IMG_SIZE))
    width = float(w_map[y, x]) * max_dim
    height = float(h_map[y, x]) * max_dim

    # OpenCV box: 
    angle_deg = -math.degrees(theta)  
    box = ((x, y), (width, height), angle_deg)
    box_pts = cv2.boxPoints(box).astype(np.int32)

    return {
        'center': (x, y),
        'theta': theta,
        'width': width,
        'height': height,
        'box_pts': box_pts,
        'heatmap': heat
    }


def evaluate_accuracy_simple(model, loader, device,
                             heat_thresh=0.1,
                             angle_thresh_deg=40):
    model.eval()

    total = 0
    positive = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images).cpu()
            targets = targets.cpu()

            for b in range(outputs.shape[0]):
                out_b = outputs[b]
                tgt_b = targets[b]

                pred = infer_grasp_from_target(out_b)
                x_pred, y_pred = pred['center']
                theta_pred = pred['theta']

                gt = infer_grasp_from_target(tgt_b)
                theta_gt = gt['theta']

                heat_gt = tgt_b[0].numpy()          
                if not (0 <= x_pred < IMG_SIZE and 0 <= y_pred < IMG_SIZE):
                    total += 1
                    continue

                heat_val = heat_gt[y_pred, x_pred]

                inside_heat = heat_val >= heat_thresh

                ang_err = angle_error(theta_pred, theta_gt)
                ok_angle = ang_err <= angle_thresh_deg

                is_positive = inside_heat and ok_angle

                total += 1
                if is_positive:
                    positive += 1

    acc = positive / total
    print(f"Positive {positive}/{total}")
    print(f"Accuracy {acc*100:.2f}%")
    return acc


class PerturbedValDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, perturb_fn=None):
        self.base = base_dataset
        self.perturb_fn = perturb_fn

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        inp, target = self.base[idx]     

        rgb = (inp[:3].permute(1,2,0).numpy() * 255).astype(np.uint8)

        if self.perturb_fn is not None:
            rgb = self.perturb_fn(rgb)

        rgb = rgb.astype(np.float32) / 255.0
        rgb = torch.tensor(rgb).permute(2,0,1)

        depth = inp[3:4]             

        new_inp = torch.cat([rgb, depth], dim=0)

        return new_inp, target


def add_gaussian_noise(rgb, std=10):
    noise = np.random.normal(0, std, rgb.shape).astype(np.float32)
    noisy = rgb.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_gaussian_blur(rgb, ksize=7):
    return cv2.GaussianBlur(rgb, (ksize, ksize), 0)

def change_brightness(rgb, factor=1.3):
    rgb = rgb.astype(np.float32) * factor
    return np.clip(rgb, 0, 255).astype(np.uint8)






