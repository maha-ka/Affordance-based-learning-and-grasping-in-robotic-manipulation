import sapien as sapien
import numpy as np
from PIL import Image

scene = sapien.Scene()
scene.set_timestep(1 / 100.0)
scene.add_ground(0)

scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
scene.add_point_light([1, 2, 2], [1, 1, 1])

viewer = scene.create_viewer()
viewer.set_camera_xyz(x=0, y=0, z=3)
viewer.set_camera_rpy(r=0, p=-1.8, y=0)


# table top
table_height = 0.3
builder = scene.create_actor_builder()
builder.add_box_collision(half_size=[0.4, 0.4, 0.025])
builder.add_box_visual(half_size=[0.4, 0.4, 0.025])
table = builder.build_kinematic(name='table')
table.set_pose(sapien.Pose([0, 0, table_height - 0.025]))

def manipulator(fix_root_link, balance_passive_force):
    
    # Load URDF
    loader = scene.create_urdf_loader()
    loader.fix_root_link = fix_root_link
    robot = loader.load("assets/panda/panda.urdf")
    robot.set_root_pose(sapien.Pose([-0.5, 0, table_height], [1, 0, 0, 0]))

    # Set initial joint positions
    arm_init_qpos = [0, 0, 0, 0, 0, 0, 0]
    gripper_init_qpos = [0, 0]
    
    init_qpos = arm_init_qpos + gripper_init_qpos
    robot.set_qpos(init_qpos)

    while not viewer.closed:
        for _ in range(4):
            if balance_passive_force:
                qf = robot.compute_passive_force(
                    gravity=True,
                    coriolis_and_centrifugal=True,
                )
                robot.set_qf(qf)
            scene.step()
        scene.update_render()
        viewer.render()


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fix-root-link", action="store_true")
parser.add_argument("--balance-passive-force", action="store_true")
args = parser.parse_args()
manipulator(
    fix_root_link=True,
    balance_passive_force=True,
)


near, far = 0.1, 2.0
width, height = 1024, 1024

cam_pos = np.array([0, 0, 1.0])  # 1m above table

# Camera axes
forward = np.array([0, 0, -1])   # look down
left    = np.array([-1, 0, 0])   # image x-axis
up      = np.array([0, 1, 0])    # image y-axis

mat44 = np.eye(4)
mat44[:3, :3] = np.stack([forward, left, up], axis=1)
mat44[:3, 3] = cam_pos

camera = scene.add_camera(
    name="topdown_camera",
    width=width,
    height=height,
    fovy=np.deg2rad(35),
    near=near,
    far=far,
)

camera.entity.set_pose(sapien.Pose(mat44))


# Load ALL URDF models
import os

ASSETS_ROOT = "assets/urdf_models/models"

# output folders
RGB_DIR = "code_results/rgb"
DEPTH_DIR = "code_results/depth"

os.makedirs(RGB_DIR, exist_ok=True)
os.makedirs(DEPTH_DIR, exist_ok=True)

# get all object folders
object_folders = sorted([
    f for f in os.listdir(ASSETS_ROOT)
    if os.path.isdir(os.path.join(ASSETS_ROOT, f))
    and not f.startswith("__")
])

# LOOP THROUGH ALL OBJECTS
for obj_name in object_folders:

    print(f"Processing: {obj_name}")

    obj_dir = os.path.join(ASSETS_ROOT, obj_name)

    visual_file = None
    collision_file = None

    visual_candidates = [
        "textured.obj",
        "visual.obj",
        "visual_mesh.obj",
        "model.obj",
        "mesh.obj",
    ]

    collision_candidates = [
        "collision.obj",
        "collision_mesh.obj",
        "visual_mesh.obj",
        "model.obj",
    ]

    for root, _, files in os.walk(obj_dir):

        for f in files:

            if visual_file is None and f in visual_candidates:
                visual_file = os.path.join(root, f)

            if collision_file is None and f in collision_candidates:
                collision_file = os.path.join(root, f)

    if visual_file is None:
        for root, _, files in os.walk(obj_dir):
            for f in files:
                if f.endswith(".obj"):
                    visual_file = os.path.join(root, f)
                    break

    if collision_file is None:
        collision_file = visual_file

    # skip invalid objects
    if visual_file is None:
        print(f"Skipping {obj_name} (no OBJ found)")
        continue

    # Build object
    builder = scene.create_actor_builder()

    try:
        builder.add_convex_collision_from_file(
            filename=collision_file
        )

        builder.add_visual_from_file(
            filename=visual_file
        )

        mesh = builder.build(name=obj_name)

        mesh.set_pose(
            sapien.Pose(
                p=[0.1, 0, table_height + 0.02]
            )
        )

    except Exception as e:
        print(f"Failed loading {obj_name}: {e}")
        continue

    scene.step()
    scene.update_render()
    camera.take_picture()
    
    rgba = camera.get_picture("Color")              # [H, W, 4]
    position = camera.get_picture("Position")       # [H, W, 4]

    valid = position[..., 3] < 1.0

    #  metric depth: (Jacquard-style)
    depth = np.linalg.norm(position[..., :3], axis=-1)  # meters

    # replace background with table depth (Jacquard behavior)
    # use median of valid depths (stable approximation of table plane)
    table_depth = np.median(depth[valid])
    depth[~valid] = table_depth

    # clip depth range (removes background dominance)
    dmin = np.percentile(depth[valid], 1)
    dmax = np.percentile(depth[valid], 99)
    depth = np.clip(depth, dmin, dmax)

    # save depth as Jacquard-style tiff
    #  float32 depth in meters (closest to Jacquard "perfect depth")
    depth_f32 = depth.astype(np.float32)

    depth_path = os.path.join(
        DEPTH_DIR,
        f"{obj_name}_perfect_depth.tiff"
    )

    Image.fromarray(depth_f32).save(depth_path)

    # Save RGB for 
    rgba_img = (rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)
    rgb_path = os.path.join(
        RGB_DIR,
        f"{obj_name}_RGB.png"
    )
    Image.fromarray(rgba_img).save(rgb_path)

    print(f"Saved: {obj_name}")

    # Remove object before next iteration
    scene.remove_actor(mesh)

print("DONE")