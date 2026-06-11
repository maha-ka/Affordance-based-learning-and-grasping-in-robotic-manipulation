import sapien as sapien
import numpy as np

table_height = 0.3

scene = sapien.Scene()
scene.set_timestep(1 / 100.0)
scene.add_ground(0)

scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
scene.add_point_light([1, 2, 2], [1, 1, 1])

viewer = scene.create_viewer()
viewer.set_camera_xyz(0.9, 0.7, 0.8)
viewer.set_camera_rpy(r=0, p=0, y=129)

# table top
builder = scene.create_actor_builder()
builder.add_box_collision(half_size=[1.7, 0.7, 0.025])
builder.add_box_visual(half_size=[1.7, 0.7, 0.025] , material=[0.5, 0.2, 0.1])
table = builder.build_kinematic(name='table')
table.set_pose(sapien.Pose([0, 0, table_height - 0.025]))

builder = scene.create_actor_builder()
builder.add_convex_collision_from_file(
    filename="assets/urdf_models/models/spoon/collision.obj"
)
builder.add_visual_from_file(filename="assets/urdf_models/models/spoon/textured.obj")
mesh = builder.build(name="mesh")
mesh.set_pose(sapien.Pose(p=[0.1, 0, table_height + 0.02]))


grasp_pose0 = sapien.Pose([0 , 0, 0.32], [ 1,0, 0, 0])

grasp_pose = sapien.Pose([0.0709113, -0.0105613, 0.41657], [-0, 0.713862, 0.700287, 0]) 

def create_axis(scene, length=0.1, thickness=0.003):
    axes = {}
    for name, half, color in [
        ("x", [length/2, thickness/2, thickness/2], [1,0,0]),
        ("y", [thickness/2, length/2, thickness/2], [0,1,0]),
        ("z", [thickness/2, thickness/2, length/2], [0,0,1]),
    ]:
        b = scene.create_actor_builder()
        b.add_box_visual(half_size=half, material=color)
        axes[name] = b.build_kinematic()
    return axes

ee_axes = create_axis(scene, length=0.12)
x_offset = sapien.Pose([0.06, 0, 0])
y_offset = sapien.Pose([0, 0.06, 0])
z_offset = sapien.Pose([0, 0, 0.06])

target_axes = create_axis(scene, length=0.12)
target_axes0 = create_axis(scene, length=0.12)
target_axes0["x"].set_pose(grasp_pose0 * x_offset)
target_axes0["y"].set_pose(grasp_pose0 * y_offset)
target_axes0["z"].set_pose(grasp_pose0 * z_offset)
             
target_axes2 = create_axis(scene, length=0.12)

target_axes["x"].set_pose(grasp_pose * x_offset)
target_axes["y"].set_pose(grasp_pose * y_offset)
target_axes["z"].set_pose(grasp_pose * z_offset)

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fix-root-link", action="store_true")
    parser.add_argument("--balance-passive-force", action="store_true")
    args = parser.parse_args()

    fix_root_link=True
    balance_passive_force=True

    # Load URDF
    loader = scene.create_urdf_loader()
    loader.fix_root_link = fix_root_link
    robot = loader.load("assets/panda/panda.urdf")
    robot.set_root_pose(sapien.Pose([-0.5, 0, table_height], [1, 0, 0, 0]))

    # Get end-effector link 
    ee_link = None
    for link in robot.get_links():
        if link.get_name() == "panda_hand":  
            ee_link = link
            break

    if ee_link is None:
        raise RuntimeError("End-effector link 'panda_hand' not found!")


    # Set initial joint positions     
    arm_init_qpos = [0, 0.5, 0, -1.2, 0, 1.5, 0]
    gripper_init_qpos = [0.04, 0.04]
    


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
        tcp_pose = ee_link.get_pose()
        ee_axes["x"].set_pose(tcp_pose * x_offset)
        ee_axes["y"].set_pose(tcp_pose * y_offset)
        ee_axes["z"].set_pose(tcp_pose * z_offset)
        scene.update_render()
        viewer.render()

    near, far = 0.1, 2.0
    width, height = 1024, 1024

    cam_pos = np.array([0, 0, 1.0])  

    # Camera axes
    forward = np.array([0, 0, -1])  
    left    = np.array([-1, 0, 0])  
    up      = np.array([0, 1, 0])   

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

    print("mat44\n " , mat44)
    camera.entity.set_pose(sapien.Pose(mat44))
    print("Intrinsic matrix\n", camera.get_intrinsic_matrix())

    scene.step() 
    scene.update_render()  
    camera.take_picture()  

    # Get camera outputs

    rgba = camera.get_picture("Color")              # [H, W, 4]
    position = camera.get_picture("Position")       # [H, W, 4]

    # Valid pixels (ray hit something)
    valid = position[..., 3] < 1.0

    # TRUE metric depth: ray distance (Jacquard-style)
    depth = np.linalg.norm(position[..., :3], axis=-1)  # meters

    x_img = int(652.0)
    y_img = int( 600.0)
    pc = position[y_img, x_img, :3]  # (Xc, Yc, Zc)
    print("pc",pc)


if __name__ == "__main__":
    main()
