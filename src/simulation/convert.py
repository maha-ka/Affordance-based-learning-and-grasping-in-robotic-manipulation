import numpy as np
import sapien
from scipy.spatial.transform import Rotation as R

def compute_6dof_grasp_pose(
    position_world,
    theta,
    pregrasp_offset=0.08
):
    """
    Convert (x, y, z, theta) grasp into a full 6-DoF SAPIEN pose.

    """

    z_gripper = np.array([0.0, 0.0, -1.0])  # approach direction

    # Finger closing direction from rectangle angle
    x_gripper = np.array([
        np.cos(theta),
        -np.sin(theta),
        0.0
    ])

    x_gripper /= np.linalg.norm(x_gripper)

    # Compute y using right-hand rule
    y_gripper = np.cross(z_gripper, x_gripper)
    y_gripper /= np.linalg.norm(y_gripper)

    # re orthogonalize x
    x_gripper = np.cross(y_gripper, z_gripper)
    x_gripper /= np.linalg.norm(x_gripper)

    # rotation matrix
    R_grasp = np.stack(
        [x_gripper, y_gripper, z_gripper],
        axis=1
    )

    # convert rotation matrix to quaternion (wxyz)
    quat = R.from_matrix(R_grasp).as_quat(scalar_first=True) 

    position_pregrasp = position_world - pregrasp_offset * z_gripper
    
    #  create SAPIEN pose
    grasp_pose = sapien.Pose(
        position_pregrasp,
        quat
    )

    return grasp_pose
