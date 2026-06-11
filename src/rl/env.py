import numpy as np
import sapien.core as sapien
from typing import Any
import torch
import numpy as np
import random
import sapien
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from objects import OBJECTS


@register_env("PandaReachPose-v1", max_episode_steps=200)
class PandaReachPoseEnv(BaseEnv):
    """
    Panda reaches a target 6-DoF pose (position + orientation).
    uses ManiSkill PD controllers.
    """

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        render_mode= None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise

        super().__init__(*args, robot_uids=robot_uids,render_mode=render_mode, **kwargs)
        self.prev_action = np.zeros(7, dtype=np.float32)

        
    # Cameras
    @property
    def _default_sensor_configs(self):
        return []

    def _default_viewer_camera_config(self):
        return dict(
            position=[0.9, 0.7, 0.8],   
            look_at=[0.0, 0.0, 0.3],    
        )

    # Load robot
    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        #  OBJECT 
        self.objects = {}

        for name, cfg in OBJECTS.items():

            builder = self.scene.create_actor_builder()

            builder.add_convex_collision_from_file(cfg["collision"])
            builder.add_visual_from_file(cfg["visual"])

            builder.initial_pose = sapien.Pose(
                p=[-0.015, 0, 0],
                q=[1, 0, 0, 0]
            )

            obj = builder.build(name=name)

            self.objects[name] = obj



    def render_human(self):
        viewer = super().render_human()

        if not hasattr(self, "_camera_set"):
            viewer.set_camera_xyz(0.9, 0.7, 0.8)
            viewer.set_camera_rpy(r=0, p=0, y=129)
            self._camera_set = True

        return viewer



    # Reset
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.prev_action = np.zeros(7, dtype=np.float32)
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)

            self.current_object_name = random.choice(list(OBJECTS.keys()))

            cfg = OBJECTS[self.current_object_name]

            self.current_object = self.objects[self.current_object_name]

            self.target_pose = Pose.create_from_pq(
                cfg["grasp_pose"].p,
                cfg["grasp_pose"].q
            )

            for name, obj in self.objects.items():

                if name == self.current_object_name:

                    obj.set_pose(
                        sapien.Pose(
                            p=[-0.015, 0, 0],
                            q=[1, 0, 0, 0]
                        )
                    )

                else:
                    # move away
                    obj.set_pose(
                        sapien.Pose(
                            p=[10, 10, 10]
                        )
                    )


    # Observations
    def _get_obs_extra(self, info: dict):
        tcp_pose = self.agent.tcp_pose

        obs = dict(
            tcp_pose=tcp_pose.raw_pose,
            target_pose=self.target_pose.raw_pose,
        )

        if "state" in self.obs_mode:
            obs.update(
                tcp_to_target_pos=(
                    self.target_pose.p - tcp_pose.p
                ),
            )

        obs["object_id"] = torch.tensor(
            [OBJECTS[self.current_object_name]["id"]],
            dtype=torch.float32,
            device=self.device
        )

        obs["object_pose"] = self.current_object.pose.raw_pose

        obs["prev_action"] = torch.tensor(
            self.prev_action,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        return obs


    def evaluate(self):
        grasp = self.agent.is_grasping(self.current_object)
        success = grasp

        return {
            "success": success
        }


    def compute_dense_reward(self, obs, action, info):
        tcp_pose = self.agent.tcp_pose
        self.prev_action = (
            action.detach().cpu().numpy()[0]
        )
        #pos
        pos_dist = torch.linalg.norm(
            tcp_pose.p - self.target_pose.p, axis=1
        )
        pos_reward = 1 - torch.tanh(pos_dist / 0.1)

        #  Orientation 
        dot = torch.sum(tcp_pose.q * self.target_pose.q, dim=1).abs()
        dot = torch.clamp(dot, -1.0, 1.0)
        ori_dist = 2 * torch.arccos(dot) / torch.pi
        # Grasp
        grasp = self.agent.is_grasping(self.current_object)
        grasp_reward = grasp.float() * 0.5

        #  Smoothness 
        action_penalty = 0.0001 * torch.sum(action**2, dim=1)

        reward = -0.2 * pos_dist + 0.1 * pos_reward - 0.15 * ori_dist - action_penalty + grasp_reward 

        return reward.detach().cpu().numpy()


    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs, action, info) / 5.0