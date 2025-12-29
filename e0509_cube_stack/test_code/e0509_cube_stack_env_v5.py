# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Doosan E0509 Cube Stack 강화학습 환경 v5
- 6축 로봇 + RH-P12-RN 그리퍼
- 그리퍼 1D 제어 (실제 로봇과 동일)
- 4개 관절 동기화 제어
- 그리퍼 열고 닫기: 규칙 기반으로 확실하게 동작
- 단계별 보상 시스템 (Staged Reward)
"""

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
import os

##############################################################################
# E0509 로봇 설정
##############################################################################

E0509_USD_PATH = os.path.expanduser("~/IsaacLab/source/isaaclab_assets/data/Robots/Doosan/E0509_rl_basic.usd")

E0509_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=E0509_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            # E0509 팔 관절 (6개) - 그리퍼가 아래를 향하는 자세
            "joint_1": 0.0,
            "joint_2": 0.0,
            "joint_3": 1.5708,  # 90도
            "joint_4": 0.0,
            "joint_5": 1.5708,  # 90도
            "joint_6": 0.0,
            # RH-P12-RN 그리퍼 관절 (4개) - 모두 열림 상태
            "rh_l1": 0.0,
            "rh_l2": 0.0,
            "rh_r1": 0.0,
            "rh_r2": 0.0,
        },
    ),
    actuators={
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-3]"],
            velocity_limit=87.0,
            effort_limit=87.0,
            stiffness=50.0,
            damping=4.0,
        ),
        "forearm": ImplicitActuatorCfg(
            joint_names_expr=["joint_[4-6]"],
            velocity_limit=12.0,
            effort_limit=87.0,
            stiffness=10.0,
            damping=1.0,
        ),
        # 그리퍼: 강한 stiffness로 확실하게 동작
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["rh_l1", "rh_r1"],
            velocity_limit=2.0,
            effort_limit=1000.0,
            stiffness=10000.0,
            damping=500.0,
        ),
    },
)


@configclass
class E0509CubeStackEnvCfg(DirectRLEnvCfg):
    """E0509 Cube Stack 환경 설정 v5"""

    # 환경 설정
    episode_length_s = 12.0
    decimation = 2
    action_scale = 1.5
    # 액션 공간: 6 arm + 1 gripper = 7 (유지)
    action_space = 7
    # 관측 공간: 22D (유지)
    observation_space = 22
    state_space = 0

    # 시뮬레이션 설정
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=3.0,
        replicate_physics=True,
    )

    # 테이블 설정
    table_height: float = 1.0
    table_thickness: float = 0.05
    table_stand_height: float = 0.03

    # 로봇 설정
    robot_cfg: ArticulationCfg = E0509_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.425, 0.0, 1.055),
            joint_pos={
                "joint_1": 0.0,
                "joint_2": 0.0,
                "joint_3": 1.5708,
                "joint_4": 0.0,
                "joint_5": 1.5708,
                "joint_6": 0.0,
                "rh_l1": 0.0,
                "rh_l2": 0.0,
                "rh_r1": 0.0,
                "rh_r2": 0.0,
            },
        ),
    )

    # 테이블
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.6, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.5, 0.4)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )

    # 테이블 스탠드
    table_stand_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/TableStandA",
        spawn=sim_utils.CuboidCfg(
            size=(0.18, 0.22, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.425, 0.0, 1.04)),
    )

    # 큐브 A (빨간색 - 집을 큐브) - 8.5cm x 12cm x 3.5cm
    cubeA_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/CubeA",
        spawn=sim_utils.CuboidCfg(
            size=(0.085, 0.12, 0.035),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0425)),
    )

    # 큐브 B (팔레트 - 목표 위치)
    cubeB_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/CubeB",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.6, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.15, 0.0, 1.05)),
    )

    # 큐브 크기
    cubeA_size: float = 0.035  # 높이 기준
    cubeA_width: float = 0.085  # 그리퍼가 잡을 방향 폭
    cubeB_size: float = 0.05

    # 노이즈
    start_position_noise: float = 0.15
    start_rotation_noise: float = 0.5


class E0509CubeStackEnv(DirectRLEnv):
    """
    E0509 로봇 Cube Stack 환경 v5
    
    주요 변경사항 (v4 대비):
    1. 액션 공간 7D 유지 (팔 6개 + 그리퍼 1개)
    2. 그리퍼 열고 닫기만 규칙 기반으로 확실하게 동작
    3. RL 액션은 그리퍼 방향/위치 제어에 여전히 사용
    4. 그리퍼 stiffness 강화
    """

    cfg: E0509CubeStackEnvCfg

    def __init__(self, cfg: E0509CubeStackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 관절 제한
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(self.device)
        
        # 타겟 위치 (10개 관절 모두)
        self.robot_dof_targets = torch.zeros((self.num_envs, 10), device=self.device)
        
        # 링크 인덱스
        self.hand_link_idx = self._robot.find_bodies("tool0")[0][0]
        self.finger_l_idx = self._robot.find_bodies("rh_p12_rn_l2")[0][0]
        self.finger_r_idx = self._robot.find_bodies("rh_p12_rn_r2")[0][0]
        
        # 환경 설정
        self.table_height = self.cfg.table_height + self.cfg.table_thickness / 2.0
        self.cubeA_size = self.cfg.cubeA_size
        self.cubeA_width = self.cfg.cubeA_width
        self.cubeB_size = self.cfg.cubeB_size

        # 행동 저장 (7D: 6 arm + 1 gripper)
        self.actions = torch.zeros((self.num_envs, 7), device=self.device)
        
        # 상태 추적
        self.cubeA_initial_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.cube_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cube_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 그리퍼 제어 상태
        self.should_close_gripper = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 그리퍼 열림 정도 계산용 (l1 + r1 기준)
        self.max_gripper_open = self.robot_dof_upper_limits[6] + self.robot_dof_upper_limits[7]
        
        # 디버그 출력
        print(f"=== E0509 Cube Stack Env v5 ===")
        print(f"Action space: {self.cfg.action_space} (6 arm + 1 gripper)")
        print(f"Gripper: Rule-based open/close, RL controls arm positioning")


    def _setup_scene(self):
        """씬 구성"""
        self._robot = Articulation(self.cfg.robot_cfg)
        self._table = RigidObject(self.cfg.table_cfg)
        self._table_stand = RigidObject(self.cfg.table_stand_cfg)
        self._cubeA = RigidObject(self.cfg.cubeA_cfg)
        self._cubeB = RigidObject(self.cfg.cubeB_cfg)

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["table"] = self._table
        self.scene.rigid_objects["table_stand"] = self._table_stand
        self.scene.rigid_objects["cubeA"] = self._cubeA
        self.scene.rigid_objects["cubeB"] = self._cubeB

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        clone = self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        """
        행동 적용
        
        actions: [batch, 7] - 6개 팔 관절 + 1개 그리퍼
        
        팔 관절 (0-5): RL 액션 그대로 적용
        그리퍼 (6): RL 액션은 무시하고, 규칙 기반으로 확실하게 열기/닫기
        """
        self.actions = actions.clone().clamp(-1.0, 1.0)
        
        # 팔 관절 (0-5): RL 액션 적용
        arm_actions = self.actions[:, :6]
        arm_targets = self.robot_dof_targets[:, :6] + self.cfg.action_scale * self.cfg.sim.dt * arm_actions
        arm_targets = torch.clamp(arm_targets, self.robot_dof_lower_limits[:6], self.robot_dof_upper_limits[:6])
        self.robot_dof_targets[:, :6] = arm_targets
        
        # 그리퍼: 규칙 기반으로 확실하게 열기/닫기
        # should_close_gripper가 True면 70% 닫기, False면 완전히 열기
        grip_ratio = 0.7  # 70% 닫기 (큐브 폭에 맞게)
        
        for i in range(4):
            joint_idx = 6 + i
            open_pos = self.robot_dof_lower_limits[joint_idx]
            close_pos = self.robot_dof_upper_limits[joint_idx]
            partial_close_pos = open_pos + grip_ratio * (close_pos - open_pos)
            
            gripper_target = torch.where(
                self.should_close_gripper,
                torch.full_like(self.robot_dof_targets[:, joint_idx], partial_close_pos),
                torch.full_like(self.robot_dof_targets[:, joint_idx], open_pos),
            )
            self.robot_dof_targets[:, joint_idx] = gripper_target
        
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def _apply_action(self):
        pass

    def _get_observations(self) -> dict:
        """
        관측값 - 그리퍼 상태를 1D로 표현
        
        구성 (22D):
        - cubeA_pos (3) + cubeA_quat (4) = 7
        - cubeB_pos (3) = 3
        - eef_pos (3) + eef_quat (4) = 7
        - gripper_open (1) = 1
        - cubeA_relative (3) = 3
        - finger_distance (1) = 1
        
        Total: 22
        """
        env_origins = self.scene.env_origins
        
        eef_pos_w = self._robot.data.body_pos_w[:, self.hand_link_idx, :]
        eef_quat = self._robot.data.body_quat_w[:, self.hand_link_idx, :]
        finger_l_pos_w = self._robot.data.body_pos_w[:, self.finger_l_idx, :]
        finger_r_pos_w = self._robot.data.body_pos_w[:, self.finger_r_idx, :]
        
        eef_pos = eef_pos_w - env_origins
        finger_l_pos = finger_l_pos_w - env_origins
        finger_r_pos = finger_r_pos_w - env_origins
        
        cubeA_pos_w = self._cubeA.data.root_pos_w
        cubeA_quat = self._cubeA.data.root_quat_w
        cubeB_pos_w = self._cubeB.data.root_pos_w
        
        cubeA_pos = cubeA_pos_w - env_origins
        cubeB_pos = cubeB_pos_w - env_origins
        
        cubeA_relative = cubeA_pos - eef_pos
        
        # 그리퍼 열림 정도 (1D로 정규화)
        gripper_pos = self._robot.data.joint_pos[:, 6:10]
        gripper_closed_amount = gripper_pos[:, 0] + gripper_pos[:, 1]  # l1 + r1
        gripper_open_normalized = 1.0 - (gripper_closed_amount / self.max_gripper_open)  # 0=닫힘, 1=열림
        
        # 손가락 사이 거리 (잡기 상태 판단용)
        finger_distance = torch.norm(finger_l_pos - finger_r_pos, dim=-1, keepdim=True)

        obs = torch.cat([
            cubeA_pos,                          # 3
            cubeA_quat,                         # 4
            cubeB_pos,                          # 3
            eef_pos,                            # 3
            eef_quat,                           # 4
            gripper_open_normalized.unsqueeze(-1),  # 1
            cubeA_relative,                     # 3
            finger_distance,                    # 1
        ], dim=-1)  # Total: 22

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """
        단계별 보상 시스템 v5
        
        그리퍼 열고 닫기는 규칙 기반이므로:
        - RL은 팔을 움직여 큐브를 그리퍼 사이에 잘 위치시키는 것에 집중
        - 그리퍼가 자동으로 닫히면 보상
        """
        env_origins = self.scene.env_origins
        
        # ========== 상태 정보 수집 ==========
        eef_pos_w = self._robot.data.body_pos_w[:, self.hand_link_idx, :]
        eef_quat = self._robot.data.body_quat_w[:, self.hand_link_idx, :]
        finger_l_pos_w = self._robot.data.body_pos_w[:, self.finger_l_idx, :]
        finger_r_pos_w = self._robot.data.body_pos_w[:, self.finger_r_idx, :]
        cubeA_pos_w = self._cubeA.data.root_pos_w
        cubeB_pos_w = self._cubeB.data.root_pos_w
        
        # 로컬 좌표 변환
        eef_pos = eef_pos_w - env_origins
        finger_l_pos = finger_l_pos_w - env_origins
        finger_r_pos = finger_r_pos_w - env_origins
        cubeA_pos = cubeA_pos_w - env_origins
        cubeB_pos = cubeB_pos_w - env_origins
        
        # 그리퍼 상태
        gripper_pos = self._robot.data.joint_pos[:, 6:10]
        gripper_closed_amount = gripper_pos[:, 0] + gripper_pos[:, 1]  # l1 + r1
        gripper_open = self.max_gripper_open - gripper_closed_amount  # 클수록 열림
        
        # 거리 계산
        eef_to_cubeA = torch.norm(cubeA_pos - eef_pos, dim=-1)
        dl_to_cubeA = torch.norm(cubeA_pos - finger_l_pos, dim=-1)
        dr_to_cubeA = torch.norm(cubeA_pos - finger_r_pos, dim=-1)
        d_fingers_avg = (dl_to_cubeA + dr_to_cubeA + eef_to_cubeA) / 3.0
        
        # l2-r2 사이 거리 (실제 잡는 부분)
        finger_distance = torch.norm(finger_l_pos - finger_r_pos, dim=-1)
        
        # 그리퍼 중심점
        gripper_center = (finger_l_pos + finger_r_pos) / 2.0
        gripper_center_to_cube = torch.norm(cubeA_pos - gripper_center, dim=-1)
        
        # 높이 정보
        cubeA_height = cubeA_pos[:, 2] - self.table_height
        target_stack_height = self.cubeB_size + self.cubeA_size / 2.0
        
        # 그리퍼 방향
        qw, qx, qy, qz = eef_quat[:, 0], eef_quat[:, 1], eef_quat[:, 2], eef_quat[:, 3]
        gripper_z_component = 2.0 * (qx * qz + qw * qy)
        local_x_world_z = 2.0 * (qx * qz - qw * qy)
        
        # ========== 그리퍼 규칙 기반 제어 조건 ==========
        
        # 박스가 그리퍼 사이에 있는지 확인
        cube_xy = cubeA_pos[:, :2]
        gripper_center_xy = gripper_center[:, :2]
        xy_dist_to_cube = torch.norm(cube_xy - gripper_center_xy, dim=-1)
        height_diff_gripper_cube = torch.abs(gripper_center[:, 2] - cubeA_pos[:, 2])
        
        # 박스가 그리퍼 사이에 위치 (xy 거리 < 4cm, 높이 차이 < 4cm)
        box_between_fingers = (xy_dist_to_cube < 0.04) & (height_diff_gripper_cube < 0.04)
        
        # 그리퍼 닫기 조건 업데이트
        # 박스가 사이에 있거나, 이미 잡은 상태면 닫기 유지
        self.should_close_gripper = box_between_fingers | self.cube_grasped
        
        # ========== 잡기 상태 판단 ==========
        
        # 그리퍼가 닫혔고 큐브가 그리퍼 근처에 있으면 잡은 것
        gripper_closed = self.should_close_gripper
        cube_near_gripper = eef_to_cubeA < 0.08
        cube_grasped_now = gripper_closed & box_between_fingers & cube_near_gripper
        
        # 상태 업데이트
        self.cube_grasped = self.cube_grasped | cube_grasped_now
        
        # 현재 실제로 잡고 있는지 (큐브가 들려있고 그리퍼 근처)
        lifted_height = cubeA_height - self.cubeA_size / 2.0
        currently_holding = gripper_closed & (eef_to_cubeA < 0.1) & (lifted_height > 0.01)
        
        # 충분히 들어올렸는지
        cube_lifted = self.cube_grasped & (lifted_height > 0.05)
        self.cube_lifted = self.cube_lifted | cube_lifted
        
        # cubeB 위에 정렬
        xy_dist_to_cubeB = torch.norm(cubeA_pos[:, :2] - cubeB_pos[:, :2], dim=-1)
        aligned_over_cubeB = self.cube_lifted & (xy_dist_to_cubeB < 0.05)
        
        # 스택 성공
        height_diff = torch.abs(cubeA_height - target_stack_height)
        stack_success = aligned_over_cubeB & (height_diff < 0.02)
        
        # ========== Stage 1: 접근 보상 ==========
        approach_reward = 1.0 - torch.tanh(5.0 * d_fingers_avg)
        
        # 박스가 그리퍼 사이에 잘 위치하면 보너스
        positioning_bonus = box_between_fingers.float() * 3.0
        
        # ========== Stage 2: 그립 보상 ==========
        # 실제로 잡혔으면 보상
        grip_reward = cube_grasped_now.float() * 3.0
        
        # 안정적으로 잡고 있으면 추가 보상
        stable_grip_reward = currently_holding.float() * 2.0
        
        # ========== Stage 3: 리프트 보상 ==========
        lift_reward = currently_holding.float() * torch.tanh(10.0 * torch.clamp(lifted_height, min=0.0)) * 3.0
        
        lift_success_bonus = cube_lifted.float() * 5.0
        
        # ========== Stage 4: 이동 보상 ==========
        move_reward = self.cube_lifted.float() * (1.0 - torch.tanh(10.0 * xy_dist_to_cubeB)) * 3.0
        
        height_maintain_reward = self.cube_lifted.float() * (lifted_height > 0.03).float() * 0.5
        
        align_bonus = aligned_over_cubeB.float() * 5.0
        
        # ========== Stage 5: 스택 보상 ==========
        place_reward = aligned_over_cubeB.float() * (1.0 - torch.tanh(20.0 * height_diff)) * 2.0
        
        stack_success_bonus = stack_success.float() * 30.0
        
        # ========== 패널티 ==========
        
        # 그리퍼 뒤집힘
        flipped_penalty = (gripper_z_component > 0.0).float() * 5.0
        
        # 그리퍼 기울어짐
        cos_down = (-local_x_world_z).clamp(-1.0, 1.0)
        tilted_penalty = (1.0 - cos_down).clamp(min=0.0) * 2.0
        
        # 큐브 떨어뜨림
        cube_dropped = self.cube_grasped & ~currently_holding & (lifted_height < 0.01) & ~stack_success
        drop_penalty = cube_dropped.float() * 10.0
        
        # 바닥 충돌
        finger_min_height = torch.min(finger_l_pos[:, 2], finger_r_pos[:, 2])
        floor_collision = (finger_min_height < self.table_height).float()
        floor_penalty = floor_collision * 5.0
        
        # 액션 스무딩
        action_penalty = torch.sum(self.actions ** 2, dim=-1) * 0.005
        
        # ========== 총 보상 ==========
        rewards = (
            # Stage 1: 접근
            2.0 * approach_reward
            + positioning_bonus
            # Stage 2: 그립
            + grip_reward
            + stable_grip_reward
            # Stage 3: 리프트
            + lift_reward
            + lift_success_bonus
            # Stage 4: 이동
            + move_reward
            + height_maintain_reward
            + align_bonus
            # Stage 5: 스택
            + place_reward
            + stack_success_bonus
            # 패널티
            - flipped_penalty
            - tilted_penalty
            - drop_penalty
            - floor_penalty
            - action_penalty
        )
        
        # ========== TensorBoard 로깅 ==========
        self.extras["log"] = {
            # Stage별 보상
            "reward/stage1_approach": (2.0 * approach_reward).mean().item(),
            "reward/stage1_positioning": positioning_bonus.mean().item(),
            "reward/stage2_grip": grip_reward.mean().item(),
            "reward/stage2_stable_grip": stable_grip_reward.mean().item(),
            "reward/stage3_lift": lift_reward.mean().item(),
            "reward/stage3_lift_bonus": lift_success_bonus.mean().item(),
            "reward/stage4_move": move_reward.mean().item(),
            "reward/stage4_align_bonus": align_bonus.mean().item(),
            "reward/stage5_place": place_reward.mean().item(),
            "reward/stage5_stack_success": stack_success_bonus.mean().item(),
            
            # 패널티
            "penalty/flipped": flipped_penalty.mean().item(),
            "penalty/tilted": tilted_penalty.mean().item(),
            "penalty/drop": drop_penalty.mean().item(),
            "penalty/floor": floor_penalty.mean().item(),
            
            # 상태 정보
            "state/d_fingers_avg": d_fingers_avg.mean().item(),
            "state/gripper_open": gripper_open.mean().item(),
            "state/finger_distance": finger_distance.mean().item(),
            "state/xy_dist_to_cube": xy_dist_to_cube.mean().item(),
            "state/cubeA_height": cubeA_height.mean().item(),
            "state/xy_dist_to_cubeB": xy_dist_to_cubeB.mean().item(),
            "state/eef_to_cubeA": eef_to_cubeA.mean().item(),
            
            # 진행률
            "progress/box_between_fingers": box_between_fingers.sum().item(),
            "progress/gripper_closed": gripper_closed.sum().item(),
            "progress/cube_grasped_now": cube_grasped_now.sum().item(),
            "progress/currently_holding": currently_holding.sum().item(),
            "progress/cube_grasped_count": self.cube_grasped.sum().item(),
            "progress/cube_lifted_count": self.cube_lifted.sum().item(),
            "progress/aligned_count": aligned_over_cubeB.sum().item(),
            "progress/stack_success_count": stack_success.sum().item(),
        }

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """종료 조건"""
        env_origins = self.scene.env_origins
        
        eef_pos_w = self._robot.data.body_pos_w[:, self.hand_link_idx, :]
        eef_quat = self._robot.data.body_quat_w[:, self.hand_link_idx, :]
        cubeA_pos_w = self._cubeA.data.root_pos_w
        cubeB_pos_w = self._cubeB.data.root_pos_w

        finger_l_pos_w = self._robot.data.body_pos_w[:, self.finger_l_idx, :]
        finger_r_pos_w = self._robot.data.body_pos_w[:, self.finger_r_idx, :]
        
        eef_pos = eef_pos_w - env_origins
        cubeA_pos = cubeA_pos_w - env_origins
        cubeB_pos = cubeB_pos_w - env_origins

        finger_l_pos = finger_l_pos_w - env_origins
        finger_r_pos = finger_r_pos_w - env_origins
        finger_min_height = torch.min(finger_l_pos[:, 2], finger_r_pos[:, 2])

        cubeA_height = cubeA_pos[:, 2] - self.table_height
        target_height = self.cubeB_size + self.cubeA_size / 2.0
        
        xy_dist = torch.norm(cubeA_pos[:, :2] - cubeB_pos[:, :2], dim=-1)
        height_diff = torch.abs(cubeA_height - target_height)
        eef_to_cubeA = torch.norm(cubeA_pos - eef_pos, dim=-1)

        # 성공 조건
        stack_success = (xy_dist < 0.03) & (height_diff < 0.02)
        
        # 실패 조건
        cubeA_fallen = cubeA_pos[:, 2] < (self.table_height - 0.1)
        cubeB_fallen = cubeB_pos[:, 2] < (self.table_height - 0.1)
        finger_floor_collision = finger_min_height < (self.table_height + 0.02)

        qw, qx, qy, qz = eef_quat[:, 0], eef_quat[:, 1], eef_quat[:, 2], eef_quat[:, 3]
        gripper_z = 2.0 * (qx * qz + qw * qy)
        gripper_flipped = gripper_z > 0.5
        
        eef_dist = torch.norm(eef_pos[:, :2] - torch.tensor([-0.45, 0.0], device=self.device), dim=-1)
        workspace_violation = eef_dist > 1.0

        terminated = stack_success | cubeA_fallen | cubeB_fallen | finger_floor_collision | gripper_flipped | workspace_violation
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        
        self.extras.setdefault("log", {})
        self.extras["log"].update({
            "done/stack_success": stack_success.sum().item(),
            "done/cubeA_fallen": cubeA_fallen.sum().item(),
            "done/cubeB_fallen": cubeB_fallen.sum().item(),
            "done/floor_collision": finger_floor_collision.sum().item(),
            "done/gripper_flipped": gripper_flipped.sum().item(),
            "done/workspace_violation": workspace_violation.sum().item(),
            "done/timeout": (self.episode_length_buf >= self.max_episode_length - 1).sum().item(),
        })

        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int]):
        """환경 리셋"""
        if len(env_ids) == 0:
            return

        # 로봇 리셋
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_pos += 0.1 * (torch.rand_like(joint_pos) - 0.5)
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        
        # 그리퍼 완전히 열기
        joint_pos[:, 6:10] = self.robot_dof_lower_limits[6:10]

        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.robot_dof_targets[env_ids] = joint_pos

        # 큐브 리셋
        self._reset_cubes(env_ids)
        
        # 상태 초기화
        self.cube_grasped[env_ids] = False
        self.cube_lifted[env_ids] = False
        self.should_close_gripper[env_ids] = False

        super()._reset_idx(env_ids)

    def _reset_cubes(self, env_ids: Sequence[int]):
        """큐브 위치 리셋"""
        env_origins = self.scene.env_origins[env_ids]
        num_resets = len(env_ids)
        
        # CubeB (팔레트) 위치 - 고정
        cubeB_pos_local = torch.zeros((num_resets, 3), device=self.device)
        cubeB_pos_local[:, 0] = 0.23
        cubeB_pos_local[:, 1] = 0
        cubeB_pos_local[:, 2] = self.table_height + self.cubeB_size / 2.0
        cubeB_pos = cubeB_pos_local + env_origins
        cubeB_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(num_resets, 1)
        
        self._cubeB.write_root_pose_to_sim(torch.cat([cubeB_pos, cubeB_quat], dim=-1), env_ids)
        self._cubeB.write_root_velocity_to_sim(torch.zeros((num_resets, 6), device=self.device), env_ids)

        # CubeA (집을 큐브) 위치 - 랜덤
        xmin, xmax = -0.13, 0.06
        ymin, ymax = -0.15, 0.15
        cubeA_pos_local = torch.zeros((num_resets, 3), device=self.device)
        cubeA_pos_local[:, 0] = xmin + (xmax - xmin) * torch.rand(num_resets, device=self.device)
        cubeA_pos_local[:, 1] = ymin + (ymax - ymin) * torch.rand(num_resets, device=self.device)
        cubeA_pos_local[:, 2] = self.table_height + self.cubeA_size / 2.0
        cubeA_pos = cubeA_pos_local + env_origins
        cubeA_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(num_resets, 1)
        
        self._cubeA.write_root_pose_to_sim(torch.cat([cubeA_pos, cubeA_quat], dim=-1), env_ids)
        self._cubeA.write_root_velocity_to_sim(torch.zeros((num_resets, 6), device=self.device), env_ids)
        
        # 초기 위치 저장
        self.cubeA_initial_pos[env_ids] = cubeA_pos_local
