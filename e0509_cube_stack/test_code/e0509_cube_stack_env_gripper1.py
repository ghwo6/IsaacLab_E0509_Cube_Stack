# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Doosan E0509 Cube Stack 강화학습 환경 v3
- 6축 로봇 + RH-P12-RN 그리퍼
- 단계별 보상 시스템 (Staged Reward)
- 새로운 그립 로직: 완전히 열고 접근 → 박스가 사이에 오면 닫기
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
            # RH-P12-RN 그리퍼 관절 (4개)
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
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["rh_l1", "rh_r1"],
            velocity_limit=0.5,
            effort_limit=200.0,
            stiffness=2000.0,
            damping=100.0,
        ),
    },
)


@configclass
class E0509CubeStackEnvCfg(DirectRLEnvCfg):
    """E0509 Cube Stack 환경 설정"""

    # 환경 설정
    episode_length_s = 12.0
    decimation = 2
    action_scale = 1.5
    action_space = 10  # 6 arm + 4 gripper
    observation_space = 25
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
    E0509 로봇 Cube Stack 환경 v3
    
    새로운 그립 로직:
    1. 그리퍼 완전히 열고 접근
    2. 박스가 l2-r2 사이에 위치하면 닫기 시작
    3. 박스 폭에 맞게 적당히 닫기
    """

    cfg: E0509CubeStackEnvCfg

    def __init__(self, cfg: E0509CubeStackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 관절 제한
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(self.device)
        
        # 타겟 위치
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

        # 행동 저장
        self.actions = torch.zeros((self.num_envs, 10), device=self.device)
        
        # 상태 추적
        self.cubeA_initial_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.cube_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cube_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 그리퍼 설정
        self.max_gripper_open = self.robot_dof_upper_limits[6] + self.robot_dof_upper_limits[8]
        
        # 디버그 출력
        # print(f"=== Gripper Config ===")
        # print(f"Max gripper open: {self.max_gripper_open:.3f} rad")
        # print(f"Cube width: {self.cubeA_width * 100:.1f} cm")


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
        """행동 적용"""
        self.actions = actions.clone().clamp(-1.0, 1.0)
        
        targets = self.robot_dof_targets + self.cfg.action_scale * self.cfg.sim.dt * self.actions
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def _apply_action(self):
        pass

    def _get_observations(self) -> dict:
        """관측값"""
        env_origins = self.scene.env_origins
        
        eef_pos_w = self._robot.data.body_pos_w[:, self.hand_link_idx, :]
        eef_quat = self._robot.data.body_quat_w[:, self.hand_link_idx, :]
        gripper_pos = self._robot.data.joint_pos[:, 6:10]
        
        eef_pos = eef_pos_w - env_origins
        
        cubeA_pos_w = self._cubeA.data.root_pos_w
        cubeA_quat = self._cubeA.data.root_quat_w
        cubeB_pos_w = self._cubeB.data.root_pos_w
        
        cubeA_pos = cubeA_pos_w - env_origins
        cubeB_pos = cubeB_pos_w - env_origins
        
        cubeA_relative = cubeA_pos - eef_pos
        cubeA_to_cubeB = cubeB_pos - cubeA_pos

        obs = torch.cat([
            cubeA_pos, cubeA_quat,     # 7
            cubeB_pos,                  # 3
            eef_pos, eef_quat,         # 7
            gripper_pos,                # 4
            cubeA_relative,             # 3
            cubeA_to_cubeB,             # 3
        ], dim=-1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """
        단계별 보상 시스템 v3
        
        새로운 그립 로직:
        1. 그리퍼 완전히 열고 접근 (gripper_open > 0.9 * max)
        2. 박스가 l2-r2 사이에 위치 확인 (finger_distance와 cube 위치로 판단)
        3. 박스 폭에 맞게 닫기
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
        # 수정: 값이 작을수록 열림 → 반전해서 사용
        gripper_closed_amount = gripper_pos[:, 0] + gripper_pos[:, 2]  # 클수록 닫힘
        max_gripper_closed = self.robot_dof_upper_limits[6] + self.robot_dof_upper_limits[8]
        gripper_open = max_gripper_closed - gripper_closed_amount  # 클수록 열림
        
        # 거리 계산
        eef_to_cubeA = torch.norm(cubeA_pos - eef_pos, dim=-1)
        dl_to_cubeA = torch.norm(cubeA_pos - finger_l_pos, dim=-1)
        dr_to_cubeA = torch.norm(cubeA_pos - finger_r_pos, dim=-1)
        d_fingers_avg = (dl_to_cubeA + dr_to_cubeA + eef_to_cubeA) / 3.0
        
        # l2-r2 사이 거리 (그리퍼 열림 정도)
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
        
        # ========== 새로운 그립 조건 ==========
        
        # 1. 그리퍼가 충분히 열려있는지 (90% 이상)
        gripper_wide_open = gripper_open > (self.max_gripper_open * 0.9)
        
        # 2. 박스가 그리퍼 사이에 있는지
        #    - 그리퍼 중심이 큐브 근처 (xy 평면에서)
        #    - 그리퍼 높이가 큐브 높이와 비슷
        cube_xy = cubeA_pos[:, :2]
        gripper_center_xy = gripper_center[:, :2]
        xy_dist_to_cube = torch.norm(cube_xy - gripper_center_xy, dim=-1)
        
        height_diff_gripper_cube = torch.abs(gripper_center[:, 2] - cubeA_pos[:, 2])
        
        # 박스가 그리퍼 사이에 위치 (xy 거리 < 3cm, 높이 차이 < 3cm)
        box_between_fingers = (xy_dist_to_cube < 0.03) & (height_diff_gripper_cube < 0.03)
        
        # 3. 잡을 준비 완료 (열린 그리퍼 + 박스가 사이에 있음)
        grasp_ready = gripper_wide_open & box_between_fingers
        
        # 4. 박스를 잡았는지 (그리퍼가 박스 폭에 맞게 닫힘)
        #    박스 폭 8.5cm에 약간의 마진 추가
        #    finger_distance가 박스 폭보다 약간 큰 상태 (8.5cm ~ 10cm)
        gripper_holding = (finger_distance > self.cubeA_width) & (finger_distance < self.cubeA_width + 0.025)
        cube_grasped = box_between_fingers & gripper_holding
        
        # 상태 업데이트
        self.cube_grasped = self.cube_grasped | cube_grasped
        
        # 충분히 들어올렸는지
        lifted_height = cubeA_height - self.cubeA_size / 2.0
        cube_lifted = self.cube_grasped & (lifted_height > 0.05)
        self.cube_lifted = self.cube_lifted | cube_lifted
        
        # cubeB 위에 정렬
        xy_dist_to_cubeB = torch.norm(cubeA_pos[:, :2] - cubeB_pos[:, :2], dim=-1)
        aligned_over_cubeB = self.cube_lifted & (xy_dist_to_cubeB < 0.03)
        
        # 스택 성공
        height_diff = torch.abs(cubeA_height - target_stack_height)
        stack_success = aligned_over_cubeB & (height_diff < 0.02) & (eef_to_cubeA > 0.04)
        
        # ========== Stage 1: 접근 보상 ==========
        # 그리퍼 열고 접근
        approach_reward = 1.0 - torch.tanh(5.0 * d_fingers_avg)
        
        # 접근할 때 그리퍼 열기 보상 (가까울수록 더 열어야 함)
        proximity = torch.clamp(1.0 - d_fingers_avg / 0.2, min=0.0)
        open_gripper_reward = proximity * (gripper_open / self.max_gripper_open)
        
        # 잡을 준비 완료 보너스 (열린 그리퍼로 박스 사이에 위치)
        grasp_ready_bonus = grasp_ready.float() * 3.0
        
        # ========== Stage 2: 그립 보상 ==========
        # grasp_ready 상태에서 닫기 유도
        # 목표: finger_distance가 cubeA_width + 약간의 마진이 되도록
        target_finger_dist = self.cubeA_width + 0.01  # 8.5cm + 1cm 마진
        
        # grasp_ready일 때만 닫기 보상 (너무 많이 닫지 않도록)
        grip_error = torch.abs(finger_distance - target_finger_dist)
        grip_reward = grasp_ready.float() * (1.0 - torch.tanh(20.0 * grip_error))
        
        # 실제로 잡았을 때 보너스
        grasp_success_bonus = cube_grasped.float() * 5.0
        
        # ========== Stage 3: 리프트 보상 ==========
        lift_reward = self.cube_grasped.float() * torch.tanh(10.0 * torch.clamp(lifted_height, min=0.0))
        
        # xy 이동 최소화 (수직 리프트)
        xy_drift = torch.norm(cubeA_pos[:, :2] - self.cubeA_initial_pos[:, :2], dim=-1)
        xy_drift_penalty = self.cube_grasped.float() * (~self.cube_lifted).float() * xy_drift * 2.0
        
        lift_success_bonus = cube_lifted.float() * 3.0
        
        # ========== Stage 4: 이동 보상 ==========
        move_reward = self.cube_lifted.float() * (1.0 - torch.tanh(10.0 * xy_dist_to_cubeB))
        height_maintain_reward = self.cube_lifted.float() * (lifted_height > 0.03).float() * 0.5
        align_bonus = aligned_over_cubeB.float() * 3.0
        
        # ========== Stage 5: 스택 보상 ==========
        place_reward = aligned_over_cubeB.float() * (1.0 - torch.tanh(20.0 * height_diff))
        
        # 내려놓을 때 그리퍼 열기
        release_reward = aligned_over_cubeB.float() * (height_diff < 0.03).float() * (gripper_open / self.max_gripper_open)
        
        stack_success_bonus = stack_success.float() * 20.0
        
        # ========== 패널티 ==========
        
        # 그리퍼 뒤집힘
        flipped_penalty = (gripper_z_component > 0.0).float() * 5.0
        
        # 그리퍼 기울어짐
        cos_down = (-local_x_world_z).clamp(-1.0, 1.0)
        tilted_penalty = (1.0 - cos_down).clamp(min=0.0) * 2.0
        
        # cubeB 충돌
        too_close_to_cubeB = torch.norm(cubeA_pos - cubeB_pos, dim=-1) < 0.06
        cubeB_collision = self.cube_lifted & too_close_to_cubeB & (cubeA_height < self.cubeB_size + 0.02)
        collision_penalty = cubeB_collision.float() * 3.0
        
        # 큐브 떨어뜨림
        cube_dropped = self.cube_grasped & (eef_to_cubeA > 0.15) & (cubeA_height < 0.02)
        drop_penalty = cube_dropped.float() * 5.0
        
        # 바닥 충돌
        finger_min_height = torch.min(finger_l_pos[:, 2], finger_r_pos[:, 2])
        floor_collision = (finger_min_height < self.table_height).float()
        floor_penalty = floor_collision * 5.0
        
        # 접근 중 그리퍼 닫기 패널티 (grasp_ready 전에 닫으면 안됨)
        not_ready = ~grasp_ready & ~self.cube_grasped
        gripper_closed_early = gripper_open < (self.max_gripper_open * 0.7)
        premature_close_penalty = (not_ready & gripper_closed_early).float() * 2.0
        
        # 액션 스무딩
        action_penalty = torch.sum(self.actions ** 2, dim=-1) * 0.005
        
        # ========== 총 보상 ==========
        rewards = (
            # Stage 1: 접근
            2.0 * approach_reward
            + 2.0 * open_gripper_reward
            + grasp_ready_bonus
            # Stage 2: 그립
            + 3.0 * grip_reward
            + grasp_success_bonus
            # Stage 3: 리프트
            + 3.0 * lift_reward
            + lift_success_bonus
            - xy_drift_penalty
            # Stage 4: 이동
            + 3.0 * move_reward
            + height_maintain_reward
            + align_bonus
            # Stage 5: 스택
            + 2.0 * place_reward
            + 2.0 * release_reward
            + stack_success_bonus
            # 패널티
            - flipped_penalty
            - tilted_penalty
            - collision_penalty
            - drop_penalty
            - floor_penalty
            - premature_close_penalty
            - action_penalty
        )
        
        # ========== TensorBoard 로깅 ==========
        self.extras["log"] = {
            # Stage별 보상
            "reward/stage1_approach": (2.0 * approach_reward).mean().item(),
            "reward/stage1_open_gripper": (2.0 * open_gripper_reward).mean().item(),
            "reward/stage1_grasp_ready_bonus": grasp_ready_bonus.mean().item(),
            "reward/stage2_grip": (3.0 * grip_reward).mean().item(),
            "reward/stage2_grasp_success": grasp_success_bonus.mean().item(),
            "reward/stage3_lift": (3.0 * lift_reward).mean().item(),
            "reward/stage3_lift_bonus": lift_success_bonus.mean().item(),
            "reward/stage4_move": (3.0 * move_reward).mean().item(),
            "reward/stage4_align_bonus": align_bonus.mean().item(),
            "reward/stage5_place": (2.0 * place_reward).mean().item(),
            "reward/stage5_stack_success": stack_success_bonus.mean().item(),
            
            # 패널티
            "penalty/flipped": flipped_penalty.mean().item(),
            "penalty/tilted": tilted_penalty.mean().item(),
            "penalty/collision": collision_penalty.mean().item(),
            "penalty/drop": drop_penalty.mean().item(),
            "penalty/floor": floor_penalty.mean().item(),
            "penalty/premature_close": premature_close_penalty.mean().item(),
            
            # 상태 정보
            "state/d_fingers_avg": d_fingers_avg.mean().item(),
            "state/gripper_open": gripper_open.mean().item(),
            "state/finger_distance": finger_distance.mean().item(),
            "state/xy_dist_to_cube": xy_dist_to_cube.mean().item(),
            "state/cubeA_height": cubeA_height.mean().item(),
            "state/xy_dist_to_cubeB": xy_dist_to_cubeB.mean().item(),
            "state/gripper_z": gripper_z_component.mean().item(),
            
            # 단계 진행률
            "progress/gripper_wide_open": gripper_wide_open.sum().item(),
            "progress/box_between_fingers": box_between_fingers.sum().item(),
            "progress/grasp_ready_count": grasp_ready.sum().item(),
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
        stack_success = (xy_dist < 0.02) & (height_diff < 0.02) & (eef_to_cubeA > 0.04)
        
        # 실패 조건
        cubeA_fallen = cubeA_pos[:, 2] < (self.table_height - 0.1)
        cubeB_fallen = cubeB_pos[:, 2] < (self.table_height - 0.1)
        finger_floor_collision = finger_min_height < (self.table_height + 0.03)

        qw, qx, qy, qz = eef_quat[:, 0], eef_quat[:, 1], eef_quat[:, 2], eef_quat[:, 3]
        gripper_z = 2.0 * (qx * qz + qw * qy)
        gripper_flipped = gripper_z > 0.5
        
        eef_dist = torch.norm(eef_pos[:, :2] - torch.tensor([-0.45, 0.0], device=self.device), dim=-1)
        workspace_violation = eef_dist > 1.0

        terminated = stack_success | cubeA_fallen | cubeB_fallen | finger_floor_collision | gripper_flipped | workspace_violation
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        
        # 종료 원인 로깅
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
