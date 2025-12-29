#!/usr/bin/env python3
"""
E0509 그리퍼 테스트 스크립트

그리퍼 동작을 검증하기 위한 스크립트:
1. 사인파로 열고 닫기 반복
2. finger_distance 실시간 출력
3. 강성(stiffness) 확인

실행 방법:
cd ~/IsaacLab
./isaaclab.sh -p <이 스크립트 경로>

예시:
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/e0509_cube_stack/test_gripper.py
"""

import argparse
import torch
import math
import os

from isaaclab.app import AppLauncher

# 인자 파싱
parser = argparse.ArgumentParser(description="E0509 Gripper Test")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Isaac Sim 실행
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Lab 임포트 (시뮬레이터 실행 후)
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext, SimulationCfg

##############################################################################
# 설정
##############################################################################

E0509_USD_PATH = os.path.expanduser("~/IsaacLab/source/isaaclab_assets/data/Robots/Doosan/E0509_rl_basic.usd")

# 그리퍼 강성 테스트를 위해 값 조절 가능
GRIPPER_STIFFNESS = 5000.0  # 기본값, 필요시 증가
GRIPPER_DAMPING = 200.0     # 기본값, 필요시 증가

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
        pos=(0.0, 0.0, 1.05),
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
    actuators={
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-3]"],
            velocity_limit=87.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "forearm": ImplicitActuatorCfg(
            joint_names_expr=["joint_[4-6]"],
            velocity_limit=87.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["rh_l1", "rh_l2", "rh_r1", "rh_r2"],
            velocity_limit=1.0,
            effort_limit=400.0,
            stiffness=GRIPPER_STIFFNESS,
            damping=GRIPPER_DAMPING,
        ),
    },
    prim_path="/World/Robot",
)

# 테스트용 큐브 (그리퍼로 잡아볼 용도)
CUBE_CFG = RigidObjectCfg(
    prim_path="/World/Cube",
    spawn=sim_utils.CuboidCfg(
        size=(0.085, 0.12, 0.035),  # cubeA와 동일
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),  # 그리퍼 아래에 위치
)

# 테이블
TABLE_CFG = RigidObjectCfg(
    prim_path="/World/Table",
    spawn=sim_utils.CuboidCfg(
        size=(1.0, 1.0, 0.05),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.975)),
)


def main():
    """메인 함수"""
    
    # 시뮬레이션 컨텍스트
    sim_cfg = SimulationCfg(dt=1/120, render_interval=2)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(1.5, 1.5, 1.5), target=(0.0, 0.0, 1.0))
    
    # 조명
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    
    # 바닥
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/Ground", cfg)
    
    # 에셋 생성
    robot = Articulation(E0509_CFG)
    table = RigidObject(TABLE_CFG)
    cube = RigidObject(CUBE_CFG)
    
    # 시뮬레이션 시작
    sim.reset()
    
    # 관절 정보 출력
    print("\n" + "="*60)
    print("E0509 그리퍼 테스트")
    print("="*60)
    
    joint_names = robot.data.joint_names
    print(f"\n관절 이름: {joint_names}")
    
    dof_lower = robot.data.soft_joint_pos_limits[0, :, 0]
    dof_upper = robot.data.soft_joint_pos_limits[0, :, 1]
    
    print(f"\n그리퍼 관절 범위:")
    for i in range(6, 10):
        print(f"  {joint_names[i]}: {dof_lower[i]:.3f} ~ {dof_upper[i]:.3f}")
    
    # 링크 인덱스
    finger_l_idx = robot.find_bodies("rh_p12_rn_l2")[0][0]
    finger_r_idx = robot.find_bodies("rh_p12_rn_r2")[0][0]
    tool_idx = robot.find_bodies("tool0")[0][0]
    
    print(f"\n링크 인덱스:")
    print(f"  tool0: {tool_idx}")
    print(f"  rh_p12_rn_l2: {finger_l_idx}")
    print(f"  rh_p12_rn_r2: {finger_r_idx}")
    
    # 그리퍼 열림/닫힘 위치
    gripper_open_pos = dof_lower[6:10].clone()
    gripper_close_pos = dof_upper[6:10].clone()
    
    print(f"\n그리퍼 열림 위치: {gripper_open_pos}")
    print(f"그리퍼 닫힘 위치: {gripper_close_pos}")
    
    # 타겟 위치 초기화
    joint_targets = robot.data.default_joint_pos.clone()
    
    # 팔 고정 (그리퍼가 아래를 향하도록)
    joint_targets[:, 0] = 0.0      # joint_1
    joint_targets[:, 1] = 0.0      # joint_2
    joint_targets[:, 2] = 1.5708   # joint_3 (90도)
    joint_targets[:, 3] = 0.0      # joint_4
    joint_targets[:, 4] = 1.5708   # joint_5 (90도)
    joint_targets[:, 5] = 0.0      # joint_6
    
    # 그리퍼 열림 상태로 시작
    joint_targets[:, 6:10] = gripper_open_pos
    
    print("\n" + "="*60)
    print("테스트 시작: 사인파로 그리퍼 열고 닫기")
    print("키보드 인터럽트(Ctrl+C)로 종료")
    print("="*60 + "\n")
    
    # 시뮬레이션 루프
    step = 0
    cycle_period = 240  # 2초 주기 (120Hz * 2)
    
    try:
        while simulation_app.is_running():
            # 사인파로 그리퍼 제어 (0~1 범위)
            t = (step % cycle_period) / cycle_period  # 0 ~ 1
            gripper_ratio = (1 - math.cos(2 * math.pi * t)) / 2  # 0 → 1 → 0 (부드럽게)
            
            # 그리퍼 타겟 계산 (4개 관절 동기화)
            for i in range(4):
                joint_targets[:, 6 + i] = (
                    gripper_open_pos[i] + 
                    gripper_ratio * (gripper_close_pos[i] - gripper_open_pos[i])
                )
            
            # 타겟 적용
            robot.set_joint_position_target(joint_targets)
            
            # 시뮬레이션 스텝
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim_cfg.dt)
            cube.update(sim_cfg.dt)
            
            # 20스텝마다 정보 출력
            if step % 20 == 0:
                # 현재 그리퍼 관절 위치
                joint_pos = robot.data.joint_pos[0, 6:10]
                
                # 손가락 위치
                finger_l_pos = robot.data.body_pos_w[0, finger_l_idx]
                finger_r_pos = robot.data.body_pos_w[0, finger_r_idx]
                finger_distance = torch.norm(finger_l_pos - finger_r_pos).item()
                
                # 손가락 방향 (z축 확인)
                finger_l_z = finger_l_pos[2].item()
                finger_r_z = finger_r_pos[2].item()
                z_diff = abs(finger_l_z - finger_r_z)
                
                # 큐브 위치
                cube_pos = cube.data.root_pos_w[0]
                
                # 타겟 vs 실제 비교
                target_l1 = joint_targets[0, 6].item()
                actual_l1 = joint_pos[0].item()
                pos_error = abs(target_l1 - actual_l1)
                
                print(f"Step {step:5d} | "
                      f"Ratio: {gripper_ratio:.2f} | "
                      f"Finger Dist: {finger_distance*100:.1f}cm | "
                      f"Z Diff: {z_diff*1000:.1f}mm | "
                      f"Target L1: {target_l1:.3f} | "
                      f"Actual L1: {actual_l1:.3f} | "
                      f"Error: {pos_error:.4f}")
            
            step += 1
            
    except KeyboardInterrupt:
        print("\n\n테스트 종료")
    
    # 정리
    simulation_app.close()


if __name__ == "__main__":
    main()
