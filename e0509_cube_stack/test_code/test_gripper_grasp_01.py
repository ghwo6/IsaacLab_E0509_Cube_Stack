#!/usr/bin/env python3
"""
E0509 그리퍼 잡기 테스트 (단순 버전)

로봇 초기 자세에서:
1. 그리퍼 사이에 큐브가 이미 위치 (높은 받침대 위)
2. 그리퍼 닫기
3. 들어올리기
4. 큐브가 잘 잡히는지 확인

실행 방법:
cd ~/IsaacLab
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/e0509_cube_stack/test_gripper_grasp.py
"""

import argparse
import torch
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="E0509 Gripper Grasp Test")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext, SimulationCfg

##############################################################################
# 설정
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
        pos=(0.0, 0.0, 0.0),  # 바닥에 로봇 배치
        joint_pos={
            "joint_1": 0.0,
            "joint_2": 0.0,
            "joint_3": 1.5708,  # 90도 - 그리퍼가 아래를 향함
            "joint_4": 0.0,
            "joint_5": 1.5708,  # 90도
            "joint_6": 0.0,
            "rh_l1": 0.0,  # 열림
            "rh_r1": 0.0,
            "rh_l2": 0.0,
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
            velocity_limit=2.0,
            effort_limit=1000.0,
            stiffness=10000.0,
            damping=500.0,
        ),
    },
    prim_path="/World/Robot",
)

# 높은 받침대 (큐브를 그리퍼 사이에 위치시키기 위해)
# 높이는 시뮬레이션 시작 후 그리퍼 위치 확인하여 조정
PEDESTAL_CFG = RigidObjectCfg(
    prim_path="/World/Pedestal",
    spawn=sim_utils.CuboidCfg(
        size=(0.2, 0.2, 0.24),  # 받침대
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.37, 0.0, 0.12)),
)

# 큐브 (8.5cm x 12cm x 3.5cm)
CUBE_CFG = RigidObjectCfg(
    prim_path="/World/Cube",
    spawn=sim_utils.CuboidCfg(
        size=(0.085, 0.12, 0.035),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.37, 0.0, 0.2575)),
)


def main():
    sim_cfg = SimulationCfg(dt=1/120, render_interval=2)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(1.0, 1.0, 1.0), target=(0.0, 0.0, 0.5))
    
    # 조명
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    
    # 바닥
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/Ground", cfg)
    
    # 에셋 생성
    robot = Articulation(E0509_CFG)
    pedestal = RigidObject(PEDESTAL_CFG)
    cube = RigidObject(CUBE_CFG)
    
    sim.reset()
    
    print("\n" + "="*60)
    print("E0509 그리퍼 잡기 테스트 (단순 버전)")
    print("="*60)
    
    # 관절 정보
    joint_names = robot.data.joint_names
    dof_lower = robot.data.soft_joint_pos_limits[0, :, 0]
    dof_upper = robot.data.soft_joint_pos_limits[0, :, 1]
    
    print(f"\n관절 이름: {joint_names}")
    
    # 링크 인덱스
    finger_l_idx = robot.find_bodies("rh_p12_rn_l2")[0][0]
    finger_r_idx = robot.find_bodies("rh_p12_rn_r2")[0][0]
    tool_idx = robot.find_bodies("tool0")[0][0]
    
    # 초기 위치 확인을 위해 몇 스텝 실행
    for _ in range(10):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_cfg.dt)
    
    # 초기 EEF 위치 출력
    eef_pos = robot.data.body_pos_w[0, tool_idx]
    finger_l_pos = robot.data.body_pos_w[0, finger_l_idx]
    finger_r_pos = robot.data.body_pos_w[0, finger_r_idx]
    gripper_center = (finger_l_pos + finger_r_pos) / 2.0
    
    print(f"\n초기 위치:")
    print(f"  Tool0 (EEF): ({eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f})")
    print(f"  Finger L2: ({finger_l_pos[0]:.3f}, {finger_l_pos[1]:.3f}, {finger_l_pos[2]:.3f})")
    print(f"  Finger R2: ({finger_r_pos[0]:.3f}, {finger_r_pos[1]:.3f}, {finger_r_pos[2]:.3f})")
    print(f"  Gripper Center: ({gripper_center[0]:.3f}, {gripper_center[1]:.3f}, {gripper_center[2]:.3f})")
    
    cube_pos = cube.data.root_pos_w[0]
    print(f"  Cube: ({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f})")
    
    print(f"\n========================================")
    print(f"큐브 위치를 그리퍼 센터에 맞게 조정하세요!")
    print(f"받침대 상단 Z = 0.5")
    print(f"큐브가 받침대 위: Z = 0.5 + 0.0175 = 0.5175")
    print(f"그리퍼 센터 Z = {gripper_center[2]:.3f}")
    print(f"========================================")
    
    # 그리퍼 열림/닫힘 위치
    gripper_open_pos = dof_lower[6:10].clone()
    gripper_close_pos = dof_upper[6:10].clone()
    grip_ratio = 0.75
    
    # 타겟 초기화
    joint_targets = robot.data.default_joint_pos.clone()
    
    print("\n" + "="*60)
    print("테스트 진행:")
    print("  Phase 0 (0-3초): 그리퍼 열린 상태 유지")
    print("  Phase 1 (3-6초): 그리퍼 닫기")
    print("  Phase 2 (6-9초): 팔 들어올리기 (joint_2 변경)")
    print("  Phase 3 (9-15초): 유지 및 관찰")
    print("="*60 + "\n")
    
    step = 0
    phase = 0
    phase_steps = [360, 360, 360, 720]  # 3초, 3초, 3초, 6초
    phase_start = 0
    
    initial_cube_z = None
    
    try:
        while simulation_app.is_running():
            # Phase 전환
            if step - phase_start >= phase_steps[phase] and phase < 3:
                phase += 1
                phase_start = step
                print(f"\n>>> Phase {phase} 시작 (step {step})")
            
            # Phase별 동작
            if phase == 0:
                # 그리퍼 열린 상태 유지
                joint_targets[:, 6:10] = gripper_open_pos
                
            elif phase == 1:
                # 그리퍼 닫기
                for i in range(4):
                    open_p = gripper_open_pos[i].item()
                    close_p = gripper_close_pos[i].item()
                    joint_targets[:, 6+i] = open_p + grip_ratio * (close_p - open_p)
                    
            elif phase == 2:
                # 팔 들어올리기 (joint_2만 조금 변경)
                joint_targets[:, 1] = -0.3  # joint_2를 뒤로 (위로 올라감)
                
            elif phase == 3:
                # 유지
                pass
            
            robot.set_joint_position_target(joint_targets)
            
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim_cfg.dt)
            cube.update(sim_cfg.dt)
            pedestal.update(sim_cfg.dt)
            
            # 60스텝마다 출력
            if step % 60 == 0:
                joint_pos = robot.data.joint_pos[0]
                finger_l_pos = robot.data.body_pos_w[0, finger_l_idx]
                finger_r_pos = robot.data.body_pos_w[0, finger_r_idx]
                finger_distance = torch.norm(finger_l_pos - finger_r_pos).item()
                z_diff = abs(finger_l_pos[2].item() - finger_r_pos[2].item())
                
                cube_pos = cube.data.root_pos_w[0]
                cube_z = cube_pos[2].item()
                
                if initial_cube_z is None:
                    initial_cube_z = cube_z
                
                cube_lifted = cube_z - initial_cube_z
                
                l1, r1, l2, r2 = joint_pos[6].item(), joint_pos[7].item(), joint_pos[8].item(), joint_pos[9].item()
                
                print(f"Phase {phase} | Step {step:4d} | "
                      f"Finger: {finger_distance*100:.1f}cm | "
                      f"Z Diff: {z_diff*1000:.1f}mm | "
                      f"Cube Z: {cube_z:.3f} | "
                      f"Lifted: {cube_lifted*100:.1f}cm | "
                      f"l1:{l1:.2f} r1:{r1:.2f} l2:{l2:.2f} r2:{r2:.2f}")
            
            step += 1
            
            if step > 1800:  # 15초
                break
                
    except KeyboardInterrupt:
        print("\n\n테스트 중단")
    
    print("\n" + "="*60)
    print("테스트 완료")
    print("="*60)
    
    cube_pos = cube.data.root_pos_w[0]
    final_lifted = cube_pos[2].item() - initial_cube_z if initial_cube_z else 0
    
    print(f"\n최종 결과:")
    print(f"  초기 큐브 Z: {initial_cube_z:.3f}")
    print(f"  최종 큐브 Z: {cube_pos[2].item():.3f}")
    print(f"  들어올린 높이: {final_lifted*100:.1f}cm")
    
    if final_lifted > 0.03:
        print("\n>>> 성공: 큐브가 들어올려짐!")
    else:
        print("\n>>> 실패: 큐브를 잡지 못함")
    
    simulation_app.close()


if __name__ == "__main__":
    main()
