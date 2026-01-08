# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""E0509 Cube Stack 환경 패키지"""

import gymnasium as gym

# 1. 내 환경과 설정을 가져옵니다.
from .e0509_cube_stack_env import E0509CubeStackEnv, E0509CubeStackEnvCfg
from .agents.rsl_rl_ppo_cfg import E0509CubeStackPPORunnerCfg

gym.register(
    id="Isaac-E0509-CubeStack-Direct-v0",
    
    # [수정됨] 기본 엔진(DirectRLEnv)이 아니라, 내가 만든 'E0509CubeStackEnv'를 실행합니다.
    entry_point=E0509CubeStackEnv,
    
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": E0509CubeStackEnvCfg,
        "rsl_rl_cfg_entry_point": E0509CubeStackPPORunnerCfg,
    },
)

print("[E0509CubeStack] Registered: Isaac-E0509-CubeStack-Direct-v0")