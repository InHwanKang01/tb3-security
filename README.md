# ROS 기반 로봇의 네트워크 이상 감지 및 안전 주행 시스템

충북대학교 정보통신공학부 졸업작품 (ROS 2 Humble, TurtleBot3 Burger)

로봇에 들어오는 속도 명령(`cmd_vel`)을 검사해 속도 위조, 메시지 폭주, 명령 지연 공격을 탐지하고,
공격이면 로봇이 스스로 가장 가까운 벽 옆에 나란히 멈추는(안전모드) 시스템이다.

## 구조

```
조종기/공격자 ─► /cmd_vel_input ─► stamp_node ─► /cmd_vel_stamped ─► security_node ─► /cmd_vel ─► 로봇
                                                  (delay_attack_node ─► /cmd_vel_delayed ─┘)
라이다 /scan ─► security_node          security_node ─► /security_status ─► dashboard
```

- `detection_core.py` 탐지 엔진: 하드 한계(폭주·지연·불가능한 속도)는 즉시 탐지,
  애매한 속도 이상은 Isolation Forest(정상 운전 학습)가 확인해야 탐지
- `safety_core.py` 안전모드: 라이다로 벽을 찾아 다가간 뒤 90° 돌아 옆면이 벽을 보게 멈춤
- `security_node.py` 위 두 모듈을 ROS 노드로 연결
- `dashboard.py` 실시간 대시보드 (Flask, http://localhost:5000)
- `web_sim.py` 로봇 없이 브라우저에서 공격·방어 시연 (http://127.0.0.1:5050)
- `evaluate.py`, `stealth_eval.py` 오프라인 성능 평가 (결과: `eval_output/`)

## 실행

```bash
colcon build --packages-select tb3_security
source install/setup.bash
ros2 run tb3_security stamp_node
ros2 run tb3_security security_node
ros2 run tb3_security dashboard
ros2 run tb3_security attack_manager_node   # 공격 자동 실행 (config/attack_scenario.yaml)

# 로봇 없이 시뮬레이터
ros2 run tb3_security web_sim
```

## 결과

| 항목 | 결과 |
| --- | --- |
| 속도 위조 / 메시지 폭주 / 명령 지연 탐지율 | 100% / 100% / 100% |
| 정상 운전을 공격으로 잘못 본 비율 | 0.4% |
| 300번 중 벽과 나란히 멈춘 횟수 | 291번 |

한계: 로봇이 평소 내는 속도 범위 안에서 보낸 가짜 명령은 정상 조종과 구분하지 못한다(메시지 인증 필요).
