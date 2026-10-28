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

## 유의사항

이 코드는 아래 환경에 맞춰 작성하고 시험했다. 로봇의 종류와 설정, 네트워크 연결 설정 등이 다르면
세부 코드와 설정값을 환경에 맞게 바꿔야 할 수 있다.

- **로봇**: TurtleBot3 Burger 기준이다. 탐지 기준 중 "로봇이 낼 수 없는 속도"(선속도 0.3 m/s, 각속도 3.0 rad/s)는
  Burger의 최대 속도(0.22 m/s, 2.84 rad/s)에 맞춘 값이다. 다른 로봇이나 다른 조종 프로그램
  (예: 기본 0.5 m/s인 `teleop_twist_keyboard`)을 쓰면 정상 조종도 공격으로 판단할 수 있다.
  → `detection_core.py`의 기준값 수정
- **학습 모델**: Isolation Forest는 이 로봇의 정상 운전 기록으로 학습했다. 로봇이나 운전 방식이 바뀌면
  `data_collector_node`로 정상 운전을 다시 기록하고 `train_model`로 다시 학습해야 한다.
- **라이다와 안전모드**: LDS-01(360개)과 LDS-02(약 260개) 라이다를 모두 고려했다.
  벽과의 거리(0.20 m)와 회전 동작은 Burger 크기 기준이므로, 로봇 크기가 다르면 `safety_core.py`의 값을 바꿔야 한다.
- **네트워크**: 메시지 폭주(초당 50개)와 명령 지연(1초) 기준은 이 실험 환경의 네트워크에서 정한 값이다.
  Wi-Fi 상태, 원격 PC와 로봇의 연결 방식, `ROS_DOMAIN_ID`, DDS(RMW) 설정, QoS에 따라
  평소 지연과 전송 속도가 달라지므로 기준값을 다시 확인해야 한다.
  지연은 `stamp_node`가 붙인 시각과 `security_node`가 받은 시각의 차이로 재므로,
  두 노드를 다른 컴퓨터에서 실행하면 두 컴퓨터의 시계를 맞춰야 한다(NTP 등).
- **ROS 버전**: ROS 2 Humble에서 시험했다. 다른 배포판에서는 패키지나 API 차이로 수정이 필요할 수 있다.
