"""Automated attack scenario manager.

Reads `attack_scenario.yaml`, then every `interval` seconds picks a random
attack type and turns it ON for `duration` seconds via `/attack_control`.
Also publishes a human-readable log line to `/attack_log` (for dashboard).

YAML structure:
    interval: 30
    duration: 5
    attacks: [flooding, spoofing, delay]
"""
import os
import random
import time

import yaml

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


DEFAULT_YAML = os.path.expanduser(
    '~/graduation_ws/src/tb3_security/config/attack_scenario.yaml')

# YAML이 없을 때 사용할 기본값
DEFAULT_INTERVAL = 30.0
DEFAULT_DURATION = 5.0
DEFAULT_ATTACKS = ['flooding', 'spoofing', 'delay']


class AttackManagerNode(Node):
    def __init__(self):
        super().__init__('attack_manager_node')

        self.declare_parameter('config_path', DEFAULT_YAML)
        cfg_path = self.get_parameter('config_path').get_parameter_value().string_value

        cfg = load_scenario(cfg_path)
        self.interval = float(cfg.get('interval', DEFAULT_INTERVAL))
        self.duration = float(cfg.get('duration', DEFAULT_DURATION))
        self.attacks = list(cfg.get('attacks', DEFAULT_ATTACKS))
        if self.duration >= self.interval:
            self.get_logger().warn(
                f'duration ({self.duration}s) >= interval ({self.interval}s); '
                f'attacks will overlap or chain back-to-back.')
        if not self.attacks:
            raise RuntimeError('attack_scenario.yaml: "attacks" list is empty')

        self.ctrl_pub = self.create_publisher(String, '/attack_control', 10)
        self.log_pub = self.create_publisher(String, '/attack_log', 10)

        self.current_attack = None
        self.attack_start_time = None
        self.next_trigger_time = time.time() + self.interval

        # 작은 주기로 깨어나 ON/OFF 시점을 체크 (블로킹 sleep 회피)
        self.tick_timer = self.create_timer(0.2, self.tick)

        self.get_logger().info(
            f'Attack Manager started - interval={self.interval}s, '
            f'duration={self.duration}s, attacks={self.attacks}')
        self._publish_log(
            f'manager_started: interval={self.interval}s duration={self.duration}s '
            f'attacks={self.attacks}')

    def tick(self):
        now = time.time()

        # 진행 중인 공격이 있으면 종료 시점 체크
        if self.current_attack is not None:
            if now - self.attack_start_time >= self.duration:
                self._stop_attack()
            return

        # 다음 트리거 시점이면 새 공격 발사
        if now >= self.next_trigger_time:
            self._start_random_attack()

    def _start_random_attack(self):
        attack = random.choice(self.attacks)
        self.current_attack = attack
        self.attack_start_time = time.time()

        self._publish_control(attack, 'ON')
        self.get_logger().warn(f'[SCENARIO] Triggered {attack} (duration={self.duration}s)')
        self._publish_log(f'attack_on:{attack} duration={self.duration}s')

    def _stop_attack(self):
        attack = self.current_attack
        self._publish_control(attack, 'OFF')
        self.get_logger().info(f'[SCENARIO] Stopped {attack}')
        self._publish_log(f'attack_off:{attack}')

        self.current_attack = None
        self.attack_start_time = None
        self.next_trigger_time = time.time() + self.interval

    def _publish_control(self, attack_type, state):
        msg = String()
        msg.data = f'{attack_type}: {state}'
        self.ctrl_pub.publish(msg)

    def _publish_log(self, text):
        msg = String()
        msg.data = f'{time.time():.3f}|{text}'
        self.log_pub.publish(msg)


def load_scenario(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            return {}
        return data
    except Exception as e:
        # 임시로 빈 dict 반환; 노드 초기화에서 기본값 사용
        print(f'[attack_manager] failed to load {path}: {e}')
        return {}


def main(args=None):
    rclpy.init(args=args)
    node = AttackManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    # 종료 시 진행 중 공격은 OFF로 정리
    if node.current_attack is not None:
        node._publish_control(node.current_attack, 'OFF')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
