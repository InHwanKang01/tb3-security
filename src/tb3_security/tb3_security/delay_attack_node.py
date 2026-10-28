import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import String


ATTACK_TYPE = 'delay'


class DelayAttackNode(Node):
    def __init__(self):
        super().__init__('delay_attack_node')

        self.active = False
        self.delay_seconds = 1.5

        self.sub = self.create_subscription(
            TwistStamped, '/cmd_vel_stamped', self.delay_callback, 10)
        self.pub = self.create_publisher(TwistStamped, '/cmd_vel_delayed', 10)
        self.ctrl_sub = self.create_subscription(
            String, '/attack_control', self.control_callback, 10)

        self.get_logger().info(
            f'Delay Attack Node ready - idle until '
            f'"{ATTACK_TYPE}: ON" received on /attack_control ({self.delay_seconds}s delay)')

    def control_callback(self, msg):
        cmd = parse_attack_control(msg.data)
        if cmd is None:
            return
        target, state = cmd
        if target != ATTACK_TYPE:
            return
        if state == 'ON' and not self.active:
            self.active = True
            self.get_logger().warn(f'DELAY attack: ON ({self.delay_seconds}s)')
        elif state == 'OFF' and self.active:
            self.active = False
            self.get_logger().info('DELAY attack: OFF')

    def delay_callback(self, msg):
        if not self.active:
            return
        # 동기 sleep — 공격 의도이므로 콜백 자체를 막아 지연 시뮬레이션
        time.sleep(self.delay_seconds)
        self.pub.publish(msg)
        self.get_logger().info(
            f'DELAYED: linear.x={msg.twist.linear.x:.2f} (delayed {self.delay_seconds}s)')


def parse_attack_control(data):
    if ':' not in data:
        return None
    target, _, state = data.partition(':')
    target = target.strip().lower()
    state = state.strip().upper()
    if state not in ('ON', 'OFF'):
        return None
    return target, state


def main(args=None):
    rclpy.init(args=args)
    node = DelayAttackNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
