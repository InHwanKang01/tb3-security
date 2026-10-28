import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped

class StampNode(Node):
    def __init__(self):
        super().__init__('stamp_node')

        self.sub = self.create_subscription(
            Twist,
            '/cmd_vel_input',
            self.callback,
            10
        )

        self.pub = self.create_publisher(
            TwistStamped,
            '/cmd_vel_stamped',
            10
        )

        self.get_logger().info('Stamp Node started')

    def callback(self, msg):
        stamped = TwistStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.twist = msg
        self.pub.publish(stamped)

def main(args=None):
    rclpy.init(args=args)
    node = StampNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
