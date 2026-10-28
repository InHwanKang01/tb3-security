import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
import math

class ScanTest(Node):
    def __init__(self):
        super().__init__('scan_test')
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.sub = self.create_subscription(
            LaserScan, '/scan', self.callback, qos)
        self.get_logger().info('Scan Test started')

    def callback(self, msg):
        ranges = msg.ranges
        num = len(ranges)
        self.get_logger().info(f'Total ranges: {num}')

        def get_valid(start_idx, end_idx):
            start_idx = start_idx % num
            end_idx = end_idx % num
            if start_idx > end_idx:
                vals = list(ranges[start_idx:]) + list(ranges[:end_idx])
            else:
                vals = list(ranges[start_idx:end_idx])
            valid = [r for r in vals if r > 0.12 and r < 12.0 and not math.isinf(r) and not math.isnan(r)]
            return valid

        front = get_valid(num - 14, 15)
        back = get_valid(116, 144)
        left = get_valid(36, 93)
        right = get_valid(166, 223)

        self.get_logger().info(f'Front valid: {len(front)}, Back valid: {len(back)}, Left valid: {len(left)}, Right valid: {len(right)}')

        if front:
            self.get_logger().info(f'Front min: {min(front):.2f}m')
        else:
            self.get_logger().info('Front: no valid data')

        if back:
            self.get_logger().info(f'Back min: {min(back):.2f}m')
        else:
            self.get_logger().info('Back: no valid data')

        if left:
            self.get_logger().info(f'Left avg: {sum(left)/len(left):.2f}m')
        else:
            self.get_logger().info('Left: no valid data')

        if right:
            self.get_logger().info(f'Right avg: {sum(right)/len(right):.2f}m')
        else:
            self.get_logger().info('Right: no valid data')

        # 가장 가까운 방향
        distances = {}
        if front:
            distances['front'] = min(front)
        if back:
            distances['back'] = min(back)
        if left:
            distances['left'] = sum(left)/len(left)
        if right:
            distances['right'] = sum(right)/len(right)

        if distances:
            closest = min(distances, key=distances.get)
            self.get_logger().info(f'Closest wall: {closest} ({distances[closest]:.2f}m)')

        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = ScanTest()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
