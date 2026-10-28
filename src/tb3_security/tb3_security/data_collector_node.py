"""Collect normal driving data (TwistStamped) to a CSV for Isolation Forest training.

CSV columns: timestamp, linear_x, angular_z, publish_rate, delay

Subscribes to /cmd_vel_stamped so it captures the same stream security_node sees.
Run only while driving normally (teleop, planned paths, …). Attacker nodes
should NOT be running during collection.
"""
import csv
import os
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped


PUBLISH_RATE_WINDOW = 1.0
DEFAULT_CSV_PATH = os.path.expanduser(
    '~/graduation_ws/src/tb3_security/data/normal_driving.csv')


class DataCollectorNode(Node):
    def __init__(self):
        super().__init__('data_collector_node')

        self.declare_parameter('csv_path', DEFAULT_CSV_PATH)
        csv_path = self.get_parameter('csv_path').get_parameter_value().string_value

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        is_new = not os.path.exists(csv_path)
        self.csv_path = csv_path
        self.csv_file = open(csv_path, 'a', newline='')
        self.writer = csv.writer(self.csv_file)
        if is_new:
            self.writer.writerow(
                ['timestamp', 'linear_x', 'angular_z', 'publish_rate', 'delay'])
            self.csv_file.flush()

        self.msg_timestamps = []
        self.sample_count = 0

        self.sub = self.create_subscription(
            TwistStamped, '/cmd_vel_stamped', self.stamped_callback, 10)
        self.log_timer = self.create_timer(5.0, self._log_progress)

        self.get_logger().info(f'Data collector started → {csv_path}')

    def update_publish_rate(self):
        now = time.time()
        self.msg_timestamps.append(now)
        self.msg_timestamps = [t for t in self.msg_timestamps if now - t < PUBLISH_RATE_WINDOW]
        return len(self.msg_timestamps) / PUBLISH_RATE_WINDOW

    def calculate_command_delay(self, msg):
        sent = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        now = self.get_clock().now().nanoseconds * 1e-9
        return max(now - sent, 0.0)

    def stamped_callback(self, msg):
        ts = time.time()
        linear_x = msg.twist.linear.x
        angular_z = msg.twist.angular.z
        rate = self.update_publish_rate()
        delay = self.calculate_command_delay(msg)

        self.writer.writerow([
            f'{ts:.6f}',
            f'{linear_x:.6f}',
            f'{angular_z:.6f}',
            f'{rate:.3f}',
            f'{delay:.6f}',
        ])
        self.csv_file.flush()
        self.sample_count += 1

    def _log_progress(self):
        self.get_logger().info(
            f'Samples collected: {self.sample_count} → {self.csv_path}')

    def destroy_node(self):
        try:
            self.csv_file.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
