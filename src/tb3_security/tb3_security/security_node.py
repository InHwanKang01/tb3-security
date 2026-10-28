import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import time

from tb3_security.detection_core import (
    DetectionEngine,
    DEFAULT_MODEL_PATH,
    PUBLISH_RATE_WINDOW,
)
from tb3_security.safety_core import SafetyController, clamp_cmd

# NOTE: 탐지 임계값/게이트 로직은 detection_core.py, 라이다 섹터/안전 모드 FSM 과
# 속도 제한 상수는 safety_core.py 에 있음. security_node 와 오프라인 평가기
# (evaluate.py), 웹 시뮬레이터(web_sim.py)가 같은 코어를 공유한다.


class SecurityNode(Node):
    def __init__(self):
        super().__init__('security_node')

        # ROS 파라미터
        self.declare_parameter('model_path', DEFAULT_MODEL_PATH)
        model_path = self.get_parameter('model_path').get_parameter_value().string_value

        # 메시지 수신 기록 및 현재 feature 캐시
        self.msg_timestamps = []
        self.current_publish_rate = 0.0
        self.current_delay = 0.0
        self.current_linear = 0.0
        self.current_angular = 0.0

        # 라이다 섹터 + 안전 모드 FSM — security_node 와 웹 시뮬레이터가 공유하는 코어
        self.safety = SafetyController(logger=self.get_logger())

        # 탐지 엔진 (ML 게이트 + 규칙) — security_node 와 평가기가 공유하는 코어
        self.engine = DetectionEngine(model_path=model_path, logger=self.get_logger())

        # subscribe
        self.stamped_sub = self.create_subscription(
            TwistStamped, '/cmd_vel_stamped', self.stamped_callback, 10)
        self.delayed_sub = self.create_subscription(
            TwistStamped, '/cmd_vel_delayed', self.stamped_callback, 10)

        scan_qos = QoSProfile(depth=10)
        scan_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, scan_qos)

        # publish
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/security_status', 10)

        # 타이머
        self.safe_timer = self.create_timer(0.1, self.safe_mode_action)
        self.recovery_timer = self.create_timer(1.0, self.check_recovery)
        self.status_timer = self.create_timer(0.5, self.publish_status)

        self.get_logger().info('Security Node started - NORMAL mode')

    # ------------------------------------------------------------------ #
    # LIDAR
    # ------------------------------------------------------------------ #
    def scan_callback(self, msg):
        self.safety.update_scan(msg.ranges, msg.angle_min, msg.angle_increment)

    # ------------------------------------------------------------------ #
    # Feature extraction (timing lives in the node; pure detection in engine)
    # ------------------------------------------------------------------ #
    def update_publish_rate(self):
        now = time.time()
        self.msg_timestamps.append(now)
        self.msg_timestamps = [t for t in self.msg_timestamps if now - t < PUBLISH_RATE_WINDOW]
        self.current_publish_rate = len(self.msg_timestamps) / PUBLISH_RATE_WINDOW
        return self.current_publish_rate

    def calculate_command_delay(self, stamped_msg):
        sent = stamped_msg.header.stamp.sec + stamped_msg.header.stamp.nanosec * 1e-9
        now = self.get_clock().now().nanoseconds * 1e-9
        delay = now - sent
        self.current_delay = max(delay, 0.0)
        return self.current_delay

    def extract_features(self, msg, rate):
        """Return (linear_x, angular_z, publish_rate, delay) from a TwistStamped."""
        delay = self.calculate_command_delay(msg)
        self.current_linear = msg.twist.linear.x
        self.current_angular = msg.twist.angular.z
        return (self.current_linear, self.current_angular, rate, delay)

    # ------------------------------------------------------------------ #
    # 안전 모드 FSM
    # ------------------------------------------------------------------ #
    def enter_safe_mode(self, reason):
        self.safety.enter_safe_mode(reason, time.time())

    def safe_mode_action(self):
        action = self.safety.safe_mode_action(time.time())
        if action is None:
            return
        safe_msg = Twist()
        safe_msg.linear.x, safe_msg.angular.z = action
        self.cmd_pub.publish(safe_msg)

    def check_recovery(self):
        if self.safety.check_recovery(time.time()):
            # msg_timestamps 는 지우지 않음: 안전 모드 중에도 계속 세고 있으므로 복귀 직후
            # publish_rate 가 실제 값이다 (지우면 rate≈1 로 시작해 ML 오탐 → 안전 모드 재진입)
            self.engine.reset_history()

    def publish_status(self):
        sc = self.safety
        status = String()
        status.data = (
            f'mode:{sc.mode}|'
            f'front:{sc.front_min:.2f}|'
            f'back:{sc.back_min:.2f}|'
            f'left:{sc.left_min:.2f}|'
            f'right:{sc.right_min:.2f}|'
            f'wall:{sc.wall_direction}|'
            f'detection:{sc.last_detection_reason}|'
            f'msg_rate:{len(self.msg_timestamps)}|'
            f'ml_anomaly:{int(self.engine.last_ml_anomaly)}|'
            f'ml_score:{self.engine.last_ml_score:.4f}|'
            f'linear_x:{self.current_linear:.3f}|'
            f'angular_z:{self.current_angular:.3f}|'
            f'publish_rate:{self.current_publish_rate:.2f}|'
            f'delay:{self.current_delay:.3f}|'
            f'attack_type:{self.engine.last_attack_type}'
        )
        self.status_pub.publish(status)

    # ------------------------------------------------------------------ #
    # Main callback
    # ------------------------------------------------------------------ #
    def stamped_callback(self, msg):
        # 도착률은 모드와 무관하게 항상 측정 (복귀 직후에도 publish_rate 가 정확하도록)
        rate = self.update_publish_rate()
        if self.safety.mode != 'NORMAL':
            return

        features = self.extract_features(msg, rate)

        # 하이브리드 게이트: 하드 한계(flooding/delay/불가능한 속도) OR (ML AND (통계 OR 속도 임계값)). 코어 엔진이 담당.
        result = self.engine.step(features, time.time())

        if result['final_anomaly']:
            self.enter_safe_mode(result['reason'])
            return

        # 정상 → 클램프 후 전송
        out = Twist()
        out.linear.x, out.angular.z = clamp_cmd(msg.twist.linear.x, msg.twist.angular.z)
        self.cmd_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = SecurityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
