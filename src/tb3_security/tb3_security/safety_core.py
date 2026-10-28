"""Framework-agnostic safe-mode controller shared by security_node and web_sim.

Holds the LIDAR wall estimation and the safe-mode FSM
(NORMAL -> SAFE_ROTATE -> SAFE_FORWARD -> SAFE_ALIGN -> SAFE_STOP -> NORMAL)
with no ROS/rclpy dependency, mirroring what ``detection_core`` does for the
detection logic. Time is injected via ``now`` so the browser simulator can run
the exact same state machine on simulated time.

Goal of safe mode: pull over next to the nearest wall and stop *parallel* to
it, with the robot's left or right side facing the wall (like parking at a
curb), so a long vehicle does not stick out into the space.

  SAFE_ROTATE   stop and wait for fresh scans (PLAN_SCANS), pick the wall,
                then face it squarely (forward or reverse, whichever needs
                the smaller turn)
  SAFE_FORWARD  drive straight at the wall until the perpendicular distance
                reaches SAFE_WALL_DISTANCE
  SAFE_ALIGN    turn 90 deg so the wall is exactly on the left or right side
  SAFE_STOP     hold still; NORMAL after RECOVERY_TIME

The wall is a straight line fitted to the scan points around its normal, so
the estimate does not depend on fixed index ranges (works for 260- and
360-sample scans) and is re-tracked on every scan while the robot moves.
"""

import math

# 정상 모드 설정값
MAX_LINEAR = 0.22
MAX_ANGULAR = 2.84

# 안전 모드 설정값
SAFE_LINEAR = 0.08               # 벽 접근 속도
SAFE_LINEAR_SLOW = 0.04          # 목표 거리 10 cm 이내
SAFE_ANGULAR = 0.3               # 회전 최대 속도
SAFE_ANGULAR_MIN = 0.06          # 비례 제어 최소 속도 (마찰로 멈추지 않게)
ALIGN_ANGULAR = 0.2              # 정렬 회전 최대 속도
ANGLE_GAIN = 1.2                 # 각도 오차(rad) -> 회전 속도 비례 계수
SAFE_WALL_DISTANCE = 0.20        # 정렬 후 라이다 중심 ~ 벽 거리 (m)
WALL_STOP_DISTANCE = 0.15        # 진행 방향에 이보다 가까운 물체 -> 접근 중단
# 정지 명령 후 이만큼 새 스캔을 받고 벽을 고른다 (회전 중 찍힌 옛 스캔이면 방향이 틀어짐)
PLAN_SCANS = 2
ROTATE_TOLERANCE = math.radians(4)
ALIGN_TOLERANCE = math.radians(2)   # 벽과 평행 판정 (각도)
REVERSE_PENALTY = math.radians(15)  # 같은 조건이면 전진 접근 선호
WALL_FIT_WINDOW = math.radians(35)  # 벽 법선 주변 ±35° 점으로 직선 근사
WALL_FIT_MIN_POINTS = 6
TRACK_MAX_TURN = math.radians(12)   # 스캔 간 벽 방향 변화 허용치 (회전 0.3 rad/s ≈ 3.4°/스캔)
TRACK_MAX_SHIFT = 0.15              # 스캔 간 벽 거리 변화 허용치 (m)
FOOT_MARGIN = 0.04                  # 수선의 발 양쪽으로 최소 이만큼 벽면이 보여야 함
TRACK_GATE = 0.08                   # 추적 중 예측 직선에서 이 거리 안의 점만 벽으로 본다
WALL_LOST_SCANS = 3                 # 연속 이만큼 벽을 못 보면 (0.6 s) 벽을 다시 고른다
MAX_REPLANS = 2                     # 다시 고르기 최대 횟수, 넘으면 그 자리 정지
RANGE_MIN, RANGE_MAX = 0.12, 12.0

# 복귀 시간
RECOVERY_TIME = 10.0
# 각 단계 시간 제한 (못 끝내도 정지 → 복귀 타이머 시작)
SAFE_ROTATE_TIMEOUT = 15.0
SAFE_FORWARD_TIMEOUT = 20.0      # 최소값. 실제 제한은 벽 거리로 계산 (최대 60 s)
SAFE_ALIGN_TIMEOUT = 15.0

SECTORS = ('front', 'left', 'back', 'right')   # 90° 씩, front = ±45°


def clamp_cmd(linear_x, angular_z):
    """정상 모드 속도 제한 (security_node 가 로봇으로 보내기 전 적용)."""
    return (max(min(linear_x, MAX_LINEAR), -MAX_LINEAR),
            max(min(angular_z, MAX_ANGULAR), -MAX_ANGULAR))


def wrap(a):
    """각도를 [-pi, pi) 로."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def sector_of(bearing):
    """로봇 기준 방향(rad) -> 'front' / 'left' / 'back' / 'right'."""
    return SECTORS[int(((wrap(bearing) + math.pi / 4) % (2 * math.pi)) // (math.pi / 2))]


class SafetyController:
    """LIDAR 벽 추정 + 안전 모드 FSM. 노드는 타이머/발행만 담당한다."""

    def __init__(self, logger=None):
        self.logger = logger

        # FSM 상태
        self.mode = 'NORMAL'
        self.safe_stop_time = None

        # 라이다 데이터: 방향별 최소 거리 (90° 섹터, 모두 같은 기준)
        self.front_min = float('inf')
        self.back_min = float('inf')
        self.left_min = float('inf')
        self.right_min = float('inf')
        self.points = []               # [(bearing, range)] 유효한 점

        # 벽 정렬
        self.wall_direction = None     # 처음 고른 벽 방향 (표시용)
        self.wall_bearing = None       # 벽 법선 방향 (로봇 기준, rad) - 매 스캔 갱신
        self.wall_dist = float('inf')  # 벽까지 수직 거리
        self.wall_side = 0             # 정렬 후 벽이 올 쪽: +1 왼쪽, -1 오른쪽
        self.travel_dir = 1            # 접근 방향: +1 전진, -1 후진
        self._plan_why = None          # 벽 고르기 대기 중이면 그 이유 (정지 후 새 스캔 기다림)
        self._plan_scans = -1          # 정지 명령 이후 받은 스캔 수 (-1: 아직 정지 명령 전)
        self.rotate_start_time = None
        self.forward_start_time = None
        self.forward_timeout = SAFE_FORWARD_TIMEOUT
        self.align_start_time = None
        self._last_cmd, self._last_t = (0.0, 0.0), None   # 스캔 사이 벽 위치 예측용
        self._fresh = False            # 직전 action 이후 스캔으로 벽을 갱신했는지
        self._lost = 0                 # 연속으로 벽을 못 본 스캔 수
        self._replans = 0

        # 마지막 감지 이유
        self.last_detection_reason = ''

    def _log(self, level, msg):
        if self.logger is not None:
            getattr(self.logger, level, lambda *_: None)(msg)

    # ------------------------------------------------------------------ #
    # LIDAR
    # ------------------------------------------------------------------ #
    def update_scan(self, ranges, angle_min=0.0, angle_increment=None):
        """ranges[i] 의 방향 = angle_min + i * angle_increment (기본: 한 바퀴 균등)."""
        num = len(ranges)
        if num == 0:
            return
        if angle_increment is None:
            angle_increment = 2.0 * math.pi / num

        pts = []
        mins = dict.fromkeys(SECTORS, float('inf'))
        for i, r in enumerate(ranges):
            if not (RANGE_MIN < r < RANGE_MAX) or math.isnan(r):
                continue
            b = wrap(angle_min + i * angle_increment)
            pts.append((b, r))
            s = sector_of(b)
            if r < mins[s]:
                mins[s] = r
        if not pts:
            return
        self.points = pts
        if self._plan_why is not None and self._plan_scans >= 0:
            self._plan_scans += 1
        self.front_min, self.left_min = mins['front'], mins['left']
        self.back_min, self.right_min = mins['back'], mins['right']

        if self.mode != 'NORMAL' and self.wall_bearing is not None:
            fit = self._fit_wall(self.wall_bearing, self.wall_dist)
            # 한 스캔(0.2 s) 사이 벽이 크게 튀면 지나가던 장애물로 갈아탄 것 -> 무시
            if fit is not None and abs(wrap(fit[0] - self.wall_bearing)) <= TRACK_MAX_TURN \
                    and abs(fit[1] - self.wall_dist) <= TRACK_MAX_SHIFT:
                self.wall_bearing, self.wall_dist = fit
                self._fresh, self._lost = True, 0
            else:
                self._lost += 1

    def _fit_wall(self, center, dist=None):
        """center 방향 ±WALL_FIT_WINDOW 의 점들로 직선 근사 -> (법선 방향, 수직 거리).

        추적 중(dist 있음): 예측한 벽 직선(법선 center, 거리 dist) 근처 점만 쓴다.
        처음 고를 때: 창 안의 가장 가까운 점을 벽의 수선 발로 보고, 같은 직선
        위라면 나와야 할 거리 r0/cos(각도차) 에서 벗어난 점은 뺀다.
        (모서리 너머 다른 벽, 장애물이 섞이지 않게)
        """
        win = [(b, r) for b, r in self.points if abs(wrap(b - center)) <= WALL_FIT_WINDOW]
        if len(win) < WALL_FIT_MIN_POINTS:
            return None
        if dist is not None:
            sel = [(r * math.cos(b), r * math.sin(b)) for b, r in win
                   if abs(r * math.cos(wrap(b - center)) - dist) < TRACK_GATE]
        else:
            b0, r0 = min(win, key=lambda p: p[1])
            sel = [(r * math.cos(b), r * math.sin(b)) for b, r in win
                   if abs(r - r0 / max(math.cos(wrap(b - b0)), 0.2)) < 0.03 + 0.1 * r0]
        for _ in range(2):            # 2회차: 직선에서 5 cm 넘게 벗어난 점 제외
            if len(sel) < WALL_FIT_MIN_POINTS:
                return None
            n = len(sel)
            mx = sum(p[0] for p in sel) / n
            my = sum(p[1] for p in sel) / n
            sxx = sum((p[0] - mx) ** 2 for p in sel)
            syy = sum((p[1] - my) ** 2 for p in sel)
            sxy = sum((p[0] - mx) * (p[1] - my) for p in sel)
            normal = 0.5 * math.atan2(2 * sxy, sxx - syy) + math.pi / 2   # 최소 분산 방향
            nx, ny = math.cos(normal), math.sin(normal)
            dist = mx * nx + my * ny
            if dist < 0:                                  # 법선이 로봇 -> 벽 방향이 되게
                nx, ny, dist, normal = -nx, -ny, -dist, normal + math.pi
            sel = [p for p in sel if abs(p[0] * nx + p[1] * ny - dist) < 0.05]
        if len(sel) < WALL_FIT_MIN_POINTS:
            return None
        # 수선의 발이 실제 벽면 위에 있어야 함 (양쪽에 점). 볼록 모서리나 짧은 면의
        # 끝을 지나친 경우는 벽이 아님 -> None
        along = [-p[0] * ny + p[1] * nx for p in sel]
        if min(along) > -FOOT_MARGIN or max(along) < FOOT_MARGIN:
            return None
        return wrap(normal), dist

    def _acquire_wall(self):
        """가장 가까운 '벽면' 을 찾는다. 거리 극소점마다 직선 근사를 해 보고
        검증을 통과한 것 중 가장 가까운 것. 없으면 가장 가까운 점 그대로."""
        pts = sorted(self.points)
        n = len(pts)
        cands = [pts[i] for i in range(n)
                 if all(pts[i][1] <= pts[(i + k) % n][1] for k in (-3, -2, -1, 1, 2, 3))]
        best = None
        for b, r in sorted(cands, key=lambda p: p[1])[:8]:
            fit = self._fit_wall(b)
            if fit is not None and (best is None or fit[1] < best[1]):
                best = fit
        if best is not None:
            return best
        return min(self.points, key=lambda p: p[1])

    def _travel_clearance(self):
        """진행 방향 ±25° 안의 가장 가까운 물체 거리."""
        heading = 0.0 if self.travel_dir > 0 else math.pi
        near = [r for b, r in self.points if abs(wrap(b - heading)) <= math.radians(25)]
        return min(near) if near else float('inf')

    # ------------------------------------------------------------------ #
    # 안전 모드 FSM
    # ------------------------------------------------------------------ #
    def enter_safe_mode(self, reason, now):
        if self.mode != 'NORMAL':
            return
        self.last_detection_reason = reason
        self.safe_stop_time = None
        self._replans = 0
        self._request_plan(now, f'Entering SAFE mode - reason: {reason}')

    def _request_plan(self, now, why):
        """일단 멈추고, 멈춘 뒤 찍힌 스캔 PLAN_SCANS 개를 받은 다음 _plan.
        (공격으로 빠르게 돌던 중의 스캔은 최대 0.2 s 전 자세라 벽 방향이 틀어진다)"""
        self.mode = 'SAFE_ROTATE'
        self.rotate_start_time = now
        self.wall_bearing, self.wall_dist = None, float('inf')
        self._plan_why, self._plan_scans = why, -1

    def _plan(self, now, why):
        """가장 가까운 벽을 골라 접근 방법을 정하고 SAFE_ROTATE (또는 바로 ALIGN) 시작."""
        self._plan_why = None
        self._fresh, self._lost = False, 0
        if not self.points:
            self.wall_direction = None
            self._stop(now, 'no LIDAR data - stopping in place')
            return

        # 가장 가까운 벽면 (직선 근사)
        self.wall_bearing, self.wall_dist = self._acquire_wall()
        self.wall_direction = sector_of(self.wall_bearing)

        # 접근 방법 선택: 벽을 정면(전진) 또는 정후면(후진)으로 보게 회전 - 덜 도는 쪽.
        # 회전 후 법선 방향 = rotate_target (0 = 앞, pi = 뒤).
        fwd = abs(wrap(self.wall_bearing))
        rev = abs(wrap(self.wall_bearing - math.pi)) + REVERSE_PENALTY
        self.travel_dir, self.rotate_target = (1, 0.0) if fwd <= rev else (-1, math.pi)
        self._last_cmd, self._last_t = (0.0, 0.0), now
        # 벽 앞에서 90° 돌아 벽이 오게 될 쪽 (+1 왼쪽, -1 오른쪽). ALIGN 시작 때 확정.
        self.wall_side = 1 if self.travel_dir > 0 else -1

        if self.wall_dist <= SAFE_WALL_DISTANCE + 0.02:
            self._start_align(now, 'already at the wall')
        else:
            self.mode = 'SAFE_ROTATE'
            self.rotate_start_time = now

        self._log('warn',
                  f'[MODE] {why} - wall: {self.wall_direction} '
                  f'({self.wall_dist:.2f}m, {math.degrees(self.wall_bearing):.0f}deg) - '
                  f'{"forward" if self.travel_dir > 0 else "reverse"} straight approach')
        self._log('warn',
                  f'[SAFE] Front: {self.front_min:.2f}m, Back: {self.back_min:.2f}m, '
                  f'Left: {self.left_min:.2f}m, Right: {self.right_min:.2f}m')

    def _turn_toward(self, target, max_speed):
        """벽 법선이 target 방향이 되도록 회전 명령 (비례 제어). (angular, 오차)."""
        err = wrap(self.wall_bearing - target)      # 로봇이 err 만큼 돌면 맞음
        speed = min(max_speed, max(SAFE_ANGULAR_MIN, ANGLE_GAIN * abs(err)))
        return math.copysign(speed, err), err

    def _start_align(self, now, why):
        # 벽을 왼쪽/오른쪽 중 덜 도는 쪽에 둔다
        left = abs(wrap(self.wall_bearing - math.pi / 2))
        right = abs(wrap(self.wall_bearing + math.pi / 2))
        self.wall_side = 1 if left <= right else -1
        self.mode = 'SAFE_ALIGN'
        self.align_start_time = now
        self._log('info', f'[SAFE] {why} ({self.wall_dist:.2f}m) - aligning parallel')

    def _stop(self, now, why, level='info'):
        self.mode = 'SAFE_STOP'
        self.safe_stop_time = now
        self._log(level, f'[SAFE] {why}')

    def safe_mode_action(self, now):
        """10 Hz 타이머 본체. 안전 모드면 (linear, angular) 반환, NORMAL 이면 None."""
        if self.mode == 'NORMAL':
            return None

        # 예측: 지난 명령만큼 움직였다고 보고 벽 위치를 옮겨 둔다 (다음 스캔이 보정)
        # (스캔이 방금 갱신했으면 그 값이 최신이므로 건너뜀)
        if self._fresh:
            self._fresh = False
        elif self._last_t is not None and self.wall_bearing is not None:
            dt = now - self._last_t
            lin, ang = self._last_cmd
            self.wall_dist -= lin * math.cos(self.wall_bearing) * dt
            self.wall_bearing = wrap(self.wall_bearing - ang * dt)
        self._last_t = now

        linear, angular = 0.0, 0.0

        # 벽 고르기 대기: 정지 명령을 보내고, 그 뒤 새 스캔을 받으면 고른다
        if self._plan_why is not None:
            if self._plan_scans < 0:
                self._plan_scans = 0
            elif self._plan_scans >= PLAN_SCANS:
                self._plan(now, self._plan_why)
            self._last_cmd = (0.0, 0.0)
            return 0.0, 0.0

        # 벽을 놓침 (짧은 장애물 면을 지나침, 가려짐) -> 추측으로 움직이지 말고 다시 고른다
        if self.mode != 'SAFE_STOP' and self._lost >= WALL_LOST_SCANS:
            if self._replans < MAX_REPLANS:
                self._replans += 1
                self._request_plan(
                    now, f'Wall lost - re-selecting ({self._replans}/{MAX_REPLANS})')
            else:
                self._stop(now, 'Wall lost again - stopping in place', 'warn')
            self._last_cmd = (0.0, 0.0)
            return 0.0, 0.0

        if self.mode == 'SAFE_ROTATE':
            angular, err = self._turn_toward(self.rotate_target, SAFE_ANGULAR)
            if abs(err) <= ROTATE_TOLERANCE or now - self.rotate_start_time > SAFE_ROTATE_TIMEOUT:
                angular = 0.0
                self.mode = 'SAFE_FORWARD'
                self.forward_start_time = now
                path = max(self.wall_dist - SAFE_WALL_DISTANCE, 0.0)
                self.forward_timeout = min(max(SAFE_FORWARD_TIMEOUT,
                                               1.5 * path / SAFE_LINEAR_SLOW), 60.0)
                self._log('info', '[SAFE] Rotation done - approaching wall '
                                  f'({path:.2f}m, limit {self.forward_timeout:.0f}s)')

        elif self.mode == 'SAFE_FORWARD':
            remaining = self.wall_dist - SAFE_WALL_DISTANCE
            if remaining <= 0.0:
                self._start_align(now, 'Wall reached')
            elif self._travel_clearance() <= WALL_STOP_DISTANCE:
                # 다른 벽/장애물이 앞을 막음 -> 이제 그게 가장 가까운 벽이므로 그쪽에 붙인다
                if self._replans < MAX_REPLANS:
                    self._replans += 1
                    self._request_plan(now, 'Path blocked - parking along the obstacle ahead')
                else:
                    self._start_align(now, 'Path blocked')
            elif now - self.forward_start_time > self.forward_timeout:
                self._stop(now, f'Wall not reached in {self.forward_timeout:.0f}s '
                                '- stopping in place', 'warn')
            else:
                speed = SAFE_LINEAR if remaining > 0.10 else SAFE_LINEAR_SLOW
                linear = self.travel_dir * speed

        elif self.mode == 'SAFE_ALIGN':
            angular, err = self._turn_toward(self.wall_side * math.pi / 2, ALIGN_ANGULAR)
            if abs(err) <= ALIGN_TOLERANCE:
                angular = 0.0
                self._stop(now, f'Parallel to wall ({math.degrees(err):+.1f}deg, '
                                f'{self.wall_dist:.2f}m) - stopping')
            elif now - self.align_start_time > SAFE_ALIGN_TIMEOUT:
                angular = 0.0
                self._stop(now, f'Alignment not converged in {SAFE_ALIGN_TIMEOUT:.0f}s '
                                f'({math.degrees(err):+.1f}deg) - stopping as is', 'warn')

        self._last_cmd = (linear, angular)
        return linear, angular

    def check_recovery(self, now):
        """1 Hz 타이머 본체. NORMAL 로 복귀한 순간 True (노드가 이력 초기화)."""
        if self.mode == 'SAFE_STOP' and self.safe_stop_time is not None:
            remaining = RECOVERY_TIME - (now - self.safe_stop_time)
            if remaining > 0:
                self._log('info', f'[SAFE] Recovering in {remaining:.0f}s...')
            else:
                self.mode = 'NORMAL'
                self.safe_stop_time = None
                self.wall_direction = None
                self._plan_why = None
                self.wall_bearing = None
                self.wall_dist = float('inf')
                self._last_t = None
                self.rotate_start_time = None
                self.forward_start_time = None
                self.align_start_time = None
                self._log('info', '[MODE] Returning to NORMAL mode')
                return True
        return False

    def recovery_remaining(self, now):
        if self.mode == 'SAFE_STOP' and self.safe_stop_time is not None:
            return max(RECOVERY_TIME - (now - self.safe_stop_time), 0.0)
        return None
