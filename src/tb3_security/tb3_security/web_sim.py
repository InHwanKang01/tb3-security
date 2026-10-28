"""Browser-based TurtleBot3 security simulator -- no robot, no ROS, no Gazebo.

A 2D world (room + obstacles, diff-drive robot, 260-ray LIDAR) is simulated
on *simulated time*, and the defense layer runs on it through the very same
cores the real robot uses:

  * ``detection_core.DetectionEngine``  -- hybrid gate + trained IsolationForest
  * ``safety_core.SafetyController``    -- LIDAR sectoring + safe-mode FSM

The ROS plumbing around them is reproduced as a discrete-event model so the
numbers mean the same thing as on the robot:

  teleop / attackers -> stamp_node -> security_node  (keep_last(10) queues,
  single-threaded executor; each callback occupies the node for the *measured*
  wall time of the real engine.step x CPU factor) -> /cmd_vel -> robot.
  delay_attack_node holds each message 1.5 s (sync sleep) like the real node.

Run:
    ros2 run tb3_security web_sim            # http://127.0.0.1:5050
    PYTHONPATH=. python3 -m tb3_security.web_sim --port 5050
"""

import argparse
import collections
import heapq
import math
import os
import random
import threading
import time

from flask import Flask, Response, jsonify, request

from tb3_security.detection_core import (
    DEFAULT_MODEL_PATH, DELAY_THRESHOLD, DetectionEngine, FLOOD_THRESHOLD,
    PUBLISH_RATE_WINDOW, VELOCITY_THRESHOLD,
)
from tb3_security.safety_core import (
    MAX_ANGULAR, MAX_LINEAR, SafetyController, clamp_cmd,
)

# --------------------------------------------------------------------------- #
# World / robot / sensor model
# --------------------------------------------------------------------------- #
ROOM_W, ROOM_H = 4.0, 3.0
OBSTACLES = [  # axis-aligned boxes (x0, y0, x1, y1)
    (1.5, 1.1, 2.0, 1.7),
    (2.9, 0.0, 3.1, 0.9),
    (0.0, 2.2, 0.7, 2.4),
]
ROBOT_RADIUS = 0.105          # TurtleBot3 burger footprint
START_POSE = (0.7, 0.7, 0.0)

# safety_core works on ray angles (not fixed indices), so any sample count
# works; 260 mimics the LDS-02.
NUM_RAYS = 260
RANGE_MIN, RANGE_MAX = 0.12, 3.5
SCAN_PERIOD = 0.2             # 5 Hz LDS
PHYSICS_DT = 0.02

# ROS plumbing model
QUEUE_DEPTH = 10              # create_subscription(..., 10) -> keep_last(10)
TELEOP_PERIOD = 0.1           # teleop / driver publish rate (10 Hz)
SPOOF_PERIOD = 0.5            # abnormal_velocity_node timer
FLOOD_PERIOD = 0.002          # flood_attack_node timer
DELAY_SECONDS = 1.5           # delay_attack_node sleep
SAFE_PERIOD, RECOVERY_PERIOD = 0.1, 1.0
SAMPLE_PERIOD = 0.1           # history / chart sampling
HISTORY_WINDOW = 30.0
ATTACKS = ('spoofing', 'flooding', 'delay')
EPISODE_GRACE = 2.0           # delayed msgs keep arriving ~1.5 s after delay OFF


def _segments():
    segs = [((0, 0), (ROOM_W, 0)), ((ROOM_W, 0), (ROOM_W, ROOM_H)),
            ((ROOM_W, ROOM_H), (0, ROOM_H)), ((0, ROOM_H), (0, 0))]
    for x0, y0, x1, y1 in OBSTACLES:
        segs += [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                 ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
    return segs


SEGMENTS = _segments()


def _ray_hit(ox, oy, dx, dy):
    best = RANGE_MAX + 1.0
    for (ax, ay), (bx, by) in SEGMENTS:
        ex, ey = bx - ax, by - ay
        den = dx * ey - dy * ex
        if abs(den) < 1e-12:
            continue
        t = ((ax - ox) * ey - (ay - oy) * ex) / den
        u = ((ax - ox) * dy - (ay - oy) * dx) / den
        if t > 0 and 0 <= u <= 1 and t < best:
            best = t
    return best


def _dist_to_walls(x, y):
    best = float('inf')
    for (ax, ay), (bx, by) in SEGMENTS:
        ex, ey = bx - ax, by - ay
        L = ex * ex + ey * ey
        u = max(0.0, min(1.0, ((x - ax) * ex + (y - ay) * ey) / L))
        best = min(best, math.hypot(x - (ax + u * ex), y - (ay + u * ey)))
    return best


def _scan(x, y, yaw, rng):
    out = []
    for i in range(NUM_RAYS):
        a = yaw + 2 * math.pi * i / NUM_RAYS
        r = _ray_hit(x, y, math.cos(a), math.sin(a))
        if r > RANGE_MAX:
            out.append(float('inf'))
        else:
            out.append(max(r + rng.gauss(0, 0.005), 0.0))
    return out


def _model_threshold(engine):
    try:
        return float(engine.model.named_steps['iforest'].offset_)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
class Simulation:
    def __init__(self, engine, scenario):
        self.engine = engine
        self.scenario = scenario
        self.lock = threading.RLock()
        self.ml_threshold = _model_threshold(engine)
        self.reset()

    # ----- logging (SafetyController logger interface) ------------------ #
    def info(self, msg):
        if not msg.startswith('[SAFE] Recovering'):
            self._event('fsm', msg)

    warn = info

    def error(self, msg):
        self._event('error', msg)

    def _event(self, kind, text):
        self.events.append({'t': round(self.t, 2), 'kind': kind, 'text': text})

    # ----- setup --------------------------------------------------------- #
    def reset(self):
        with self.lock:
            self.rng = random.Random(7)
            self.t = 0.0
            self.heap = []
            self.seq = 0
            self.paused = False
            self.events = collections.deque(maxlen=200)
            self.history = collections.deque()
            self.episodes = []
            self.false_alarms = 0
            self.vetoed = 0
            self.collisions = 0
            self.colliding = False
            self.trail = collections.deque(maxlen=600)
            self.counters = collections.Counter()

            # robot
            self.x, self.y, self.yaw = START_POSE
            self.cmd = (0.0, 0.0)
            self.driver_cmd = (0.0, 0.0)          # last command the driver sent
            self.ranges = _scan(self.x, self.y, self.yaw, self.rng)

            # driver (teleop stand-in)
            self.driver_mode = 'auto'
            self.manual = [0.0, 0.0]
            self.auto_state, self.auto_turn_dir, self.auto_until = 'cruise', 1.0, 0.0
            self.auto_wander, self.sharp_until = 0.0, -1.0
            self.auto_lin, self.auto_turn_rate = 0.15, 1.5

            # attackers / scenario
            self.attack_until = {a: -1.0 for a in ATTACKS}
            self.auto_scenario = False
            self.next_scenario = 0.0
            self.defense = True
            self.cpu_factor = 1.0

            # security_node (mirrors SecurityNode state)
            self.engine.reset_history()
            self.engine.last_ml_score = 0.0
            self.engine.last_ml_anomaly = False
            self.safety = SafetyController(logger=self)
            self.msg_timestamps = []
            self.current = {'linear_x': 0.0, 'angular_z': 0.0,
                            'publish_rate': 0.0, 'delay': 0.0}
            self.last_result = None
            self.last_step_ms = 0.0
            self.q_stamped = collections.deque()
            self.q_delayed = collections.deque()
            self.q_scan = collections.deque()
            self.node_busy_until = 0.0
            self.node_round = []
            self.next_safe, self.next_recovery = SAFE_PERIOD, RECOVERY_PERIOD

            # delay_attack_node
            self.dq = collections.deque()
            self.delay_busy_until = 0.0

            for kind, t0 in [('teleop', 0.0), ('physics', 0.0), ('scan', 0.0),
                             ('node_timer', SAFE_PERIOD), ('sample', 0.0),
                             ('spoof', 0.0), ('flood', 0.0), ('scenario', 0.0)]:
                self.schedule(t0, kind)
            self.wall0, self.sim0 = time.monotonic(), 0.0
            self._event('info', 'simulation reset')

    def schedule(self, t, kind, data=None):
        self.seq += 1
        heapq.heappush(self.heap, (t, self.seq, kind, data))

    # ----- controls ------------------------------------------------------ #
    def start_attack(self, attack, duration):
        if attack not in ATTACKS:
            return
        if self.attack_until[attack] < self.t:
            self.episodes.append({'attack': attack, 'start': round(self.t, 2),
                                  'end': self.t + duration, 'latency': None,
                                  'detected_as': None, 'status': 'active',
                                  'saw_normal': self.safety.mode == 'NORMAL'})
            self._event('attack', f'{attack}: ON ({duration:.0f}s)')
        self.attack_until[attack] = self.t + duration
        for ep in self.episodes:
            if ep['attack'] == attack and ep['status'] == 'active':
                ep['end'] = self.attack_until[attack]

    def stop_attack(self, attack):
        if attack in ATTACKS and self.attack_until[attack] >= self.t:
            self.attack_until[attack] = self.t - 1e-9
            for ep in self.episodes:
                if ep['attack'] == attack and ep['status'] == 'active':
                    ep['end'] = self.t

    def _update_episodes(self):
        for a in ATTACKS:
            if 0 <= self.attack_until[a] < self.t and self.attack_until[a] > self.t - 0.1:
                self._event('attack', f'{a}: OFF')
        for ep in self.episodes:
            if ep['status'] != 'active':
                continue
            if self.safety.mode == 'NORMAL' and self.t <= ep['end']:
                ep['saw_normal'] = True   # the detector saw the attack while armed
            if self.t > ep['end'] + EPISODE_GRACE:
                # never reached the detector while NORMAL -> already blocked by safe mode
                ep['status'] = 'missed' if ep['saw_normal'] else 'blocked'

    def active(self, attack):
        return self.attack_until[attack] >= self.t

    # ----- driver -------------------------------------------------------- #
    def _driver_cmd(self):
        if self.t < self.sharp_until:
            return 0.0, 2.5                       # legit sharp turn (in training data)
        if self.driver_mode == 'manual':
            return tuple(self.manual)
        if self.driver_mode == 'stop':
            return 0.0, 0.0
        r = self.ranges

        def sector(lo, hi):
            vals = [r[i % NUM_RAYS] for i in range(lo, hi) if r[i % NUM_RAYS] > RANGE_MIN]
            return vals or [RANGE_MAX]
        front = min(sector(-18, 19))
        left = sum(sector(40, 90)) / len(sector(40, 90))
        right = sum(sector(170, 220)) / len(sector(170, 220))
        # Randomised within the normal-driving training range (|lin| <= 0.22,
        # |ang| <= 2.84) so patrol itself should not look like an attack.
        if self.auto_state == 'cruise':
            if front < 0.3 + self.auto_lin or self.colliding:  # faster -> brake earlier
                self.auto_state = 'turn'
                self.auto_turn_dir = 1.0 if left >= right else -1.0
                self.auto_turn_rate = self.rng.uniform(1.0, 2.0)
                self.auto_until = self.t + 6.0 / self.auto_turn_rate  # ~6 rad max
                return 0.0, self.auto_turn_rate * self.auto_turn_dir
            if self.t >= self.auto_until:
                if self.rng.random() < 0.1:               # brief pause
                    self.auto_lin, self.auto_wander = 0.0, 0.0
                    self.auto_until = self.t + self.rng.uniform(1.0, 2.0)
                else:
                    self.auto_lin = round(self.rng.uniform(0.06, 0.20), 2)
                    self.auto_wander = round(self.rng.uniform(-0.6, 0.6), 1)
                    self.auto_until = self.t + self.rng.uniform(2.0, 5.0)
            return self.auto_lin, self.auto_wander
        if front > 0.9 or self.t >= self.auto_until:
            self.auto_state, self.auto_until = 'cruise', self.t
            return 0.0, 0.0
        return 0.0, self.auto_turn_rate * self.auto_turn_dir

    # ----- message plumbing --------------------------------------------- #
    def publish_input(self, lin, ang, src):
        """/cmd_vel_input -> stamp_node -> /cmd_vel_stamped (to node + delay node)."""
        self.counters[f'pub_{src}'] += 1
        stamp = self.t + abs(self.rng.gauss(0.0003, 0.0001))
        msg = (lin, ang, stamp, src)
        lat = max(self.rng.gauss(0.0007, 0.0002), 0.0002)
        self.schedule(stamp + lat, 'arrive_stamped', msg)
        self.schedule(stamp + lat, 'arrive_delaynode', msg)

    def _enqueue(self, q, msg, name):
        if len(q) >= QUEUE_DEPTH:
            q.popleft()
            self.counters[f'drop_{name}'] += 1
        q.append(msg)

    # ----- security_node executor --------------------------------------- #
    def _node_ready(self):
        work = []
        if self.next_safe <= self.t or self.next_recovery <= self.t:
            work.append('timers')
        if self.q_stamped:
            work.append('stamped')
        if self.q_delayed:
            work.append('delayed')
        if self.q_scan:
            work.append('scan')
        return work

    def pump_node(self):
        while self.node_busy_until <= self.t:
            if not self.node_round:
                self.node_round = self._node_ready()
                if not self.node_round:
                    return
            kind = self.node_round.pop(0)
            cost = self._node_callback(kind)
            if cost > 0:
                self.node_busy_until = self.t + cost
                self.schedule(self.node_busy_until, 'node_wake')
                return

    def _node_callback(self, kind):
        if kind == 'timers':
            if self.next_safe <= self.t:
                while self.next_safe <= self.t:
                    self.next_safe += SAFE_PERIOD
                action = self.safety.safe_mode_action(self.t)
                if action is not None:
                    self.schedule(self.t + 0.001, 'robot_cmd', action)
            if self.next_recovery <= self.t:
                while self.next_recovery <= self.t:
                    self.next_recovery += RECOVERY_PERIOD
                if self.safety.check_recovery(self.t):
                    self.engine.reset_history()  # msg_timestamps kept (node does the same)
            return 0.0002
        if kind == 'scan':
            if self.q_scan:
                self.safety.update_scan(self.q_scan.popleft())
            return 0.0005
        q = self.q_stamped if kind == 'stamped' else self.q_delayed
        if not q:
            return 0.0
        lin, ang, stamp, src = q.popleft()
        return self._stamped_callback(lin, ang, stamp, src)

    def _stamped_callback(self, lin, ang, stamp, src):
        """Mirrors SecurityNode.stamped_callback (+ extract_features)."""
        w0 = time.perf_counter()
        self.msg_timestamps.append(self.t)  # arrival rate counted in every mode
        self.msg_timestamps = [t for t in self.msg_timestamps
                               if self.t - t < PUBLISH_RATE_WINDOW]
        rate = len(self.msg_timestamps) / PUBLISH_RATE_WINDOW
        if self.safety.mode != 'NORMAL':
            return 0.00005
        self.counters['processed'] += 1
        delay = max(self.t - stamp, 0.0)
        features = (lin, ang, rate, delay)
        self.current = dict(zip(('linear_x', 'angular_z', 'publish_rate', 'delay'), features))
        if not self.defense:  # comparison mode: forward everything (firmware still clamps)
            self.last_result = None
            self.schedule(self.t + 0.001, 'robot_cmd', clamp_cmd(lin, ang))
            return 0.0001

        result = self.engine.step(features, self.t)
        wall = time.perf_counter() - w0
        cost = wall * self.cpu_factor
        self.last_step_ms = wall * 1000.0
        self.last_result = result

        rule = result['sliding_anomaly'] or result['threshold_anomaly']
        if result['final_anomaly']:
            self._on_detection(result)
        else:
            if rule:
                self.vetoed += 1
            self.schedule(self.t + cost + 0.001, 'robot_cmd', clamp_cmd(lin, ang))
        return cost

    def _on_detection(self, result):
        active = [a for a in ATTACKS if self.active(a)]
        self._event('detect', f'DETECTED [{result["attack_type"]}] {result["reason"]}')
        hit = False
        for ep in self.episodes:
            if ep['status'] == 'active':
                ep.update(status='detected', latency=round(self.t - ep['start'], 3),
                          detected_as=result['attack_type'])
                hit = True
        if not active and not hit:
            self.false_alarms += 1
        self.safety.enter_safe_mode(result['reason'], self.t)

    # ----- delay_attack_node -------------------------------------------- #
    def pump_delaynode(self):
        while self.delay_busy_until <= self.t and self.dq:
            msg = self.dq.popleft()
            if self.active('delay'):
                self.delay_busy_until = self.t + DELAY_SECONDS
                self.schedule(self.delay_busy_until, 'delay_out', msg)
                return

    # ----- event loop ---------------------------------------------------- #
    def handle(self, kind, data):
        t = self.t
        if kind == 'teleop':
            self.driver_cmd = self._driver_cmd()
            self.publish_input(*self.driver_cmd, 'teleop')
            self.schedule(t + TELEOP_PERIOD, 'teleop')
        elif kind == 'spoof':
            if self.active('spoofing'):
                self.publish_input(3.0, 5.0, 'spoofing')
            self.schedule(t + SPOOF_PERIOD, 'spoof')
        elif kind == 'flood':
            if self.active('flooding'):
                self.publish_input(0.1, 0.0, 'flooding')
                self.schedule(t + FLOOD_PERIOD, 'flood')
            else:
                self.schedule(t + 0.05, 'flood')
        elif kind == 'arrive_stamped':
            self._enqueue(self.q_stamped, data, 'stamped')
            self.pump_node()
        elif kind == 'arrive_delayed':
            self._enqueue(self.q_delayed, data, 'delayed')
            self.pump_node()
        elif kind == 'arrive_delaynode':
            self._enqueue(self.dq, data, 'delaynode')
            self.pump_delaynode()
        elif kind == 'delay_out':
            lin, ang, stamp, _ = data
            self.counters['pub_delay'] += 1
            lat = max(self.rng.gauss(0.0007, 0.0002), 0.0002)
            self.schedule(t + lat, 'arrive_delayed', (lin, ang, stamp, 'delay'))
            self.pump_delaynode()
        elif kind in ('node_wake', 'node_timer'):
            if kind == 'node_timer':
                self.schedule(t + SAFE_PERIOD, 'node_timer')
            self.pump_node()
        elif kind == 'scan':
            self.ranges = _scan(self.x, self.y, self.yaw, self.rng)
            self._enqueue(self.q_scan, self.ranges, 'scan')
            self.pump_node()
            self.schedule(t + SCAN_PERIOD, 'scan')
        elif kind == 'robot_cmd':
            self.cmd = data
        elif kind == 'physics':
            self._physics()
            self.schedule(t + PHYSICS_DT, 'physics')
        elif kind == 'sample':
            self._sample()
            self.schedule(t + SAMPLE_PERIOD, 'sample')
        elif kind == 'scenario':
            self._scenario_tick()
            self.schedule(t + 0.2, 'scenario')

    def _physics(self):
        lin, ang = clamp_cmd(*self.cmd)        # OpenCR firmware limits
        nyaw = self.yaw + ang * PHYSICS_DT
        nx = self.x + lin * math.cos(self.yaw) * PHYSICS_DT
        ny = self.y + lin * math.sin(self.yaw) * PHYSICS_DT
        hit = _dist_to_walls(nx, ny) < ROBOT_RADIUS and \
            _dist_to_walls(nx, ny) < _dist_to_walls(self.x, self.y)
        if hit:
            if not self.colliding:
                self.collisions += 1
                self._event('collision', 'robot hit a wall')
        else:
            self.x, self.y = nx, ny
        self.colliding = hit
        self.yaw = math.atan2(math.sin(nyaw), math.cos(nyaw))
        if not self.trail or math.hypot(self.trail[-1][0] - self.x,
                                        self.trail[-1][1] - self.y) > 0.02:
            self.trail.append((round(self.x, 3), round(self.y, 3)))

    def _sample(self):
        self._update_episodes()
        c = self.current
        self.history.append((round(self.t, 2), round(c['linear_x'], 3), round(c['angular_z'], 3),
                             round(c['publish_rate'], 1), round(c['delay'], 4),
                             round(self.engine.last_ml_score, 4),
                             int(self.safety.mode != 'NORMAL'),
                             next((a for a in ATTACKS if self.active(a)), '')))
        while self.history and self.t - self.history[0][0] > HISTORY_WINDOW:
            self.history.popleft()

    def _scenario_tick(self):
        if self.auto_scenario and self.t >= self.next_scenario:
            self.start_attack(self.rng.choice(self.scenario['attacks']),
                              self.scenario['duration'])
            self.next_scenario = self.t + self.scenario['interval']

    def run_until(self, target, budget_s=0.03):
        """Process events up to sim time ``target`` (bounded wall time per call)."""
        w0 = time.perf_counter()
        while self.heap and self.heap[0][0] <= target:
            t, _, kind, data = heapq.heappop(self.heap)
            self.t = t
            self.handle(kind, data)
            if time.perf_counter() - w0 > budget_s:
                return False
        self.t = max(self.t, target)
        return True

    # ----- snapshot ------------------------------------------------------ #
    def snapshot(self):
        def num(v):
            return None if v is None or math.isinf(v) or math.isnan(v) else round(v, 3)
        sc, r = self.safety, self.last_result or {}
        return {
            't': round(self.t, 2),
            'realtime': round(self.realtime_ratio, 2),
            'paused': self.paused,
            'robot': {'x': self.x, 'y': self.y, 'yaw': self.yaw,
                      'cmd': [round(v, 3) for v in clamp_cmd(*self.cmd)],
                      'driver_cmd': [round(v, 3) for v in self.driver_cmd]},
            'ranges': [num(v) for v in self.ranges],
            'trail': list(self.trail),
            'mode': sc.mode,
            'wall': sc.wall_direction,
            'sectors': {k: num(getattr(sc, k)) for k in
                        ('front_min', 'back_min', 'left_min', 'right_min', 'wall_dist')},
            'wall_bearing': num(sc.wall_bearing),
            'recovery_remaining': num(sc.recovery_remaining(self.t)),
            'detection': sc.last_detection_reason,
            'attack_type': self.engine.last_attack_type,
            'ml_score': round(self.engine.last_ml_score, 4),
            'ml_threshold': self.ml_threshold,
            'ml_enabled': self.engine.ml_enabled,
            'gate': {k: bool(r.get(k, False)) for k in
                     ('hard_limit_anomaly', 'ml_anomaly', 'sliding_anomaly',
                      'threshold_anomaly', 'gated_anomaly', 'final_anomaly')},
            'features': {k: round(v, 4) for k, v in self.current.items()},
            'step_ms': round(self.last_step_ms, 2),
            'cpu_factor': self.cpu_factor,
            'defense': self.defense,
            'driver': self.driver_mode,
            'manual': [round(v, 2) for v in self.manual],
            'attacks': {a: round(max(self.attack_until[a] - self.t, 0.0), 1) for a in ATTACKS},
            'auto_scenario': self.auto_scenario,
            'next_scenario': round(max(self.next_scenario - self.t, 0.0), 1),
            'episodes': [{k: v for k, v in ep.items() if k not in ('end', 'saw_normal')}
                         for ep in self.episodes[-12:]],
            'false_alarms': self.false_alarms,
            'vetoed': self.vetoed,
            'collisions': self.collisions,
            'counters': dict(self.counters),
            'queues': {'stamped': len(self.q_stamped), 'delayed': len(self.q_delayed),
                       'delaynode': len(self.dq)},
            'events': list(self.events)[-40:],
            'history': list(self.history),
            'world': {'w': ROOM_W, 'h': ROOM_H, 'obstacles': OBSTACLES,
                      'radius': ROBOT_RADIUS, 'num_rays': NUM_RAYS},
            'limits': {'velocity': VELOCITY_THRESHOLD, 'angular': VELOCITY_THRESHOLD * 2,
                       'flood': FLOOD_THRESHOLD, 'delay': DELAY_THRESHOLD,
                       'max_linear': MAX_LINEAR, 'max_angular': MAX_ANGULAR},
        }

    realtime_ratio = 1.0


def sim_thread(sim):
    last_wall, last_sim = time.monotonic(), 0.0
    while True:
        with sim.lock:
            if sim.paused:
                sim.wall0, sim.sim0 = time.monotonic(), sim.t
            else:
                target = sim.sim0 + (time.monotonic() - sim.wall0)
                caught_up = sim.run_until(target)
                if not caught_up and target - sim.t > 2.0:
                    # can't keep up (CPU-bound ML) -> slow sim time instead of piling up
                    sim.wall0, sim.sim0 = time.monotonic(), sim.t
            now = time.monotonic()
            if now - last_wall >= 1.0:
                sim.realtime_ratio = (sim.t - last_sim) / (now - last_wall)
                last_wall, last_sim = now, sim.t
        time.sleep(0.004)


# --------------------------------------------------------------------------- #
# Web app
# --------------------------------------------------------------------------- #
def load_scenario():
    cfg = {'interval': 30.0, 'duration': 5.0, 'attacks': list(ATTACKS)}
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'config', 'attack_scenario.yaml')
    try:
        import yaml
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        cfg.update({k: data[k] for k in cfg if k in data})
    except Exception:  # noqa: BLE001 -- defaults like attack_manager_node
        pass
    return cfg


def create_app(sim):
    app = Flask(__name__)

    @app.route('/')
    def index():
        return Response(PAGE, mimetype='text/html')

    @app.route('/state')
    def state():
        with sim.lock:
            return jsonify(sim.snapshot())

    @app.route('/control', methods=['POST'])
    def control():
        c = request.get_json(force=True, silent=True) or {}
        act = c.get('action')
        with sim.lock:
            if act == 'attack':
                if c.get('on', True):
                    duration = float(c.get('duration', sim.scenario['duration']))
                    sim.start_attack(c.get('type'), duration)
                else:
                    sim.stop_attack(c.get('type'))
            elif act == 'driver':
                sim.driver_mode = c.get('mode', 'auto')
                sim.manual = [0.0, 0.0]
            elif act == 'key':
                k = c.get('key', '')
                lin, ang = sim.manual
                if k == 'w':
                    lin = min(lin + 0.01, MAX_LINEAR)
                elif k == 'x':
                    lin = max(lin - 0.01, -MAX_LINEAR)
                elif k == 'a':
                    ang = min(ang + 0.1, MAX_ANGULAR)
                elif k == 'd':
                    ang = max(ang - 0.1, -MAX_ANGULAR)
                elif k in ('s', ' '):
                    lin, ang = 0.0, 0.0
                sim.manual = [round(lin, 2), round(ang, 2)]
                sim.driver_mode = 'manual'
            elif act == 'sharp_turn':
                sim.sharp_until = sim.t + 1.5
                sim._event('info', 'driver: legit sharp turn (angular 2.5 for 1.5s)')
            elif act == 'defense':
                sim.defense = bool(c.get('on', True))
                sim._event('info', f'defense {"ON" if sim.defense else "OFF"}')
            elif act == 'cpu':
                sim.cpu_factor = max(0.25, min(float(c.get('factor', 1.0)), 8.0))
            elif act == 'scenario':
                sim.auto_scenario = bool(c.get('on', True))
                sim.next_scenario = sim.t + 2.0
            elif act == 'pause':
                sim.paused = bool(c.get('on', not sim.paused))
            elif act == 'reset':
                sim.reset()
        return jsonify(ok=True)

    return app


def main(args=None):
    ap = argparse.ArgumentParser(description='tb3_security browser simulator')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=5050)
    ap.add_argument('--model', default=DEFAULT_MODEL_PATH)
    ns, _ = ap.parse_known_args(args)

    engine = DetectionEngine(model_path=ns.model)
    if not engine.ml_enabled:
        print('[web_sim] WARNING: ML model not loaded - only hard limits will fire.')
    sim = Simulation(engine, load_scenario())
    threading.Thread(target=sim_thread, args=(sim,), daemon=True).start()
    print(f'[web_sim] open http://{ns.host}:{ns.port}')
    create_app(sim).run(host=ns.host, port=ns.port, threaded=True, use_reloader=False)


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
PAGE = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TB3 보안 시뮬레이터</title>
<style>
:root{--bg:#0f1318;--panel:#171c23;--panel2:#1e252e;--line:#2b3440;--text:#e6ebf1;
--muted:#8b97a6;--ok:#3fb27f;--warn:#e5a33a;--bad:#e5534b;--acc:#4c9be0;
--spoof:#e4572e;--flood:#f3a712;--delay:#a66bd3;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:14px/1.45 system-ui,-apple-system,"Noto Sans KR",sans-serif}
header{display:flex;align-items:center;gap:14px;flex-wrap:wrap;padding:12px 18px;
border-bottom:1px solid var(--line);background:var(--panel)}
h1{font-size:16px;margin:0;font-weight:650;letter-spacing:.2px}
.sub{color:var(--muted);font-size:12px}
.pill{padding:3px 10px;border-radius:999px;font-weight:650;font-size:12px;
border:1px solid var(--line);background:var(--panel2)}
.pill.ok{color:var(--ok);border-color:#2a5a45}.pill.bad{color:#fff;background:var(--bad);border-color:var(--bad)}
.pill.warn{color:var(--warn);border-color:#6a4d1d}
main{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(330px,1fr);gap:14px;padding:14px 18px}
@media (max-width:980px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);
margin:0 0 10px;font-weight:650}
.col{display:flex;flex-direction:column;gap:14px;min-width:0}
canvas{width:100%;display:block}
#map{aspect-ratio:4/3;background:#0b0e12;border-radius:8px}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
button{font:inherit;font-size:13px;color:var(--text);background:var(--panel2);
border:1px solid var(--line);border-radius:8px;padding:7px 11px;cursor:pointer}
button:hover{border-color:#4a5a6e}button.on{border-color:var(--acc);background:#1d3247}
button.atk{min-width:96px;font-weight:600}
button.atk.spoofing{border-left:4px solid var(--spoof)}button.atk.flooding{border-left:4px solid var(--flood)}
button.atk.delay{border-left:4px solid var(--delay)}button.atk.live{background:#3a1f1f;border-color:var(--bad)}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 12px;font-size:13px}
.kv b{color:var(--muted);font-weight:500}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.fsm{display:flex;gap:4px;flex-wrap:wrap}
.fsm span{padding:5px 8px;border-radius:6px;background:var(--panel2);border:1px solid var(--line);
font-size:11px;font-weight:600;color:var(--muted)}
.fsm span.cur{color:#fff;background:var(--bad);border-color:var(--bad)}
.fsm span.cur.n{background:var(--ok);border-color:var(--ok)}
.lamps{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
.lamp{padding:6px 8px;border-radius:7px;background:var(--panel2);border:1px solid var(--line);
font-size:12px;color:var(--muted)}.lamp.lit{color:#fff;border-color:var(--warn);background:#3b2c12}
.lamp.final.lit{border-color:var(--bad);background:#4a1c1c}
.formula{font-size:12px;color:var(--muted);margin-top:8px}
.bar{height:8px;background:var(--panel2);border-radius:4px;position:relative;overflow:hidden;margin-top:4px}
.bar i{position:absolute;top:0;bottom:0;left:0;background:var(--acc)}
.bar u{position:absolute;top:-2px;bottom:-2px;width:2px;background:var(--bad)}
table{width:100%;border-collapse:collapse;font-size:12px}
td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left}th{color:var(--muted);font-weight:500}
.s-detected{color:var(--ok)}.s-missed{color:var(--bad)}.s-active{color:var(--warn)}.s-blocked{color:var(--muted)}
.tabs{display:flex;gap:4px;margin:-2px 0 10px;border-bottom:1px solid var(--line)}
.tab{border:0;border-bottom:2px solid transparent;border-radius:0;background:none;padding:6px 10px;
font-size:12px;font-weight:650;letter-spacing:.8px;color:var(--muted)}
.tab:hover{color:var(--text)}.tab.on{color:var(--text);border-bottom-color:var(--acc);background:none}
.badge{margin-left:6px;padding:0 6px;border-radius:999px;background:var(--bad);color:#fff;font-size:11px}
.badge:empty{display:none}
#log{height:190px;overflow:auto;font-size:12px}
#log div{padding:2px 0;border-bottom:1px dashed #222a33}
.k-detect{color:#ff8a80}.k-attack{color:var(--warn)}.k-collision{color:var(--bad)}.k-fsm{color:#9cc9f5}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.stat{background:var(--panel2);border-radius:8px;padding:8px}
.stat b{display:block;font-size:20px}.stat span{font-size:11px;color:var(--muted)}
.charts canvas{height:62px;margin-bottom:6px;background:#0b0e12;border-radius:6px}
input[type=range]{width:140px}
.spd{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.spd .lbl{font-size:12px;color:var(--muted)}
.spd .big{font:700 38px/1.1 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:-1px}
.spd .big small{font-size:14px;font-weight:500;color:var(--muted);margin-left:4px;letter-spacing:0}
.gauge{position:relative;height:12px;background:var(--panel2);border-radius:6px;margin:8px 0 6px;overflow:hidden}
.gauge i{position:absolute;top:0;bottom:0;background:var(--acc);border-radius:6px}
.gauge u{position:absolute;top:0;bottom:0;width:2px;background:var(--bad);opacity:.8}
.gauge b{position:absolute;top:-2px;bottom:-2px;left:50%;width:1px;background:#5c6876}
.spd .sent{font-size:12px;color:var(--muted)}.spd .sent span{color:var(--text);font-family:ui-monospace,monospace}
#spdNote{margin-top:8px;font-size:12px;padding:5px 8px;border-radius:6px;display:none}
#spdNote.on{display:block;background:#3a1f1f;color:#ffb3ad}
.hint{font-size:12px;color:var(--muted)}
kbd{border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 5px;font-size:11px}
</style></head><body>
<header>
  <h1>TurtleBot3 보안 시뮬레이터</h1>
  <span class="sub">로봇·ROS 없이 · 실제 DetectionEngine + SafetyController 사용</span>
  <span id="modePill" class="pill ok">NORMAL</span>
  <span id="defPill" class="pill ok">방어 ON</span>
  <span class="sub mono" id="clock"></span>
</header>
<main>
 <div class="col">
  <div class="card">
   <h2>가상 맵 · LIDAR</h2>
   <canvas id="map" width="960" height="720"></canvas>
   <div class="row" style="margin-top:8px">
    <span class="hint">운전:</span>
    <button data-drv="auto">자동 순찰</button><button data-drv="manual">수동 (키보드)</button>
    <button data-drv="stop">정지</button>
    <button id="sharp" title="정상 데이터에 있는 급회전 — 규칙은 걸리지만 ML이 거부해야 정상">정상 급회전</button>
    <span class="hint">수동: <kbd>W</kbd><kbd>X</kbd> 선속도 · <kbd>A</kbd><kbd>D</kbd> 각속도 · <kbd>S</kbd> 정지</span>
   </div>
  </div>
  <div class="card charts">
   <h2>특징값 (최근 30초) — 붉은 선 = 규칙 임계값, 회색 띠 = 공격 구간</h2>
   <canvas id="c_lin" width="900" height="62"></canvas>
   <canvas id="c_ang" width="900" height="62"></canvas>
   <canvas id="c_rate" width="900" height="62"></canvas>
   <canvas id="c_delay" width="900" height="62"></canvas>
   <canvas id="c_ml" width="900" height="62"></canvas>
  </div>
 </div>
 <div class="col">
  <div class="card">
   <h2>현재 속도 — 로봇이 받은 /cmd_vel</h2>
   <div class="spd">
    <div><div class="lbl">선속도 linear.x</div>
     <div class="big" id="vLin">0.00<small>m/s</small></div>
     <div class="gauge"><b></b><i id="gLin"></i></div>
     <div class="sent">조종 명령 <span id="sLin">0.00</span> m/s · 한계 ±0.22</div></div>
    <div><div class="lbl">각속도 angular.z</div>
     <div class="big" id="vAng">0.00<small>rad/s</small></div>
     <div class="gauge"><b></b><u id="tAngP"></u><u id="tAngN"></u><i id="gAng"></i></div>
     <div class="sent">조종 명령 <span id="sAng">0.00</span> rad/s · 붉은 선 = 규칙 ±1.0</div></div>
   </div>
   <div id="spdNote"></div>
  </div>
  <div class="card">
   <h2>공격</h2>
   <div class="row">
    <button class="atk spoofing" data-atk="spoofing">속도 스푸핑</button>
    <button class="atk flooding" data-atk="flooding">플러딩</button>
    <button class="atk delay" data-atk="delay">지연</button>
   </div>
   <div class="row" style="margin-top:8px">
    <button id="scen">자동 시나리오</button>
    <button id="def">방어 끄기</button>
    <button id="pause">일시정지</button>
    <button id="reset">리셋</button>
   </div>
   <div class="row" style="margin-top:8px">
    <span class="hint">노드 CPU 속도 배율</span>
    <input type="range" id="cpu" min="0.25" max="6" step="0.25" value="1">
    <span class="mono" id="cpuv">1.0×</span>
    <span class="hint" id="stepms"></span>
   </div>
  </div>
  <div class="card">
   <h2>탐지 판정</h2>
   <div class="lamps">
    <div class="lamp" id="g_hard">하드 한계</div><div class="lamp" id="g_ml">ML 이상</div>
    <div class="lamp" id="g_slide">통계(4σ)</div><div class="lamp" id="g_vel">속도 규칙</div>
    <div class="lamp" id="g_gated">ML ∧ 규칙</div><div class="lamp final" id="g_final">최종 탐지</div>
   </div>
   <div class="formula mono">최종 = 하드한계(flood·delay·속도 초과) ∨ (ML ∧ (통계 ∨ 속도))</div>
   <div style="margin-top:10px" class="kv">
    <b>ML 점수</b><span class="mono" id="mlscore">–</span>
   </div>
   <div class="bar"><i id="mlbar"></i><u id="mlthr"></u></div>
   <div class="kv" style="margin-top:10px">
    <b>공격 유형</b><span id="atype">none</span>
    <b>마지막 탐지</b><span class="mono" id="reason">–</span>
   </div>
  </div>
  <div class="card">
   <h2>안전 모드 FSM</h2>
   <div class="fsm" id="fsm"></div>
   <div class="kv" style="margin-top:8px">
    <b>목표 벽</b><span id="wall">–</span><b>복귀까지</b><span id="recov">–</span>
    <b>섹터</b><span class="mono" id="sect">–</span>
   </div>
  </div>
  <div class="card">
   <div class="tabs">
    <button class="tab on" data-tab="pane_res">결과</button>
    <button class="tab" data-tab="pane_log">이벤트 로그<span class="badge" id="logBadge"></span></button>
   </div>
   <div class="pane" id="pane_res">
   <div class="stats">
    <div class="stat"><b id="st_det">0</b><span>탐지한 공격</span></div>
    <div class="stat"><b id="st_miss">0</b><span>놓친 공격</span></div>
    <div class="stat"><b id="st_fp">0</b><span>오탐 (공격 없음)</span></div>
    <div class="stat"><b id="st_col">0</b><span>벽 충돌</span></div>
   </div>
   <div class="hint" style="margin:8px 0 4px">ML이 막은 규칙 오탐: <b id="st_veto">0</b> ·
     처리 <span id="st_proc">0</span> / 큐 드롭 <span id="st_drop">0</span></div>
   <table><thead><tr><th>시각</th><th>공격</th><th>결과</th><th>지연</th><th>판정</th></tr></thead>
   <tbody id="eps"></tbody></table>
   </div>
   <div class="pane" id="pane_log" hidden><div id="log" class="mono"></div></div>
  </div>
 </div>
</main>
<script>
const $=id=>document.getElementById(id);
const COL={spoofing:'#e4572e',flooding:'#f3a712',delay:'#a66bd3'};
const NAME={spoofing:'속도 스푸핑',flooding:'플러딩',delay:'지연'};
const STATUS={detected:'탐지',missed:'놓침',active:'진행 중',blocked:'안전모드가 차단'};
let S=null;
let redAlignRel=null;  // red wedge bearing (robot frame) frozen when alignment starts
let logSeenT=-1;  // sim time of the newest event already shown in the log tab
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('.tab').forEach(t=>{t.classList.toggle('on',t===b);$(t.dataset.tab).hidden=t!==b;});
  if(b.dataset.tab==='pane_log'){const L=$('log');L.scrollTop=L.scrollHeight;
    if(S&&S.events.length)logSeenT=S.events[S.events.length-1].t;$('logBadge').textContent='';}
});
function post(o){return fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(o)})}
document.querySelectorAll('[data-atk]').forEach(b=>b.onclick=()=>{
  const t=b.dataset.atk;post({action:'attack',type:t,on:!(S&&S.attacks[t]>0)})});
document.querySelectorAll('[data-drv]').forEach(b=>b.onclick=()=>post({action:'driver',mode:b.dataset.drv}));
$('sharp').onclick=()=>post({action:'sharp_turn'});
$('scen').onclick=()=>post({action:'scenario',on:!(S&&S.auto_scenario)});
$('def').onclick=()=>post({action:'defense',on:!(S&&S.defense)});
$('pause').onclick=()=>post({action:'pause',on:!(S&&S.paused)});
$('reset').onclick=()=>post({action:'reset'});
$('cpu').oninput=e=>{$('cpuv').textContent=(+e.target.value).toFixed(2)+'×';post({action:'cpu',factor:+e.target.value})};
addEventListener('keydown',e=>{const k=e.key.toLowerCase();if('wxads '.includes(k)&&k){e.preventDefault();post({action:'key',key:k})}});

function drawMap(s){
  const c=$('map'),g=c.getContext('2d'),W=c.width,H=c.height,w=s.world,pad=24;
  const sc=Math.min((W-2*pad)/w.w,(H-2*pad)/w.h),ox=(W-w.w*sc)/2,oy=(H-w.h*sc)/2;
  const P=(x,y)=>[ox+x*sc,H-oy-y*sc];
  g.clearRect(0,0,W,H);
  g.strokeStyle='#1b222b';g.lineWidth=1;
  for(let x=0;x<=w.w+1e-6;x+=0.5){const[a,b]=P(x,0),[,d]=P(x,w.h);g.beginPath();g.moveTo(a,b);g.lineTo(a,d);g.stroke()}
  for(let y=0;y<=w.h+1e-6;y+=0.5){const[a,b]=P(0,y),[c2]=P(w.w,y);g.beginPath();g.moveTo(a,b);g.lineTo(c2,b);g.stroke()}
  g.strokeStyle='#c9d3de';g.lineWidth=3;const[a0,b0]=P(0,w.h);g.strokeRect(a0,b0,w.w*sc,w.h*sc);
  g.fillStyle='#3a4552';w.obstacles.forEach(o=>{const[a,b]=P(o[0],o[3]);g.fillRect(a,b,(o[2]-o[0])*sc,(o[3]-o[1])*sc)});
  // trail
  g.strokeStyle='rgba(76,155,224,.45)';g.lineWidth=2;g.beginPath();
  s.trail.forEach((p,i)=>{const[a,b]=P(p[0],p[1]);i?g.lineTo(a,b):g.moveTo(a,b)});g.stroke();
  const r=s.robot,n=w.num_rays;
  // sectors (90 deg each, as in safety_core)
  const secs={front:0,left:90,back:180,right:270};
  for(const[k,c]of Object.entries(secs)){
    const a1=r.yaw+(c-45)*Math.PI/180,a2=r.yaw+(c+45)*Math.PI/180,[cx,cy]=P(r.x,r.y);
    g.fillStyle='rgba(255,255,255,.035)';
    g.beginPath();g.moveTo(cx,cy);g.arc(cx,cy,0.6*sc,-a2,-a1);g.closePath();g.fill();
  }
  // safe-mode target wall: red wedge centred on the tracked wall normal (world-fixed,
  // so it keeps pointing at the wall while the robot turns toward / drives to it).
  // From SAFE_ALIGN on it is frozen relative to the robot and turns with it.
  const lateParking=s.mode==='SAFE_ALIGN'||s.mode==='SAFE_STOP';
  if(!lateParking||s.wall_bearing==null)redAlignRel=null;
  else if(redAlignRel==null)redAlignRel=s.wall_bearing;
  if(s.wall_bearing!=null){
    const wb=r.yaw+(redAlignRel!=null?redAlignRel:s.wall_bearing),[cx,cy]=P(r.x,r.y);
    g.fillStyle='rgba(229,83,75,.28)';
    g.beginPath();g.moveTo(cx,cy);g.arc(cx,cy,0.6*sc,-(wb+Math.PI/4),-(wb-Math.PI/4));g.closePath();g.fill();
  }
  // wall line fitted by safety_core (what safe mode parks against)
  if(s.wall_bearing!=null&&s.sectors.wall_dist!=null){
    const nb=r.yaw+s.wall_bearing,d=s.sectors.wall_dist,fx=r.x+d*Math.cos(nb),fy=r.y+d*Math.sin(nb);
    const[a,b]=P(fx-0.6*Math.sin(nb),fy+0.6*Math.cos(nb)),[c,e]=P(fx+0.6*Math.sin(nb),fy-0.6*Math.cos(nb));
    g.strokeStyle='#ff8a80';g.lineWidth=3;g.setLineDash([6,4]);g.beginPath();g.moveTo(a,b);g.lineTo(c,e);g.stroke();g.setLineDash([]);
  }
  // lidar
  g.fillStyle='#5fe0a5';
  s.ranges.forEach((d,i)=>{if(d==null)return;const a=r.yaw+2*Math.PI*i/n;
    const[px,py]=P(r.x+d*Math.cos(a),r.y+d*Math.sin(a));g.fillRect(px-1.5,py-1.5,3,3)});
  // robot
  const[cx,cy]=P(r.x,r.y);const safe=s.mode!=='NORMAL';
  g.fillStyle=safe?'#e5534b':'#4c9be0';g.beginPath();g.arc(cx,cy,w.radius*sc,0,7);g.fill();
  g.strokeStyle='#fff';g.lineWidth=3;g.beginPath();g.moveTo(cx,cy);
  g.lineTo(cx+Math.cos(r.yaw)*w.radius*sc*1.5,cy-Math.sin(r.yaw)*w.radius*sc*1.5);g.stroke();
  const live=Object.entries(s.attacks).filter(([,v])=>v>0).map(([k])=>k);
  g.font='600 20px system-ui';g.textAlign='left';
  live.forEach((k,i)=>{g.fillStyle=COL[k];g.fillText('▲ '+NAME[k]+' 공격 중 '+s.attacks[k].toFixed(1)+'s',16,30+i*26)});
  g.fillStyle='#8b97a6';g.font='15px ui-monospace,monospace';g.textAlign='right';
  g.fillText(`cmd_vel  v=${r.cmd[0].toFixed(2)}  ω=${r.cmd[1].toFixed(2)}`,W-16,H-12);
}

function chart(id,idx,label,thr,fmt,s,opts={}){
  const c=$(id),g=c.getContext('2d'),W=c.width,H=c.height,h=s.history;g.clearRect(0,0,W,H);
  if(!h.length)return;const t1=h[h.length-1][0],t0=t1-30;
  const X=t=>(t-t0)/30*W;
  // attack / safe bands
  h.forEach((p,i)=>{if(i===0)return;const x0=X(h[i-1][0]),x1=X(p[0]);
    if(p[7]){g.fillStyle=COL[p[7]]+'33';g.fillRect(x0,0,x1-x0+1,H)}
    if(p[6]){g.fillStyle='rgba(229,83,75,.5)';g.fillRect(x0,H-4,x1-x0+1,4)}});
  let vals=h.map(p=>opts.abs?Math.abs(p[idx]):p[idx]);
  let lo=opts.lo??Math.min(0,...vals),hi=Math.max(opts.hi??0,...vals,thr??-1e9);
  if(opts.lo!=null)lo=Math.min(opts.lo,...vals);
  if(hi-lo<1e-9)hi=lo+1;const Y=v=>H-6-(v-lo)/(hi-lo)*(H-14);
  if(thr!=null){g.strokeStyle='#e5534b';g.setLineDash([5,4]);g.lineWidth=1.2;g.beginPath();g.moveTo(0,Y(thr));g.lineTo(W,Y(thr));g.stroke();g.setLineDash([])}
  g.strokeStyle='#cfe3f7';g.lineWidth=1.6;g.beginPath();
  h.forEach((p,i)=>{const x=X(p[0]),y=Y(vals[i]);i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke();
  g.fillStyle='#8b97a6';g.font='13px system-ui';g.textAlign='left';g.fillText(label,8,15);
  g.textAlign='right';g.fillStyle='#e6ebf1';g.fillText(fmt(vals[vals.length-1]),W-8,15);
}

function gauge(id,v,max){const e=$(id),f=Math.max(-1,Math.min(1,v/max))*50;
  e.style.left=(f>=0?50:50+f)+'%';e.style.width=Math.abs(f)+'%';}
function speed(s){
  const[l,a]=s.robot.cmd,[dl,da]=s.robot.driver_cmd,L=s.limits;
  $('vLin').innerHTML=l.toFixed(2)+'<small>m/s</small>';$('vAng').innerHTML=a.toFixed(2)+'<small>rad/s</small>';
  gauge('gLin',l,L.max_linear);gauge('gAng',a,L.max_angular);
  const t=L.angular/L.max_angular*50;$('tAngP').style.left=(50+t)+'%';$('tAngN').style.left=(50-t)+'%';
  $('sLin').textContent=dl.toFixed(2);$('sAng').textContent=da.toFixed(2);
  const safe=s.mode!=='NORMAL';['gLin','gAng'].forEach(id=>$(id).style.background=safe?'#e5534b':'#4c9be0');
  const n=$('spdNote'),live=Object.values(s.attacks).some(v=>v>0);
  if(safe){n.className='on';n.textContent='안전 모드가 직접 제어 중 — 조종 명령·공격 명령은 무시됨'}
  else if(!s.defense&&live){n.className='on';n.textContent='방어 OFF — 공격 명령이 그대로 로봇에 전달되는 중'}
  else n.className='';
}
function render(s){
  S=s;const safe=s.mode!=='NORMAL';speed(s);
  $('modePill').textContent=s.mode;$('modePill').className='pill '+(safe?'bad':'ok');
  $('defPill').textContent=s.defense?'방어 ON':'방어 OFF';$('defPill').className='pill '+(s.defense?'ok':'warn');
  $('clock').textContent=`t=${s.t.toFixed(1)}s · 실시간 ${(s.realtime*100).toFixed(0)}%`+(s.paused?' · 일시정지':'');
  document.querySelectorAll('[data-atk]').forEach(b=>{const v=s.attacks[b.dataset.atk];
    b.classList.toggle('live',v>0);b.textContent=NAME[b.dataset.atk]+(v>0?` ${v.toFixed(1)}s`:'')});
  document.querySelectorAll('[data-drv]').forEach(b=>b.classList.toggle('on',b.dataset.drv===s.driver));
  $('scen').classList.toggle('on',s.auto_scenario);
  $('scen').textContent=s.auto_scenario?`자동 시나리오 (다음 ${s.next_scenario.toFixed(0)}s)`:'자동 시나리오';
  $('def').textContent=s.defense?'방어 끄기':'방어 켜기';$('def').classList.toggle('on',!s.defense);
  $('pause').textContent=s.paused?'재개':'일시정지';
  $('stepms').textContent=`ML 1회 ${s.step_ms.toFixed(1)} ms → 노드 처리한계 ≈ ${s.step_ms>0?Math.round(1000/(s.step_ms*s.cpu_factor)):'–'} msg/s`;
  const G=s.gate;[['g_hard','hard_limit_anomaly'],['g_ml','ml_anomaly'],['g_slide','sliding_anomaly'],
   ['g_vel','threshold_anomaly'],['g_gated','gated_anomaly'],['g_final','final_anomaly']].forEach(([id,k])=>$(id).classList.toggle('lit',G[k]));
  $('mlscore').textContent=s.ml_enabled?`${s.ml_score.toFixed(3)}  (임계 ${s.ml_threshold?.toFixed(3)} 미만이면 이상)`:'모델 없음';
  const lo=-0.85,hi=-0.3,pct=v=>Math.max(0,Math.min(100,(v-lo)/(hi-lo)*100));
  $('mlbar').style.width=pct(s.ml_score)+'%';if(s.ml_threshold!=null)$('mlthr').style.left=pct(s.ml_threshold)+'%';
  $('mlbar').style.background=s.ml_score<s.ml_threshold?'#e5534b':'#4c9be0';
  $('atype').textContent=s.attack_type;$('reason').textContent=s.detection||'–';
  $('fsm').innerHTML=['NORMAL','SAFE_ROTATE','SAFE_FORWARD','SAFE_ALIGN','SAFE_STOP'].map(m=>
   `<span class="${m===s.mode?'cur'+(m==='NORMAL'?' n':''):''}">${m}</span>`).join('');
  $('wall').textContent=s.wall||'–';$('recov').textContent=s.recovery_remaining!=null?s.recovery_remaining.toFixed(1)+'s':'–';
  const f=v=>v==null?'∞':v.toFixed(2);const q=s.sectors;
  $('sect').textContent=`F ${f(q.front_min)} · B ${f(q.back_min)} · L ${f(q.left_min)} · R ${f(q.right_min)}`;
  const eps=s.episodes;$('st_det').textContent=eps.filter(e=>e.status==='detected').length;
  $('st_miss').textContent=eps.filter(e=>e.status==='missed').length;
  $('st_fp').textContent=s.false_alarms;$('st_col').textContent=s.collisions;$('st_veto').textContent=s.vetoed;
  const C=s.counters;$('st_proc').textContent=C.processed||0;
  $('st_drop').textContent=(C.drop_stamped||0)+(C.drop_delayed||0);
  $('eps').innerHTML=eps.slice().reverse().map(e=>`<tr><td>${e.start.toFixed(1)}s</td>
   <td style="color:${COL[e.attack]}">${NAME[e.attack]}</td><td class="s-${e.status}">${STATUS[e.status]}</td>
   <td>${e.latency!=null?e.latency.toFixed(2)+'s':'–'}</td><td>${e.detected_as||'–'}</td></tr>`).join('');
  const L=$('log'),atBottom=L.scrollTop+L.clientHeight>=L.scrollHeight-4;
  L.innerHTML=s.events.map(e=>`<div class="k-${e.kind}">[${e.t.toFixed(1)}] ${e.text.replace(/</g,'&lt;')}</div>`).join('');
  if(atBottom)L.scrollTop=L.scrollHeight;
  const lastT=s.events.length?s.events[s.events.length-1].t:-1;
  if(lastT<logSeenT)logSeenT=-1;  // simulator was reset
  if(!$('pane_log').hidden)logSeenT=lastT;
  const unseen=s.events.filter(e=>e.t>logSeenT).length;
  $('logBadge').textContent=unseen?(unseen>=40?'40+':unseen):'';
  drawMap(s);const lim=s.limits;
  chart('c_lin',1,'|linear_x| m/s',lim.velocity,v=>v.toFixed(2),s,{abs:true,hi:0.3});
  chart('c_ang',2,'|angular_z| rad/s',lim.angular,v=>v.toFixed(2),s,{abs:true,hi:1.2});
  chart('c_rate',3,'publish_rate msg/s',lim.flood,v=>v.toFixed(0),s,{hi:60});
  chart('c_delay',4,'delay s',lim.delay,v=>v.toFixed(3),s,{hi:1.2});
  chart('c_ml',5,'ML score (낮을수록 이상)',s.ml_threshold,v=>v.toFixed(3),s,{lo:-0.8,hi:-0.3});
}
async function tick(){try{const r=await fetch('/state');render(await r.json())}catch(e){}setTimeout(tick,120)}
tick();
</script></body></html>
'''


if __name__ == '__main__':
    main()
