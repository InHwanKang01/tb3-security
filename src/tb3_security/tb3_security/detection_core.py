"""Framework-agnostic detection core shared by the live node and the offline evaluator.

This module holds the *pure* hybrid detection logic (hard safety limits plus
an ML anomaly gate over the rule checks) with no ROS/rclpy dependency, so
exactly the same code path can be:

  * driven live by ``security_node`` (real robot / simulation), and
  * replayed offline by ``evaluate`` on recorded / synthesised data.

Keeping a single source of truth means the numbers in the evaluation report
reflect the behaviour that actually runs on the robot -- there is no second,
drifting copy of the thresholds or the gate logic.
"""

import math
import os

try:
    import joblib
    import numpy as np
    HAS_ML_DEPS = True
except ImportError:  # graceful degradation -- ML gate disabled, node still clamps
    HAS_ML_DEPS = False

# --------------------------------------------------------------------------- #
# Detection constants (single source of truth -- security_node imports these)
# --------------------------------------------------------------------------- #
# Rule-based thresholds
VELOCITY_THRESHOLD = 0.5     # |linear.x| (or |angular.z|/2) above -> spoofing, ML-gated
FLOOD_THRESHOLD = 50         # msgs per PUBLISH_RATE_WINDOW -> flooding, hard (bypasses ML)
DELAY_THRESHOLD = 1.0        # command age in seconds -> delay, hard (bypasses ML)
# Physically impossible commands -> spoofing, hard (bypasses ML). TurtleBot3 Burger
# tops out at 0.22 m/s / 2.84 rad/s (= max of the normal-driving data); the
# IsolationForest is flat beyond its training range and vetoed e.g. Twist(3.0, 5.0)
# whenever the publish rate looked normal (stealth eval 2026-10-25: 37-60 % recall).
SPOOF_LINEAR_LIMIT = 0.3     # |linear.x| above -> spoofing, hard
SPOOF_ANGULAR_LIMIT = 3.0    # |angular.z| above -> spoofing, hard

# Statistical (sliding-window) detection
WINDOW_SIZE = 5.0            # seconds of history kept
SIGMA_THRESHOLD = 4.0        # deviations from window mean to flag
MIN_SAMPLES = 10             # window must hold at least this many samples

# publish-rate measurement window (seconds)
PUBLISH_RATE_WINDOW = 1.0

# ML runs lazily: always when a velocity rule fires (it is the gate), otherwise
# at most this often just to keep the dashboard score fresh. One IsolationForest
# pass costs ~10 ms; running it on every message capped the node at ~50 msg/s,
# which hid floods (the measured publish_rate could never exceed the threshold).
ML_MONITOR_PERIOD = 0.2

# The ML-gated velocity path only confirms an attack after CONFIRM_COUNT hits
# within CONFIRM_WINDOW seconds. Legit sharp turns produced isolated single-message
# alarms (11 in 262 s of normal driving -> 2 with 2-in-1 s); a spoofer at 2 Hz
# still confirms on its 2nd message. Hard limits stay immediate.
CONFIRM_WINDOW = 1.0
CONFIRM_COUNT = 2

# Default trained-model location
DEFAULT_MODEL_PATH = os.path.expanduser(
    '~/graduation_ws/src/tb3_security/models/isolation_forest.pkl')

# Feature vector order used everywhere: (linear_x, angular_z, publish_rate, delay)
FEATURE_COLUMNS = ['linear_x', 'angular_z', 'publish_rate', 'delay']


def load_model(model_path=DEFAULT_MODEL_PATH, logger=None):
    """Load the sklearn Pipeline, mirroring security_node._load_model messaging.

    ``logger`` may be any object with ``.info/.warn/.error`` (e.g. a ROS logger);
    pass ``None`` for silent offline use. Returns the model or ``None`` (gate off).
    """
    def _log(level, msg):
        if logger is not None:
            getattr(logger, level, lambda *_: None)(msg)

    if not HAS_ML_DEPS:
        _log('error', 'scikit-learn/joblib/numpy not installed - ML gate disabled.')
        return None
    if not os.path.exists(model_path):
        _log('warn', f'ML model not found at {model_path} - ML gate disabled.')
        return None
    try:
        model = joblib.load(model_path)
        _log('info', f'ML model loaded from {model_path}')
        return model
    except Exception as e:  # noqa: BLE001 -- degrade gracefully like the node
        _log('error', f'Failed to load ML model ({e}) - ML gate disabled.')
        return None


def _calc_stats(values):
    """Population mean / std, matching security_node._calc_stats exactly."""
    if len(values) < 2:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(variance)


class DetectionEngine:
    """Stateful hybrid detector: hard limits OR (ML gate AND (sliding OR velocity)).

    One ``step()`` call == one incoming stamped command. The engine owns the
    sliding-window history; the FSM / actuation stays in the ROS node. Timing is
    injected via ``now`` so offline replay uses recorded timestamps instead of
    the wall clock.
    """

    def __init__(self, model=None, model_path=DEFAULT_MODEL_PATH, logger=None,
                 ml_monitor_period=ML_MONITOR_PERIOD):
        # allow passing a preloaded model (tests) or a path (normal use)
        self.model = model if model is not None else load_model(model_path, logger)
        self.ml_offset = self._find_offset(self.model)
        self.ml_monitor_period = ml_monitor_period
        self._last_ml_time = None
        self.velocity_history = []          # list of (t, |linear|, |angular|)
        self._gated_times = []              # recent unconfirmed ML-gated hits
        # last-decision cache (read by the node for /security_status)
        self.last_ml_score = 0.0
        self.last_ml_anomaly = False
        self.last_attack_type = 'none'      # none|flooding|spoofing|delay

    @staticmethod
    def _find_offset(model):
        """IsolationForest.offset_ (also inside a Pipeline), else None."""
        est = model
        if hasattr(model, 'steps'):
            est = model.steps[-1][1]
        return getattr(est, 'offset_', None)

    @property
    def ml_enabled(self):
        return self.model is not None

    def reset_history(self):
        """Clear sliding-window state (node calls this on recovery to NORMAL)."""
        self.velocity_history = []
        self.last_attack_type = 'none'
        self._last_ml_time = None
        self._gated_times = []

    # ----- Stage 1: ML anomaly ------------------------------------------- #
    def predict_ml_anomaly(self, features):
        if self.model is None:
            self.last_ml_anomaly = False
            return False
        x = np.array([features], dtype=float)
        self.last_ml_score = float(self.model.score_samples(x)[0])
        if self.ml_offset is not None:
            # one forward pass: IsolationForest.predict is exactly
            # -1 where (score_samples - offset_) < 0
            self.last_ml_anomaly = (self.last_ml_score - self.ml_offset) < 0
        else:
            self.last_ml_anomaly = int(self.model.predict(x)[0]) == -1
        return self.last_ml_anomaly

    def ml_score(self, features):
        """Raw IsolationForest score_samples (higher = more normal). For ROC/PR."""
        if self.model is None:
            return 0.0
        x = np.array([features], dtype=float)
        return float(self.model.score_samples(x)[0])

    # ----- Stage 2a: statistical sliding window -------------------------- #
    def check_sliding_window(self, features, now):
        linear_x, angular_z, _, _ = features
        self.velocity_history.append((now, abs(linear_x), abs(angular_z)))
        self.velocity_history = [
            (t, lx, az) for t, lx, az in self.velocity_history if now - t < WINDOW_SIZE
        ]
        if len(self.velocity_history) < MIN_SAMPLES:
            return False

        linear_vals = [lx for _, lx, _ in self.velocity_history[:-1]]
        angular_vals = [az for _, _, az in self.velocity_history[:-1]]
        lin_mean, lin_std = _calc_stats(linear_vals)
        ang_mean, ang_std = _calc_stats(angular_vals)

        cur_lin, cur_ang = abs(linear_x), abs(angular_z)
        lin_dev = (cur_lin - lin_mean) / lin_std if lin_std > 0.1 else 0.0
        ang_dev = (cur_ang - ang_mean) / ang_std if ang_std > 0.1 else 0.0

        if lin_dev > SIGMA_THRESHOLD or ang_dev > SIGMA_THRESHOLD:
            self.last_attack_type = 'spoofing'
            return True
        return False

    # ----- Stage 2b: velocity threshold (ambiguous -> ML-gated) ---------- #
    def check_threshold(self, features):
        """Velocity limit only. Legit sharp turns can exceed it, so ML gates it."""
        linear_x, angular_z, _, _ = features
        if abs(linear_x) > VELOCITY_THRESHOLD or abs(angular_z) > VELOCITY_THRESHOLD * 2:
            self.last_attack_type = 'spoofing'
            return True
        return False

    # ----- Stage 0: hard safety limits (bypass the ML veto) ------------- #
    def check_hard_limits(self, features):
        """Flood rate / command delay / impossible speed: never seen in normal driving.

        The IsolationForest only saw publish_rate 1-14, delay < 0.01 s and speeds
        within the robot's limits, so its score is flat outside that range and it
        would veto these alarms. They therefore trigger safe mode on their own.
        Returns the attack type or None.
        """
        linear_x, angular_z, publish_rate, delay = features
        if publish_rate * PUBLISH_RATE_WINDOW > FLOOD_THRESHOLD:
            return 'flooding'
        if delay > DELAY_THRESHOLD:
            return 'delay'
        if abs(linear_x) > SPOOF_LINEAR_LIMIT or abs(angular_z) > SPOOF_ANGULAR_LIMIT:
            return 'spoofing'
        return None

    # ----- Combined gate ------------------------------------------------- #
    def step(self, features, now, full=False):
        """Run the hybrid gate on one command.

            final = hard_limit OR (ml_anomaly AND (sliding OR velocity_threshold))

        where the ML-gated term must fire CONFIRM_COUNT times within
        CONFIRM_WINDOW s (``gated_raw_anomaly`` is the single-message value).

        ML only matters when a velocity rule fired, so it is evaluated lazily
        (plus every ``ml_monitor_period`` for the dashboard); the decision is
        identical to running it every time. ``full=True`` forces it so the
        evaluator gets ``ml_anomaly`` for every sample (AND-only / ML-only
        ablations). Returns a dict with every sub-decision.
        """
        self.last_attack_type = 'none'
        sliding = self.check_sliding_window(features, now)
        velocity = self.check_threshold(features)
        hard_type = self.check_hard_limits(features)
        hard = hard_type is not None

        due = self._last_ml_time is None or now - self._last_ml_time >= self.ml_monitor_period
        ml_evaluated = full or sliding or velocity or due
        if ml_evaluated:
            self.predict_ml_anomaly(features)
            self._last_ml_time = now
        # when not evaluated no rule fired, so the (stale) value can't affect `gated`
        ml_anomaly = self.last_ml_anomaly

        gated_raw = ml_anomaly and (sliding or velocity)
        if gated_raw:
            self._gated_times.append(now)
        self._gated_times = [t for t in self._gated_times if now - t < CONFIRM_WINDOW]
        gated = gated_raw and len(self._gated_times) >= CONFIRM_COUNT
        final = hard or gated
        if hard:
            # a 1.5 s-old or 500 Hz command is conclusive; a delayed legit sharp
            # turn also trips the velocity rule but is not spoofing
            self.last_attack_type = hard_type
        elif not gated:
            # rule tripped but ML vetoed (e.g. legit sharp turn) -> not an attack;
            # don't leak 'spoofing' into /security_status
            self.last_attack_type = 'none'

        reason = ''
        if final:
            parts = []
            if hard:
                parts.append(f'hard:{hard_type}')
            if gated:
                rules = [n for n, v in (('sliding', sliding), ('threshold', velocity)) if v]
                parts.append('ML+' + '+'.join(rules))
            reason = f'{" | ".join(parts)} (score={self.last_ml_score:.3f})'

        return {
            'ml_anomaly': ml_anomaly,
            'ml_evaluated': ml_evaluated,
            'sliding_anomaly': sliding,
            # rule-level union (velocity OR hard limit) -- keeps the rule-only
            # ablation identical to the pre-hybrid definition
            'threshold_anomaly': velocity or hard,
            'hard_limit_anomaly': hard,
            'gated_anomaly': gated,
            # single-message gate decision before the CONFIRM_COUNT check
            'gated_raw_anomaly': gated_raw,
            'final_anomaly': final,
            'attack_type': self.last_attack_type,
            'ml_score': self.last_ml_score,
            'reason': reason,
        }
