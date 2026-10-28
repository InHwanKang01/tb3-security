"""Offline, robot-free dataset builder for evaluating the detection pipeline.

Two datasets are produced, both faithful to how the attacker nodes actually
publish (see flood/abnormal_velocity/delay_attack_node) and how security_node
sees the resulting feature vectors:

  * a **classification set** -- real normal-driving rows plus synthesised attack
    samples, used for confusion matrix / precision-recall / feature-space plots.
  * a **temporal scenario** -- a time-ordered command stream where attacks turn
    ON for ``duration`` seconds every ``interval`` seconds (mirroring
    attack_manager_node), used for detection-latency and timeline plots.

Everything is seeded for reproducibility; no ROS and no robot required.
"""

import os

import numpy as np
import pandas as pd

from tb3_security.detection_core import FEATURE_COLUMNS

DEFAULT_NORMAL_CSV = os.path.expanduser(
    '~/graduation_ws/src/tb3_security/data/normal_driving.csv')

ATTACK_TYPES = ['spoofing', 'flooding', 'delay']
CLASSES = ['normal'] + ATTACK_TYPES


def load_normal(csv_path=DEFAULT_NORMAL_CSV):
    df = pd.read_csv(csv_path)
    return df[FEATURE_COLUMNS].copy()


# --------------------------------------------------------------------------- #
# Attack feature synthesis -- mirrors the real attacker nodes
# --------------------------------------------------------------------------- #
def _sample_normal_rows(normal, n, rng):
    idx = rng.integers(0, len(normal), size=n)
    return normal.iloc[idx].reset_index(drop=True)


def make_attack_samples(attack, n, normal, rng):
    """Return an (n x 4) DataFrame of feature vectors for one attack type.

    Values reproduce what security_node measures while the attack is active:
      * spoofing  -> abnormal_velocity_node: Twist(3.0, 5.0) at ~2 Hz
      * flooding  -> flood_attack_node: Twist(0.1, 0.0) at ~500 Hz
      * delay     -> delay_attack_node: normal cmd, but 1.5 s stale + low rate
    Small Gaussian jitter is added so score/PR distributions are non-degenerate.
    """
    base = _sample_normal_rows(normal, n, rng)
    if attack == 'spoofing':
        lin = rng.normal(3.0, 0.3, n)
        ang = rng.normal(5.0, 0.5, n)
        rate = np.clip(rng.normal(2.0, 0.5, n), 0.5, None)
        delay = np.abs(rng.normal(base['delay'].mean(), base['delay'].std() + 1e-4, n))
    elif attack == 'flooding':
        lin = rng.normal(0.1, 0.02, n)
        ang = rng.normal(0.0, 0.02, n)
        rate = np.clip(rng.normal(500.0, 30.0, n), 100.0, None)
        delay = np.abs(rng.normal(base['delay'].mean(), base['delay'].std() + 1e-4, n))
    elif attack == 'delay':
        lin = base['linear_x'].to_numpy()          # real command, just stale
        ang = base['angular_z'].to_numpy()
        rate = np.clip(rng.normal(0.67, 0.15, n), 0.1, None)
        delay = np.clip(rng.normal(1.5, 0.1, n), 0.0, None)
    else:
        raise ValueError(f'unknown attack: {attack}')
    return pd.DataFrame({
        'linear_x': lin, 'angular_z': ang, 'publish_rate': rate, 'delay': delay,
    })


# --------------------------------------------------------------------------- #
# Dataset A: classification set
# --------------------------------------------------------------------------- #
def build_classification_df(csv_path=DEFAULT_NORMAL_CSV, n_per_attack=600, seed=42):
    rng = np.random.default_rng(seed)
    normal = load_normal(csv_path)

    frames = [normal.assign(label='normal')]
    for atk in ATTACK_TYPES:
        s = make_attack_samples(atk, n_per_attack, normal, rng)
        frames.append(s.assign(label=atk))
    df = pd.concat(frames, ignore_index=True)
    return df


# --------------------------------------------------------------------------- #
# Dataset B: temporal scenario (mirrors attack_manager_node scheduling)
# --------------------------------------------------------------------------- #
def build_temporal_df(csv_path=DEFAULT_NORMAL_CSV, hz=10.0, total_seconds=330.0,
                      interval=30.0, duration=5.0, seed=7):
    """A 10 Hz command stream: normal driving with periodic attack episodes.

    Every ``interval`` s an attack (cycled through ATTACK_TYPES) turns ON for
    ``duration`` s. Returns a time-ordered DataFrame with columns:
    t, linear_x, angular_z, publish_rate, delay, label, episode_id.
    """
    rng = np.random.default_rng(seed)
    normal = load_normal(csv_path)
    dt = 1.0 / hz
    n = int(total_seconds * hz)

    # base normal stream: walk through recorded driving, looping if needed
    base = _sample_normal_rows(normal, n, rng) if n > len(normal) else \
        normal.iloc[:n].reset_index(drop=True)

    rows = []
    for k in range(n):
        t = k * dt
        # is an attack active at time t?
        phase = t % interval
        active = phase < duration and t >= interval  # first interval is warmup
        if active:
            atk = ATTACK_TYPES[(int(t // interval) - 1) % len(ATTACK_TYPES)]
            ep_id = int(t // interval)
            one = make_attack_samples(atk, 1, normal, rng).iloc[0]
            rows.append((t, one['linear_x'], one['angular_z'], one['publish_rate'],
                         one['delay'], atk, ep_id))
        else:
            b = base.iloc[k]
            rows.append((t, b['linear_x'], b['angular_z'], b['publish_rate'],
                         b['delay'], 'normal', -1))
    return pd.DataFrame(
        rows,
        columns=['t', 'linear_x', 'angular_z', 'publish_rate', 'delay',
                 'label', 'episode_id'],
    )


if __name__ == '__main__':
    a = build_classification_df()
    b = build_temporal_df()
    print('classification set:', a.shape, dict(a['label'].value_counts()))
    print('temporal scenario :', b.shape,
          'episodes:', sorted(x for x in b['episode_id'].unique() if x >= 0))
