"""Stealth-attack robustness evaluation: detection rate vs attack strength.

The stock attacker nodes publish blatant values (500 msg/s, 1.5 s delay,
Twist(3.0, 5.0)), so ``evaluate`` reports 100 % recall. This module asks the
harder question -- *how weak can an attack be before the detector misses it?*

Each episode is a message-level replay of what ``security_node`` sees:

  * a contiguous slice of recorded normal driving (real inter-arrival times,
    real command values) arrives at /cmd_vel_stamped,
  * after ``WARMUP`` s an attacker runs for ``ATTACK_DURATION`` s with a given
    strength, publishing like the real nodes but with a tunable parameter,
  * every arrival goes through the deployed ``DetectionEngine.step`` with the
    publish rate counted exactly like ``security_node.update_publish_rate``
    (all arrivals in the last ``PUBLISH_RATE_WINDOW`` s).

An episode counts as detected when ``final_anomaly`` fires between attack
start and attack end + ``GRACE`` s *on an arrival the attack caused* (see
``run_episode``); alarms on ordinary teleop commands in the same window are
reported separately as the false-alarm rate. Strength 0 (no attack) is in
every sweep as the no-attack reference.

Run:
    ros2 run tb3_security stealth_eval
    # or, from the package dir:
    PYTHONPATH=. python3 -m tb3_security.stealth_eval --out eval_output/stealth

Outputs: stealth_detection.png, stealth_ramp.png, stealth_metrics.txt
"""

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # headless / no display needed
import matplotlib.pyplot as plt                      # noqa: E402

from tb3_security.detection_core import (             # noqa: E402
    DetectionEngine, DEFAULT_MODEL_PATH, PUBLISH_RATE_WINDOW,
    FLOOD_THRESHOLD, DELAY_THRESHOLD, VELOCITY_THRESHOLD,
    SPOOF_LINEAR_LIMIT, SPOOF_ANGULAR_LIMIT,
)
from tb3_security.safety_core import MAX_LINEAR, MAX_ANGULAR  # noqa: E402
from tb3_security.eval_dataset import DEFAULT_NORMAL_CSV     # noqa: E402

DEFAULT_OUT = os.path.expanduser(
    '~/graduation_ws/src/tb3_security/eval_output/stealth')

WARMUP = 15.0           # s of normal driving before the attack (fills windows)
ATTACK_DURATION = 5.0   # s, same as config/attack_scenario.yaml
GRACE = 1.0             # s after attack end still credited to the attack
EPISODES = 40           # episodes per strength level
SPOOF_HZ = 2.0          # abnormal_velocity_node timer (0.5 s)
RAMP_SECONDS = 20.0     # gradual spoofing: 0 -> RAMP_MAX over this long
RAMP_MAX = 1.5

# strength sweeps; the stock attacker value is marked in the figure
SWEEPS = {
    'flooding': {
        'levels': [0, 10, 20, 30, 35, 38, 40, 42, 45, 50, 60, 100, 500],
        'stock': 500, 'unit': 'attacker msg/s (on top of teleop)',
        'title': 'Flooding: extra Twist(0.1, 0) messages',
    },
    'delay': {
        'levels': [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0],
        'stock': 1.5, 'unit': 'command age (s)',
        'title': 'Delay: stale copies of real commands',
    },
    'spoof_linear': {
        'levels': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.8, 1.0, 2.0, 3.0],
        'stock': 3.0, 'unit': 'injected linear.x (m/s), 2 Hz',
        'title': 'Spoofing: Twist(v, 0)',
    },
    'spoof_angular': {
        'levels': [0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.84, 3.0, 4.0, 5.0],
        'stock': 5.0, 'unit': 'injected angular.z (rad/s), 2 Hz',
        'title': 'Spoofing: Twist(0, w)',
    },
}


# --------------------------------------------------------------------------- #
# Message streams
# --------------------------------------------------------------------------- #
def load_normal_stream(csv_path=DEFAULT_NORMAL_CSV):
    """Recorded normal driving as (dt, linear_x, angular_z, delay) arrays."""
    df = pd.read_csv(csv_path)
    t = df['timestamp'].to_numpy(dtype=float)
    dt = np.diff(t, prepend=t[0])
    # gaps between recording sessions would freeze the stream; cap at 0.5 s
    dt = np.clip(dt, 0.0, 0.5)
    return (dt, df['linear_x'].to_numpy(float), df['angular_z'].to_numpy(float),
            df['delay'].to_numpy(float))


def _teleop_messages(stream, rng, total):
    """Contiguous slice of recorded driving, restarted at t=0, ``total`` s long."""
    dt, lin, ang, dly = stream
    n = len(dt)
    i = int(rng.integers(0, n))
    t, out = 0.0, []
    while t < total:
        out.append((t, lin[i], ang[i], dly[i]))
        i = (i + 1) % n
        t += max(dt[i], 1e-3)
    return out


def _attack_messages(kind, strength, teleop, rng, t_on, t_off):
    """Extra / replaced arrivals produced by the attacker during [t_on, t_off)."""
    dly_pool = np.array([m[3] for m in teleop])
    msgs = []
    if kind == 'flooding' and strength > 0:
        # flood_attack_node -> /cmd_vel_input -> stamp_node: fresh, tiny delay
        n = int(round(strength * (t_off - t_on)))
        for k in range(n):
            t = t_on + (k + rng.random()) / strength
            msgs.append((t, 0.1, 0.0, float(rng.choice(dly_pool))))
    elif kind == 'delay' and strength > 0:
        # delay attacker re-publishes every real command ``strength`` s late
        # (non-blocking, so its rate matches teleop); originals still arrive
        for t, lin, ang, d in teleop:
            if t_on <= t < t_off:
                msgs.append((t + strength, lin, ang, d + strength))
    elif kind in ('spoof_linear', 'spoof_angular') and strength > 0:
        lin, ang = (strength, 0.0) if kind == 'spoof_linear' else (0.0, strength)
        t = t_on + rng.random() / SPOOF_HZ
        while t < t_off:
            msgs.append((t, lin, ang, float(rng.choice(dly_pool))))
            t += 1.0 / SPOOF_HZ
    elif kind == 'ramp':
        t = t_on + rng.random() / SPOOF_HZ
        while t < t_off:
            v = RAMP_MAX * (t - t_on) / (t_off - t_on)
            msgs.append((t, v, 0.0, float(rng.choice(dly_pool))))
            t += 1.0 / SPOOF_HZ
    return msgs


def run_episode(engine, stream, kind, strength, rng, duration=ATTACK_DURATION):
    """Replay one episode.

    Returns (detected, latency_s, value_at_detection, false_alarm). ``detected``
    only counts alarms *caused by the attack*: the triggering arrival is an
    attacker message (spoofing), or the alarm is typed as the attack itself
    (flooding / delay; a flood raises the rate of every arrival).
    Alarms on ordinary teleop commands
    (legit sharp turns) are ``false_alarm`` instead, so the false-alarm floor
    does not inflate the weak-attack points. The replay does not stop on a
    false alarm (the real node would sit in safe mode), so it measures what
    the detector *can* catch.
    """
    t_on, t_off = WARMUP, WARMUP + duration
    total = t_off + GRACE + 1.0
    teleop = _teleop_messages(stream, rng, total)
    attack = _attack_messages(kind, strength, teleop, rng, t_on, t_off)
    # (time, lin, ang, delay, is_attack) sorted by arrival
    arrivals = sorted([m + (False,) for m in teleop] + [m + (True,) for m in attack])

    engine.reset_history()
    window = []
    false_alarm = False
    for t, lin, ang, dly, is_atk in arrivals:
        window.append(t)
        window = [w for w in window if t - w < PUBLISH_RATE_WINDOW]
        rate = len(window) / PUBLISH_RATE_WINDOW
        r = engine.step((lin, ang, rate, dly), t)
        if not r['final_anomaly'] or not (t_on <= t <= t_off + GRACE):
            continue
        # flood / delay must be caught *as* that attack: a stale or crowded copy of
        # a legit sharp turn tripping the velocity path is not delay detection
        caused = r['attack_type'] == kind if kind in ('flooding', 'delay') else is_atk
        if caused:
            return True, t - t_on, (lin if kind == 'ramp' else None), false_alarm
        false_alarm = True
    return False, None, None, false_alarm


def sweep(engine, stream, kind, levels, episodes, seed):
    rows = []
    for lv in levels:
        rng = np.random.default_rng([seed, int(round(lv * 1000))])
        hits, fas, lats = 0, 0, []
        for _ in range(episodes):
            det, lat, _, fa = run_episode(engine, stream, kind, lv, rng)
            hits += det
            fas += fa
            if det:
                lats.append(lat)
        rows.append({
            'attack': kind, 'strength': lv, 'detection_rate': hits / episodes,
            'false_alarm_rate': fas / episodes,
            'median_latency': float(np.median(lats)) if lats else np.nan,
        })
    return pd.DataFrame(rows)


def ramp_trials(engine, stream, episodes, seed):
    rng = np.random.default_rng([seed, 999])
    out = []
    for _ in range(episodes):
        det, lat, v, _ = run_episode(engine, stream, 'ramp', 0, rng, duration=RAMP_SECONDS)
        out.append({'detected': det, 'latency': lat, 'speed_at_detection': v})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _miss_boundary(df):
    """Largest strength that is still missed at least half the time (excluding 0)."""
    weak = df[(df.strength > 0) & (df.detection_rate < 0.5)]
    return float(weak.strength.max()) if len(weak) else None


def fig_detection(results):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    limits = {
        'flooding': (FLOOD_THRESHOLD, 'hard limit: total > %d msg/s' % FLOOD_THRESHOLD),
        'delay': (DELAY_THRESHOLD, 'hard limit: %.1f s' % DELAY_THRESHOLD),
        'spoof_linear': (VELOCITY_THRESHOLD, 'rule: |v| > %.1f (ML-gated)' % VELOCITY_THRESHOLD),
        'spoof_angular': (VELOCITY_THRESHOLD * 2,
                          'rule: |w| > %.1f (ML-gated)' % (VELOCITY_THRESHOLD * 2)),
    }
    clamps = {'spoof_linear': MAX_LINEAR, 'spoof_angular': MAX_ANGULAR}
    hard_spoof = {'spoof_linear': SPOOF_LINEAR_LIMIT, 'spoof_angular': SPOOF_ANGULAR_LIMIT}
    for ax, (kind, cfg) in zip(axes.flat, SWEEPS.items()):
        df = results[kind]
        x = np.arange(len(df))  # categorical x: levels are unevenly spaced
        rates = df.detection_rate.to_numpy(float)
        ax.plot(x, rates, 'o-', color='#E4572E', lw=1.8, label='attack detected')
        ax.plot(x, df.false_alarm_rate.to_numpy(float), 's:', color='#888', lw=1, ms=3,
                label='teleop false alarm in window')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{v:g}' for v in df.strength], fontsize=7)
        ax.set_ylim(-0.05, 1.08)
        ax.set_ylabel('detection rate')
        ax.set_xlabel(cfg['unit'], fontsize=8)
        ax.set_title(cfg['title'], fontsize=10)
        ax.grid(alpha=0.3)

        lim, lim_label = limits[kind]
        ax.axvline(np.interp(lim, df.strength.to_numpy(float), x), color='red', ls='--', lw=1,
                   label=lim_label)
        if kind in clamps:
            ax.axvspan(-0.5, np.interp(clamps[kind], df.strength.to_numpy(float), x), color='#999',
                       alpha=0.15, label='within robot clamp (%.2f)' % clamps[kind])
        stock_i = list(df.strength).index(cfg['stock'])
        ax.annotate('stock attack', (stock_i, rates[stock_i]),
                    xytext=(-10, -28), textcoords='offset points', fontsize=7,
                    ha='right', arrowprops=dict(arrowstyle='->', lw=0.7))
        for xi, (rate, lat) in enumerate(zip(rates, df.median_latency.to_numpy(float))):
            if rate > 0 and not np.isnan(lat) and df.strength.to_numpy()[xi] > 0:
                ax.text(xi, min(rate, 1.0) + 0.03, f'{lat:.2f}s', ha='center',
                        fontsize=6, color='#555')
        if kind in hard_spoof:
            ax.axvline(np.interp(hard_spoof[kind], df.strength.to_numpy(float), x),
                       color='#8B0000', ls='-', lw=1,
                       label='hard limit: %g' % hard_spoof[kind])
        ax.legend(loc='lower right' if kind in ('flooding', 'delay') else 'upper left',
                  fontsize=7)
    fig.suptitle('Stealth attacks: detection rate vs strength '
                 f'({EPISODES} episodes/point, {ATTACK_DURATION:g} s attack; '
                 'labels = median latency)',
                 fontsize=10)
    fig.tight_layout()
    return fig


def fig_ramp(ramp):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    v = ramp.speed_at_detection.dropna().to_numpy(float)
    if len(v):
        ax.hist(v, bins=np.linspace(0, RAMP_MAX, 31), color='#E4572E', alpha=0.8)
    ax.axvline(MAX_LINEAR, color='#555', ls=':', label=f'robot clamp {MAX_LINEAR} m/s')
    ax.axvline(VELOCITY_THRESHOLD, color='red', ls='--',
               label=f'velocity rule {VELOCITY_THRESHOLD} m/s')
    ax.set_xlabel('injected linear.x at first detection (m/s)')
    ax.set_ylabel('episodes')
    ax.set_title(f'Gradual spoofing 0 → {RAMP_MAX} m/s over {RAMP_SECONDS:g} s '
                 f'(detected {int(ramp.detected.sum())}/{len(ramp)})', fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def write_metrics(path, results, ramp, floor):
    with open(path, 'w') as f:
        f.write('# stealth-attack sweep: detection rate / median latency (s)\n')
        f.write(f'# {EPISODES} episodes per point, attack {ATTACK_DURATION} s, '
                f'grace {GRACE} s, warmup {WARMUP} s\n')
        f.write(f'# teleop false-alarm chance per window (no attack): {floor:.3f}\n\n')
        for kind, df in results.items():
            f.write(f'[{kind}]  missed>=50% up to: {_miss_boundary(df)}\n')
            for _, r in df.iterrows():
                lat = '-' if np.isnan(r.median_latency) else f'{r.median_latency:.2f}'
                f.write(f'  {r.strength:>7g}\tdet {r.detection_rate:.3f}\t'
                        f'false-alarm {r.false_alarm_rate:.3f}\tlatency {lat}\n')
            f.write('\n')
        v = ramp.speed_at_detection.dropna()
        f.write(f'[ramp 0->{RAMP_MAX} m/s over {RAMP_SECONDS}s]  detected '
                f'{int(ramp.detected.sum())}/{len(ramp)}')
        if len(v):
            f.write(f', speed at detection median {v.median():.2f} '
                    f'(min {v.min():.2f}, max {v.max():.2f})')
        f.write('\n')


def main(args=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--csv', default=DEFAULT_NORMAL_CSV)
    p.add_argument('--model', default=DEFAULT_MODEL_PATH)
    p.add_argument('--out', default=DEFAULT_OUT)
    p.add_argument('--episodes', type=int, default=EPISODES)
    p.add_argument('--seed', type=int, default=11)
    ns = p.parse_args(args)

    # the periodic dashboard-only ML pass never changes final_anomaly (see
    # DetectionEngine.step), so disable it: same decisions, ~50x faster replay
    engine = DetectionEngine(model_path=ns.model, ml_monitor_period=float('inf'))
    if not engine.ml_enabled:
        raise SystemExit(f'ML model not found at {ns.model}')
    stream = load_normal_stream(ns.csv)
    os.makedirs(ns.out, exist_ok=True)

    results = {}
    for kind, cfg in SWEEPS.items():
        results[kind] = sweep(engine, stream, kind, cfg['levels'], ns.episodes, ns.seed)
        print(kind, results[kind][['strength', 'detection_rate']].to_dict('list'))
    ramp = ramp_trials(engine, stream, ns.episodes, ns.seed)
    floor = float(np.mean([df.false_alarm_rate.iloc[0] for df in results.values()]))

    fig_detection(results).savefig(os.path.join(ns.out, 'stealth_detection.png'), dpi=150)
    fig_ramp(ramp).savefig(os.path.join(ns.out, 'stealth_ramp.png'), dpi=150)
    write_metrics(os.path.join(ns.out, 'stealth_metrics.txt'), results, ramp, floor)
    print(f'wrote {ns.out}')


if __name__ == '__main__':
    main()
