"""Offline evaluation of the tb3_security detection pipeline.

Replays a labelled dataset through the very same ``DetectionEngine`` that runs
live in ``security_node`` (no robot, no ROS) and produces poster-ready static
figures plus a single combined PDF report.

Run:
    ros2 run tb3_security evaluate
    # or, from the package dir:
    PYTHONPATH=. python3 -m tb3_security.evaluate --out eval_output

Outputs (default ``src/tb3_security/eval_output/``):
    figures/*.png          individual figures
    evaluation_report.pdf  all figures + summary in one document
    metrics.txt            machine/human readable metric dump
"""

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # headless / no display needed
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Patch                  # noqa: E402

from sklearn.metrics import (                         # noqa: E402
    confusion_matrix, precision_recall_fscore_support,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
from sklearn.preprocessing import StandardScaler      # noqa: E402
from sklearn.decomposition import PCA                 # noqa: E402

from tb3_security.detection_core import (             # noqa: E402
    DetectionEngine, FEATURE_COLUMNS, DEFAULT_MODEL_PATH,
)
from tb3_security import eval_dataset                 # noqa: E402

CLASSES = ['normal', 'spoofing', 'flooding', 'delay']
ATTACKS = ['spoofing', 'flooding', 'delay']
CLASS_COLORS = {'normal': '#4C9BE0', 'spoofing': '#E4572E',
                'flooding': '#F3A712', 'delay': '#8E44AD'}
RECOVERY_TIME = 10.0  # mirrors security_node.RECOVERY_TIME (safe-mode dwell)

DEFAULT_OUT = os.path.expanduser('~/graduation_ws/src/tb3_security/eval_output')


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
def _feat(row):
    return (float(row['linear_x']), float(row['angular_z']),
            float(row['publish_rate']), float(row['delay']))


def run_classification(df, engine):
    """Per-sample replay (samples are i.i.d.; sliding history reset each time).

    Records every sub-decision so combined / rule-only / ML-only gates can be
    compared from one pass.
    """
    out = []
    for _, row in df.iterrows():
        engine.reset_history()  # i.i.d. samples -> no cross-sample window
        f = _feat(row)
        r = engine.step(f, 0.0, full=True)
        # single-message decision: i.i.d. samples can't satisfy the temporal
        # 2-in-1 s confirmation, so use the unconfirmed gate here; its effect on
        # false alarms is measured by normal_replay_false_alarms()
        combined = r['hard_limit_anomaly'] or r['gated_raw_anomaly']
        # previous gate: ML AND (sliding OR any threshold) -- before/after baseline
        and_gate = r['ml_anomaly'] and (r['sliding_anomaly'] or r['threshold_anomaly'])
        rule_only = r['sliding_anomaly'] or r['threshold_anomaly']
        ml_only = r['ml_anomaly']
        out.append({
            'label': row['label'],
            'ml_score': r['ml_score'],
            'pred_combined': (r['attack_type'] if r['final_anomaly'] else 'spoofing')
            if combined else 'normal',
            'det_combined': combined,
            'det_and': and_gate,
            'det_rule': rule_only,
            'det_ml': ml_only,
        })
    return pd.DataFrame(out)


def run_temporal(df, engine):
    """Time-ordered replay (sliding window active). Returns per-sample results."""
    engine.reset_history()
    out = []
    for _, row in df.iterrows():
        f = _feat(row)
        r = engine.step(f, float(row['t']), full=True)
        out.append({
            't': float(row['t']),
            'label': row['label'],
            'episode_id': int(row['episode_id']),
            'publish_rate': float(row['publish_rate']),
            'linear_x': float(row['linear_x']),
            'angular_z': float(row['angular_z']),
            'delay': float(row['delay']),
            'ml_score': r['ml_score'],
            'final': r['final_anomaly'],
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def gate_recall_fpr(res):
    """Per-attack recall and normal false-positive rate for each gate variant."""
    stats = {}
    for gate, col in [('Hybrid (deployed)', 'det_combined'),
                      ('ML AND Rule (previous)', 'det_and'),
                      ('Rule-only', 'det_rule'),
                      ('ML-only', 'det_ml')]:
        rec = {a: float(res[res.label == a][col].mean()) for a in ATTACKS}
        fpr = float(res[res.label == 'normal'][col].mean())
        stats[gate] = {'recall': rec, 'fpr': fpr}
    return stats


def normal_replay_false_alarms(csv_path, engine):
    """Replay the recorded normal driving in order with its real timestamps.

    Publish rate is recounted per arrival like security_node. Returns
    (duration_s, n_final_alarms, n_unconfirmed_gate_hits): the last one is what
    the deployed gate would have alarmed on without the 2-in-1 s confirmation.
    """
    df = pd.read_csv(csv_path)
    t = df['timestamp'].to_numpy(float)
    t = np.concatenate([[0.0], np.cumsum(np.clip(np.diff(t), 0.0, 0.5))])
    engine.reset_history()
    window, finals, raws = [], 0, 0
    for ti, lin, ang, dly in zip(t, df['linear_x'], df['angular_z'], df['delay']):
        window = [w for w in window if ti - w < 1.0] + [ti]
        r = engine.step((float(lin), float(ang), float(len(window)), float(dly)), ti)
        finals += r['final_anomaly']
        raws += r['hard_limit_anomaly'] or r['gated_raw_anomaly']
    return float(t[-1]), finals, raws


def latency_per_episode(temporal_df, interval=30.0):
    """Time from each attack episode's start to its first confirmed detection."""
    rows = []
    for ep in sorted(e for e in temporal_df['episode_id'].unique() if e >= 0):
        seg = temporal_df[temporal_df.episode_id == ep]
        atk = seg['label'].iloc[0]
        start = interval * ep
        hits = seg[seg['final']]
        if len(hits):
            lat = float(hits['t'].min() - start)
        else:
            lat = None  # missed
        rows.append({'episode': ep, 'attack': atk, 'latency': lat})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_confusion(res):
    y_true = res['label'].tolist()
    y_pred = res['pred_combined'].tolist()
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix — deployed hybrid gate\n(row-normalised)')
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f'{cmn[i, j]:.2f}\n({cm[i, j]})',
                    ha='center', va='center',
                    color='white' if cmn[i, j] > 0.5 else 'black', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='fraction of true class')
    fig.tight_layout()
    return fig


def fig_per_class_metrics(res):
    y_true = res['label'].tolist()
    y_pred = res['pred_combined'].tolist()
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=ATTACKS, zero_division=0)
    x = np.arange(len(ATTACKS)); w = 0.26
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.bar(x - w, p, w, label='Precision', color='#4C9BE0')
    ax.bar(x, r, w, label='Recall', color='#E4572E')
    ax.bar(x + w, f1, w, label='F1', color='#2E933C')
    for i in range(len(ATTACKS)):
        for off, val in [(-w, p[i]), (0, r[i]), (w, f1[i])]:
            ax.text(x[i] + off, val + 0.02, f'{val:.2f}', ha='center', fontsize=8)
    ax.set_xticks(x, ATTACKS); ax.set_ylim(0, 1.15)
    ax.set_ylabel('score'); ax.set_title('Per-attack detection quality (deployed gate)')
    ax.legend(loc='upper right', ncol=3)
    ax.axhline(1.0, color='gray', ls=':', lw=0.8)
    fig.tight_layout()
    return fig


def fig_gate_comparison(res):
    stats = gate_recall_fpr(res)
    gates = list(stats.keys())
    x = np.arange(len(ATTACKS)); w = 0.2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6),
                                   gridspec_kw={'width_ratios': [3, 1.3]})
    colors = ['#2E933C', '#9AA5B1', '#E4572E', '#4C9BE0']
    short = ['Hybrid', 'AND(prev)', 'Rule', 'ML']
    mid = (len(gates) - 1) / 2
    for gi, g in enumerate(gates):
        vals = [stats[g]['recall'][a] for a in ATTACKS]
        ax1.bar(x + (gi - mid) * w, vals, w, label=g, color=colors[gi])
        for a_i, v in enumerate(vals):
            ax1.text(x[a_i] + (gi - mid) * w, v + 0.02, f'{v:.2f}',
                     ha='center', fontsize=7)
    ax1.set_xticks(x, ATTACKS); ax1.set_ylim(0, 1.18)
    ax1.set_ylabel('Recall (detection rate)')
    ax1.set_title('Detection rate per attack — gate comparison')
    ax1.legend(loc='upper center', ncol=4, fontsize=8)

    fprs = [stats[g]['fpr'] for g in gates]
    ax2.bar(range(len(gates)), fprs, color=colors)
    for i, v in enumerate(fprs):
        ax2.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=8)
    ax2.set_xticks(range(len(gates)), short, rotation=15)
    ax2.set_ylabel('False-positive rate (normal)')
    ax2.set_title('FPR on normal driving')
    ax2.set_ylim(0, max(fprs) * 1.3 + 0.02)
    fig.suptitle('Hybrid gate: hard limits bypass the ML veto (flooding fixed), '
                 'ML still suppresses sharp-turn false alarms', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def fig_score_distribution(res, engine):
    offset = None
    try:
        offset = float(engine.model.named_steps['iforest'].offset_)
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    bins = np.linspace(res['ml_score'].min(), res['ml_score'].max(), 60)
    for cls in CLASSES:
        s = res[res.label == cls]['ml_score']
        ax.hist(s, bins=bins, alpha=0.55, label=f'{cls} (n={len(s)})',
                color=CLASS_COLORS[cls], density=True)
    if offset is not None:
        ax.axvline(offset, color='k', ls='--', lw=1.5,
                   label=f'decision threshold ({offset:.3f})')
    ax.set_xlabel('IsolationForest score_samples  (higher = more normal)')
    ax.set_ylabel('density')
    ax.set_title('ML anomaly-score distribution by class\n'
                 '(flooding sits on the "normal" side — the blind spot)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_roc_pr(res):
    y = (res['label'] != 'normal').astype(int).to_numpy()
    score = (-res['ml_score']).to_numpy()  # higher = more anomalous
    fpr, tpr, _ = roc_curve(y, score)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y, score)
    ap = average_precision_score(y, score)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    ax1.plot(fpr, tpr, color='#E4572E', lw=2, label=f'AUC = {roc_auc:.3f}')
    ax1.plot([0, 1], [0, 1], 'k:', lw=1)
    ax1.set_xlabel('False-positive rate'); ax1.set_ylabel('True-positive rate')
    ax1.set_title('ROC — ML anomaly score (attack vs normal)')
    ax1.legend(loc='lower right')
    ax2.plot(rec, prec, color='#4C9BE0', lw=2, label=f'AP = {ap:.3f}')
    ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall — ML anomaly score')
    ax2.legend(loc='lower left')
    fig.suptitle('Discriminative power of the ML stage alone', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def fig_feature_space(df):
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)
    pcs = PCA(n_components=2, random_state=0).fit(Xs)
    Z = pcs.transform(Xs)
    fig, ax = plt.subplots(figsize=(7, 5.4))
    for cls in CLASSES:
        m = (df['label'] == cls).to_numpy()
        ax.scatter(Z[m, 0], Z[m, 1], s=10, alpha=0.5,
                   c=CLASS_COLORS[cls], label=cls, edgecolors='none')
    ev = pcs.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({ev[0]*100:.0f}% var)')
    ax.set_ylabel(f'PC2 ({ev[1]*100:.0f}% var)')
    ax.set_title('Feature space (PCA of standardised features)')
    ax.legend()
    fig.tight_layout()
    return fig


def fig_timeline(temporal, interval=30.0, duration=5.0):
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={'height_ratios': [2, 2, 0.6]})
    t = temporal['t'].to_numpy()

    # attack episode bands
    eps = sorted(e for e in temporal['episode_id'].unique() if e >= 0)
    seen = set()
    for ep in eps:
        atk = temporal[temporal.episode_id == ep]['label'].iloc[0]
        start = interval * ep
        for ax in axes[:2]:
            ax.axvspan(start, start + duration, color=CLASS_COLORS[atk],
                       alpha=0.18, label=atk if atk not in seen else None)
        seen.add(atk)

    # panel 1: publish_rate (log) with flood threshold
    axes[0].semilogy(t, temporal['publish_rate'].clip(lower=0.1).to_numpy(),
                     lw=0.8, color='#333')
    axes[0].axhline(50, color='red', ls='--', lw=1, label='flood threshold (50/s)')
    axes[0].set_ylabel('publish_rate (msg/s, log)')
    axes[0].legend(loc='upper right', fontsize=7, ncol=2)
    axes[0].set_title('Temporal scenario — attacks every %ds for %ds '
                      '(bands = ground-truth attack)' % (interval, duration))

    # panel 2: |velocity| + delay, with detection markers
    axes[1].plot(t, temporal['linear_x'].abs().to_numpy(), lw=0.7, color='#4C9BE0',
                 label='|linear_x|')
    axes[1].plot(t, temporal['angular_z'].abs().to_numpy(), lw=0.7, color='#8E44AD',
                 label='|angular_z|')
    det = temporal[temporal['final']]
    axes[1].scatter(det['t'].to_numpy(), np.full(len(det), 0.05), marker='v', s=30,
                    color='red', zorder=5, label='detection')
    axes[1].set_ylabel('|velocity|')
    axes[1].legend(loc='upper right', fontsize=7, ncol=3)

    # panel 3: response strip (NORMAL vs SAFE mode)
    mode_safe = np.zeros(len(t), dtype=bool)
    safe_until = -1.0
    finals = temporal['final'].to_numpy()
    for i, ti in enumerate(t):
        if ti < safe_until:
            mode_safe[i] = True
        elif finals[i]:
            safe_until = ti + RECOVERY_TIME
            mode_safe[i] = True
    axes[2].fill_between(t, 0, 1, where=~mode_safe, color='#2E933C', alpha=0.6,
                         step='mid')
    axes[2].fill_between(t, 0, 1, where=mode_safe, color='#E4572E', alpha=0.7,
                         step='mid')
    axes[2].set_yticks([]); axes[2].set_xlabel('time (s)')
    axes[2].set_ylabel('mode', rotation=0, ha='right', va='center')
    axes[2].legend(handles=[Patch(color='#2E933C', alpha=0.6, label='NORMAL'),
                            Patch(color='#E4572E', alpha=0.7, label='SAFE mode')],
                   loc='upper right', ncol=2, fontsize=7)
    fig.tight_layout()
    return fig


def fig_latency(lat_df):
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    xs = range(len(lat_df))
    colors = [CLASS_COLORS[a] for a in lat_df['attack']]
    heights = [0 if pd.isna(l) else l for l in lat_df['latency']]
    ax.bar(xs, heights, color=colors)
    for i, (_, r) in enumerate(lat_df.iterrows()):
        if pd.isna(r['latency']):
            ax.text(i, 0.02, 'MISSED', rotation=90, ha='center', va='bottom',
                    color='red', fontsize=8, fontweight='bold')
        else:
            ax.text(i, r['latency'] + 0.01, f"{r['latency']:.2f}s",
                    ha='center', fontsize=8)
    ax.set_xticks(list(xs), [f"#{r.episode}\n{r.attack}"
                             for _, r in lat_df.iterrows()], fontsize=8)
    ax.set_ylabel('detection latency (s)')
    ax.set_title('Time-to-detection per attack episode  '
                 '(0-height + MISSED = never detected)')
    handles = [Patch(color=CLASS_COLORS[a], label=a) for a in ATTACKS]
    ax.legend(handles=handles, fontsize=8)
    fig.tight_layout()
    return fig


def fig_summary(res, lat_df, stats):
    """Text summary page for the PDF."""
    y_true = res['label'].tolist(); y_pred = res['pred_combined'].tolist()
    acc = float(np.mean([a == b for a, b in zip(y_true, y_pred)]))
    fpr_c = stats['Hybrid (deployed)']['fpr']
    fpr_prev = stats['ML AND Rule (previous)']['fpr']
    lines = [
        'tb3_security — Offline Detection Evaluation',
        '=' * 52,
        '',
        f'Classification samples : {len(res)}  '
        f'(normal={int((res.label=="normal").sum())}, '
        f'attacks={int((res.label!="normal").sum())})',
        f'Overall accuracy       : {acc:.3f}',
        f'Normal false-pos rate  : {fpr_c:.3f}  (deployed hybrid gate; '
        f'previous AND gate {fpr_prev:.3f})',
        '',
        'Per-attack recall (detection rate):',
    ]
    for a in ATTACKS:
        c = stats['Hybrid (deployed)']['recall'][a]
        pv = stats['ML AND Rule (previous)']['recall'][a]
        ru = stats['Rule-only']['recall'][a]
        lines.append(f'   {a:9}  deployed={c:.2f}   previous-AND={pv:.2f}   '
                     f'rule-only={ru:.2f}')
    lines += [
        '',
        'Detection latency (temporal scenario):',
    ]
    for _, r in lat_df.iterrows():
        lat = 'MISSED' if pd.isna(r['latency']) else f"{r['latency']:.2f}s"
        lines.append(f'   episode #{r.episode:<2} {r.attack:9} -> {lat}')
    lines += [
        '',
        'Key finding:',
        '  * Previous gate  ML AND (sliding OR threshold)  missed flooding:',
        '    the IsolationForest score is flat in publish_rate (trained on',
        '    rate 1-14), so ML=normal vetoed the correct rule alarm.',
        '  * Fix (deployed):  hard_limit OR (ML AND (sliding OR velocity)).',
        '    Flood rate / delay are unambiguous and bypass the ML veto;',
        '    ML still gates the velocity check, suppressing false alarms on',
        '    legitimate sharp turns (angular_z up to 2.84).',
    ]
    fig = plt.figure(figsize=(8.5, 6.5))
    fig.text(0.06, 0.95, '\n'.join(lines), va='top', ha='left',
             family='monospace', fontsize=9.5)
    fig.patch.set_facecolor('white')
    return fig, acc


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main(args=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default=DEFAULT_MODEL_PATH)
    ap.add_argument('--normal-csv', default=eval_dataset.DEFAULT_NORMAL_CSV)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--n-per-attack', type=int, default=600)
    ns = ap.parse_args(args)

    fig_dir = os.path.join(ns.out, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    print('[eval] loading model + building datasets ...')
    engine = DetectionEngine(model_path=ns.model)
    if not engine.ml_enabled:
        print('[eval] WARNING: ML model not loaded — combined gate degrades to '
              'no detections. Train models/isolation_forest.pkl first.')
    clf_df = eval_dataset.build_classification_df(
        ns.normal_csv, n_per_attack=ns.n_per_attack)
    tmp_df = eval_dataset.build_temporal_df(ns.normal_csv)

    print('[eval] replaying through DetectionEngine ...')
    res = run_classification(clf_df, engine)
    temporal = run_temporal(tmp_df, engine)
    lat_df = latency_per_episode(temporal)
    stats = gate_recall_fpr(res)
    replay_s, replay_fa, replay_raw = normal_replay_false_alarms(ns.normal_csv, engine)

    print('[eval] rendering figures ...')
    figures = {
        '1_confusion_matrix': fig_confusion(res),
        '2_per_class_metrics': fig_per_class_metrics(res),
        '3_gate_comparison': fig_gate_comparison(res),
        '4_score_distribution': fig_score_distribution(res, engine),
        '5_roc_pr': fig_roc_pr(res),
        '6_feature_space_pca': fig_feature_space(clf_df),
        '7_detection_timeline': fig_timeline(temporal),
        '8_detection_latency': fig_latency(lat_df),
    }
    summary_fig, acc = fig_summary(res, lat_df, stats)

    for name, fig in figures.items():
        path = os.path.join(fig_dir, name + '.png')
        fig.savefig(path, dpi=140, bbox_inches='tight')
    summary_fig.savefig(os.path.join(fig_dir, '0_summary.png'),
                        dpi=140, bbox_inches='tight')

    pdf_path = os.path.join(ns.out, 'evaluation_report.pdf')
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(summary_fig)
        for name in sorted(figures):
            pdf.savefig(figures[name])
    plt.close('all')

    # text dump
    with open(os.path.join(ns.out, 'metrics.txt'), 'w') as fh:
        fh.write(f'overall_accuracy\t{acc:.4f}\n')
        for g, d in stats.items():
            fh.write(f'[{g}] fpr_normal\t{d["fpr"]:.4f}\n')
            for a in ATTACKS:
                fh.write(f'[{g}] recall_{a}\t{d["recall"][a]:.4f}\n')
        fh.write(f'\nnormal_replay ({replay_s:.0f} s of recorded driving, in order)\n'
                 f'  false_alarms\t{replay_fa}\n'
                 f'  without_2in1s_confirmation\t{replay_raw}\n')
        fh.write('\nlatency_per_episode\n')
        for _, r in lat_df.iterrows():
            fh.write(f'  ep{r.episode}\t{r.attack}\t'
                     f'{"MISSED" if pd.isna(r.latency) else f"{r.latency:.3f}s"}\n')

    print(f'[eval] done.\n  PDF     : {pdf_path}\n  figures : {fig_dir}/\n'
          f'  metrics : {os.path.join(ns.out, "metrics.txt")}')
    print(f'[eval] normal replay {replay_s:.0f} s: false alarms {replay_fa} '
          f'(without confirmation {replay_raw})')
    print(f'[eval] accuracy={acc:.3f}  '
          f'flooding recall (deployed)={stats["Hybrid (deployed)"]["recall"]["flooding"]:.2f}')


if __name__ == '__main__':
    main()
