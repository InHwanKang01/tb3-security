"""Train an Isolation Forest on normal driving data and dump the model.

The model is a sklearn Pipeline(StandardScaler -> IsolationForest) so that the
four features (linear_x, angular_z, publish_rate, delay) — which have very
different scales — contribute fairly to the split.

Run directly:
    python3 -m tb3_security.train_model \
        --csv data/normal_driving.csv \
        --out models/isolation_forest.pkl

Or via the console script (after `colcon build`):
    ros2 run tb3_security train_model --csv ... --out ...
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = ['linear_x', 'angular_z', 'publish_rate', 'delay']
DEFAULT_CSV = os.path.expanduser(
    '~/graduation_ws/src/tb3_security/data/normal_driving.csv')
DEFAULT_OUT = os.path.expanduser(
    '~/graduation_ws/src/tb3_security/models/isolation_forest.pkl')


def build_model(n_estimators, contamination, random_state):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('iforest', IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )),
    ])


def main():
    parser = argparse.ArgumentParser(
        description='Train Isolation Forest on normal driving data.')
    parser.add_argument('--csv', default=DEFAULT_CSV, help='Input CSV path.')
    parser.add_argument('--out', default=DEFAULT_OUT, help='Output .pkl path.')
    parser.add_argument('--contamination', type=float, default=0.01,
                        help='Expected anomaly ratio in training data (default: 0.01).')
    parser.add_argument('--estimators', type=int, default=200,
                        help='n_estimators for IsolationForest (default: 200).')
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f'CSV not found: {args.csv}', file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        print(f'CSV missing columns: {missing}', file=sys.stderr)
        sys.exit(1)

    df = df.dropna(subset=FEATURE_COLUMNS)
    X = df[FEATURE_COLUMNS].astype(float).values
    if len(X) < 100:
        print(f'Warning: only {len(X)} samples — model may underfit. '
              f'Collect more normal driving data for better results.',
              file=sys.stderr)

    print(f'Training Isolation Forest on {len(X)} samples, features={FEATURE_COLUMNS}')
    print(f'  contamination={args.contamination}, n_estimators={args.estimators}')

    model = build_model(args.estimators, args.contamination, args.random_state)
    model.fit(X)

    # 학습 데이터에 대한 간단한 정합성 리포트
    preds = model.predict(X)
    n_anom = int(np.sum(preds == -1))
    print(f'  flagged in training data: {n_anom}/{len(X)} '
          f'({100.0 * n_anom / max(len(X), 1):.2f}%)')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    joblib.dump(model, args.out)
    print(f'Model saved → {args.out}')


if __name__ == '__main__':
    main()
