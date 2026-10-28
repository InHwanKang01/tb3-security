import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
from tf2_ros import Buffer, TransformListener
from flask import Flask, render_template_string, Response, jsonify
import threading
import cv2
import numpy as np
import math
import time
from collections import deque

app = Flask(__name__)

# 최신 상태 저장 ----------------------------------------------------- #
current_status = {
    'mode': 'UNKNOWN',
    'front': '0.00',
    'back': '0.00',
    'left': '0.00',
    'right': '0.00',
    'wall': 'None',
    'detection': '',
    'msg_rate': '0',
    'ml_anomaly': '0',
    'ml_score': '0',
    'linear_x': '0',
    'angular_z': '0',
    'publish_rate': '0',
    'delay': '0',
    'attack_type': 'none',
}

# Feature 히스토리 (최근 30초) -------------------------------------- #
HISTORY_WINDOW = 30.0
feature_history = deque()  # (ts, linear_x, angular_z, publish_rate, delay)
history_lock = threading.Lock()

# 이벤트 로그 (최대 100건) ------------------------------------------ #
event_log = deque(maxlen=100)
event_lock = threading.Lock()

# 이전 감지 상태 (rising edge 검출용) ------------------------------ #
_last_detection_text = ''
_last_ml_anomaly = False

# 맵 -------------------------------------------------------------- #
latest_map_image = None
map_lock = threading.Lock()
map_meta = {
    'resolution': 0.05,
    'origin_x': 0.0,
    'origin_y': 0.0,
    'width': 0,
    'height': 0
}

# 로봇 위치 캐시 ---------------------------------------------------- #
robot_pose_cache = {
    'x': None,
    'y': None,
    'yaw_deg': None,
    'timestamp': 0.0,
    'valid': False
}
POSE_STALE_SEC = 3.0


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>TurtleBot3 Security Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root {
            --bg: #f5f7fa;
            --panel: #ffffff;
            --text: #1e293b;
            --muted: #64748b;
            --border: #e2e8f0;
            --blue: #2563eb;
            --green: #10b981;
            --red: #ef4444;
            --orange: #f59e0b;
            --shadow: 0 2px 8px rgba(0,0,0,0.04);
        }
        body {
            font-family: 'Helvetica Neue', Arial, sans-serif;
            background: var(--bg); color: var(--text); padding: 20px;
        }
        h1 { text-align: center; margin-bottom: 20px; color: var(--blue); font-size: 26px; }
        .hint { text-align: center; color: var(--muted); font-size: 12px; margin-bottom: 14px; }

        .main-layout {
            display: grid; grid-template-columns: 1fr 1fr; gap: 15px; align-items: start;
        }
        .left-panel { display: flex; flex-direction: column; gap: 15px; }

        .clickable { cursor: pointer; transition: transform 0.15s, box-shadow 0.15s; }
        .clickable:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(0,0,0,0.08); }

        .mode-box {
            text-align: center; padding: 22px; border-radius: 12px;
            font-size: 28px; font-weight: 700; color: white;
            box-shadow: var(--shadow); position: relative;
        }
        .mode-box .sub { font-size: 11px; font-weight: 500; opacity: 0.9; margin-top: 4px; }
        .mode-normal { background: linear-gradient(135deg, #10b981, #34d399); }
        .mode-safe   { background: linear-gradient(135deg, #ef4444, #f87171); animation: blink 1s infinite; }
        @keyframes blink { 50% { opacity: 0.78; } }

        /* 공격 유형 배지 */
        .attack-badge {
            display: flex; align-items: center; gap: 14px;
            padding: 14px 18px; border-radius: 10px;
            font-weight: 700; box-shadow: var(--shadow);
            border-left: 6px solid transparent;
            transition: background 0.25s, border-color 0.25s, color 0.25s;
        }
        .attack-badge .ab-tag {
            font-size: 11px; padding: 4px 8px; border-radius: 4px;
            background: rgba(255,255,255,0.6); letter-spacing: 0.5px;
        }
        .attack-badge .ab-title { font-size: 18px; }
        .attack-badge .ab-desc  { font-size: 12px; font-weight: 500; opacity: 0.85; margin-top: 2px; }
        .attack-badge.none {
            background: #f1f5f9; color: #64748b; border-color: #cbd5e1;
        }
        .attack-badge.flooding {
            background: #dbeafe; color: #1d4ed8; border-color: #1d4ed8;
            animation: pulse 1.2s infinite;
        }
        .attack-badge.spoofing {
            background: #fef3c7; color: #92400e; border-color: #d97706;
            animation: pulse 1.2s infinite;
        }
        .attack-badge.delay {
            background: #fee2e2; color: #991b1b; border-color: #b91c1c;
            animation: pulse 1.2s infinite;
        }
        @keyframes pulse {
            0%, 100% { box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
            50%      { box-shadow: 0 2px 14px rgba(239,68,68,0.35); }
        }

        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .card {
            background: var(--panel); border-radius: 10px; padding: 18px;
            border-left: 4px solid var(--blue); box-shadow: var(--shadow);
        }
        .card h3 { color: var(--muted); margin-bottom: 8px; font-size: 11px; letter-spacing: 0.5px; text-transform: uppercase; }
        .card .value { font-size: 24px; font-weight: 700; color: var(--text); }

        .lidar-box {
            background: var(--panel); border-radius: 10px; padding: 20px;
            text-align: center; box-shadow: var(--shadow);
        }
        .lidar-box h3 { color: var(--muted); margin-bottom: 12px; font-size: 11px; letter-spacing: 0.5px; text-transform: uppercase; }
        .lidar-grid {
            display: grid; grid-template-columns: 1fr 1fr 1fr; grid-template-rows: 1fr 1fr 1fr;
            gap: 8px; max-width: 300px; margin: 0 auto;
        }
        .lidar-cell {
            padding: 12px; border-radius: 8px; font-size: 13px; font-weight: 600;
            background: #eef2f7; color: var(--text);
        }
        .lidar-front { grid-column: 2; grid-row: 1; }
        .lidar-left  { grid-column: 1; grid-row: 2; }
        .lidar-robot { grid-column: 2; grid-row: 2; background: var(--blue); color: white; }
        .lidar-right { grid-column: 3; grid-row: 2; }
        .lidar-back  { grid-column: 2; grid-row: 3; }

        .detection-box {
            background: var(--panel); border-radius: 10px; padding: 18px;
            border-left: 4px solid var(--red); box-shadow: var(--shadow);
        }
        .detection-box h3 { color: var(--red); margin-bottom: 8px; font-size: 11px; letter-spacing: 0.5px; text-transform: uppercase; }
        .detection-text { font-size: 15px; color: var(--orange); font-weight: 500; }

        .map-box {
            background: var(--panel); border-radius: 10px; padding: 20px;
            border-left: 4px solid var(--green); position: sticky; top: 20px;
            box-shadow: var(--shadow);
        }
        .map-box h3 { color: var(--green); margin: 0; font-size: 13px; letter-spacing: 0.5px; text-transform: uppercase; }
        .map-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
        .robot-badge { font-size: 11px; padding: 4px 10px; border-radius: 20px; font-weight: 600; }
        .robot-badge.active  { background: var(--green); color: white; }
        .robot-badge.stale   { background: var(--orange); color: white; }
        .robot-badge.offline { background: #cbd5e1; color: #475569; }
        .map-box img {
            width: 80%; border-radius: 8px;
            border: 2px solid var(--green);
            image-rendering: pixelated; display: block; margin: 0 auto;
        }
        .map-offline { padding: 40px; color: var(--muted); font-size: 14px; text-align:center; }
        .pose-info {
            margin-top: 8px; font-size: 12px; color: var(--muted);
            display: flex; gap: 15px; justify-content: center;
        }
        .pose-info span { color: var(--blue); font-weight: 600; }

        .map-box.fullscreen {
            position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
            z-index: 1000; padding: 30px; border-radius: 0; overflow: auto;
            background: var(--panel);
        }
        .map-box.fullscreen img { width: 92%; max-height: 84vh; object-fit: contain; }
        .map-box.fullscreen::before {
            content: '✕ Click anywhere to exit fullscreen';
            display: block; text-align: right; color: var(--muted);
            font-size: 12px; margin-bottom: 8px;
        }

        .expand-panel {
            display: none;
            background: var(--panel); border-radius: 10px; padding: 18px;
            border-left: 4px solid var(--blue); box-shadow: var(--shadow);
        }
        .expand-panel.open { display: block; }
        .expand-panel h3 { color: var(--muted); font-size: 11px; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 10px; }

        /* ML 게이지 */
        .gauge {
            position: relative; height: 22px; background: #e2e8f0;
            border-radius: 11px; overflow: hidden; margin: 6px 0 12px 0;
        }
        .gauge-bar {
            position: absolute; top: 0; bottom: 0; left: 0;
            background: var(--green); transition: width 0.3s, background 0.3s;
        }
        .gauge-bar.anomaly { background: var(--red); }
        .gauge-label { font-size: 12px; color: var(--muted); display:flex; justify-content:space-between; }

        /* Feature 4개 */
        .feat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; }
        .feat-item {
            padding: 10px 12px; border-radius: 8px; background: #f8fafc;
            border: 1px solid var(--border); transition: background 0.2s, border-color 0.2s;
        }
        .feat-item.flagged { background: #fee2e2; border-color: var(--red); }
        .feat-item h4 { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
        .feat-item .v { font-size: 20px; font-weight: 700; color: var(--text); }
        .feat-item.flagged .v { color: var(--red); }
        .feat-item .r { font-size: 10px; color: var(--muted); margin-top: 4px; }

        /* 이벤트 로그 탭 */
        .log-tabs { display: flex; gap: 6px; margin-bottom: 10px; flex-wrap: wrap; }
        .log-tabs .tab {
            padding: 6px 12px; border: 1px solid var(--border);
            background: white; border-radius: 6px; cursor: pointer;
            font-size: 11px; font-weight: 600; color: var(--muted);
            font-family: inherit;
            transition: background 0.15s, color 0.15s, border-color 0.15s;
        }
        .log-tabs .tab:hover { background: #f1f5f9; }
        .log-tabs .tab.active                 { background: var(--blue);   color: white; border-color: var(--blue); }
        .log-tabs .tab.active[data-filter="flooding"] { background: #1d4ed8; border-color: #1d4ed8; }
        .log-tabs .tab.active[data-filter="spoofing"] { background: #d97706; border-color: #d97706; }
        .log-tabs .tab.active[data-filter="delay"]    { background: #b91c1c; border-color: #b91c1c; }
        .tab-count { margin-left: 6px; opacity: 0.85; font-size: 10px; }

        /* 이벤트 로그 */
        .event-log {
            max-height: 280px; overflow-y: auto;
            font-family: 'Courier New', monospace; font-size: 12px;
        }
        .event-row { padding: 6px 8px; border-bottom: 1px solid var(--border); }
        .event-row:last-child { border-bottom: none; }
        .event-row.empty { color: var(--muted); text-align: center; padding: 16px; }
        .event-row .tag { display: inline-block; padding: 2px 6px; border-radius: 3px; font-weight: 700; margin-right: 8px; color: white; font-size: 10px; }
        .event-row .tag.ML { background: var(--blue); }
        .event-row .tag.RULE { background: var(--orange); }
        .event-row .ts { color: var(--muted); margin-right: 6px; }
        .event-row .atype-tag {
            display: inline-block; padding: 1px 6px; border-radius: 3px;
            font-size: 9px; font-weight: 700; margin-right: 6px; letter-spacing: 0.3px;
        }
        .event-row .atype-tag.flooding { background: #dbeafe; color: #1d4ed8; }
        .event-row .atype-tag.spoofing { background: #fef3c7; color: #92400e; }
        .event-row .atype-tag.delay    { background: #fee2e2; color: #991b1b; }
        .event-row .atype-tag.none     { background: #f1f5f9; color: #64748b; }

        /* 차트 */
        canvas#chart { width: 100%; height: 220px; display: block; background: #fafafa; border-radius: 6px; }
        .chart-legend { display: flex; gap: 14px; font-size: 11px; margin-top: 8px; justify-content: center; color: var(--muted); }
        .chart-legend span { display: inline-flex; align-items: center; gap: 5px; }
        .chart-legend .dot { width: 10px; height: 10px; border-radius: 50%; }
    </style>
</head>
<body>
    <h1>TurtleBot3 Security Dashboard</h1>
    <div class="hint">패널 클릭 → 상세 보기 토글 | LIVE MAP 클릭 → 전체화면</div>

    <div class="main-layout">
        <div class="left-panel">
            <div id="mode-box" class="mode-box clickable mode-normal" onclick="togglePanel('panel-ml')">
                <div id="mode-text">UNKNOWN</div>
                <div class="sub">click for ML detail</div>
            </div>

            <div id="attack-badge" class="attack-badge none">
                <span class="ab-tag" id="ab-tag">NORMAL</span>
                <div>
                    <div class="ab-title" id="ab-title">No network anomaly</div>
                    <div class="ab-desc" id="ab-desc">규칙 기반 위반 없음</div>
                </div>
            </div>

            <div id="panel-ml" class="expand-panel">
                <h3>ML Anomaly Score</h3>
                <div class="gauge-label">
                    <span id="ml-score-label">score: 0.000</span>
                    <span id="ml-state-label">NORMAL</span>
                </div>
                <div class="gauge"><div id="ml-gauge" class="gauge-bar" style="width:0%"></div></div>

                <h3 style="margin-top:14px;">Current Features (flagged = outside normal range)</h3>
                <div class="feat-grid">
                    <div class="feat-item" id="feat-linear">
                        <h4>linear.x</h4>
                        <div class="v" id="feat-linear-v">-</div>
                        <div class="r">normal: |x| ≤ 0.50</div>
                    </div>
                    <div class="feat-item" id="feat-angular">
                        <h4>angular.z</h4>
                        <div class="v" id="feat-angular-v">-</div>
                        <div class="r">normal: |x| ≤ 1.00</div>
                    </div>
                    <div class="feat-item" id="feat-rate">
                        <h4>publish rate</h4>
                        <div class="v" id="feat-rate-v">-</div>
                        <div class="r">normal: ≤ 50/s</div>
                    </div>
                    <div class="feat-item" id="feat-delay">
                        <h4>delay</h4>
                        <div class="v" id="feat-delay-v">-</div>
                        <div class="r">normal: ≤ 1.0 s</div>
                    </div>
                </div>
            </div>

            <div class="grid">
                <div class="card">
                    <h3>Wall Direction</h3>
                    <div class="value" id="wall">-</div>
                </div>
                <div class="card clickable" onclick="togglePanel('panel-chart')">
                    <h3>Message Rate (click for chart)</h3>
                    <div class="value" id="msg_rate">0 msg/s</div>
                </div>
            </div>

            <div id="panel-chart" class="expand-panel">
                <h3>Recent 30 seconds (normalized)</h3>
                <canvas id="chart" width="600" height="220"></canvas>
                <div class="chart-legend">
                    <span><span class="dot" style="background:#2563eb"></span> linear.x / 0.22</span>
                    <span><span class="dot" style="background:#f59e0b"></span> angular.z / 2.84</span>
                    <span><span class="dot" style="background:#10b981"></span> publish_rate / 50</span>
                </div>
            </div>

            <div class="lidar-box">
                <h3>LIDAR Distance (m)</h3>
                <div class="lidar-grid">
                    <div class="lidar-cell lidar-front" id="cell-front">Front<br><span id="front">0.00</span>m</div>
                    <div class="lidar-cell lidar-left"  id="cell-left">Left<br><span id="left">0.00</span>m</div>
                    <div class="lidar-cell lidar-robot">BOT</div>
                    <div class="lidar-cell lidar-right" id="cell-right">Right<br><span id="right">0.00</span>m</div>
                    <div class="lidar-cell lidar-back"  id="cell-back">Back<br><span id="back">0.00</span>m</div>
                </div>
            </div>

            <div class="detection-box clickable" onclick="togglePanel('panel-events')">
                <h3>Last Detection (click for log)</h3>
                <div class="detection-text" id="detection">No anomaly detected</div>
            </div>

            <div id="panel-events" class="expand-panel">
                <h3>Event Log (click tab to filter by scenario)</h3>
                <div class="log-tabs">
                    <button class="tab active" data-filter="all">All<span class="tab-count" id="cnt-all">0</span></button>
                    <button class="tab" data-filter="flooding">Flooding<span class="tab-count" id="cnt-flooding">0</span></button>
                    <button class="tab" data-filter="spoofing">Spoofing<span class="tab-count" id="cnt-spoofing">0</span></button>
                    <button class="tab" data-filter="delay">Delay<span class="tab-count" id="cnt-delay">0</span></button>
                </div>
                <div class="event-log" id="event-log">
                    <div class="event-row empty">No events yet</div>
                </div>
            </div>
        </div>

        <div id="map-box" class="map-box clickable" onclick="toggleMapFullscreen()">
            <div class="map-header">
                <h3>LIVE MAP (click to expand)</h3>
                <span id="robot-badge" class="robot-badge offline">● ROBOT OFFLINE</span>
            </div>
            <img id="map-img" alt="Live Map" style="display:none">
            <div id="map-offline" class="map-offline">
                지도 수신 대기 중<br>Cartographer 실행 확인
            </div>
            <div class="pose-info" id="pose-info" style="display:none">
                X: <span id="pose-x">-</span>m &nbsp;
                Y: <span id="pose-y">-</span>m &nbsp;
                Yaw: <span id="pose-yaw">-</span>°
            </div>
        </div>
    </div>

<script>
function togglePanel(id) {
    document.getElementById(id).classList.toggle('open');
}
function toggleMapFullscreen() {
    document.getElementById('map-box').classList.toggle('fullscreen');
}

function toggleFlag(id, on) {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.toggle('flagged', on);
}

function updateStatus() {
    fetch('/status')
        .then(r => r.json())
        .then(d => {
            const modeBox = document.getElementById('mode-box');
            document.getElementById('mode-text').textContent = d.mode;
            modeBox.classList.toggle('mode-normal', d.mode === 'NORMAL');
            modeBox.classList.toggle('mode-safe',   d.mode !== 'NORMAL');

            document.getElementById('wall').textContent = (d.wall && d.wall !== 'None') ? d.wall : '-';
            document.getElementById('msg_rate').textContent = d.msg_rate + ' msg/s';
            document.getElementById('front').textContent = d.front;
            document.getElementById('left').textContent  = d.left;
            document.getElementById('right').textContent = d.right;
            document.getElementById('back').textContent  = d.back;

            ['front', 'left', 'right', 'back'].forEach(dir => {
                const cell = document.getElementById('cell-' + dir);
                cell.style.outline = (d.wall === dir) ? '2px solid var(--red)' : '';
            });

            document.getElementById('detection').textContent =
                d.detection ? d.detection : 'No anomaly detected';

            // --- ML 게이지 ---
            const score = parseFloat(d.ml_score || 0);
            const anomaly = (d.ml_anomaly === '1' || d.ml_anomaly === 1);
            // score_samples: 양수일수록 정상, 음수일수록 이상 (보통 -0.5 ~ 0.5)
            // 게이지: (-) 방향으로 갈수록 길어지게 -1~0.5를 100~0%로 매핑
            const pct = Math.max(0, Math.min(100, (0.5 - score) * 100));
            const gauge = document.getElementById('ml-gauge');
            gauge.style.width = pct.toFixed(1) + '%';
            gauge.classList.toggle('anomaly', anomaly);
            document.getElementById('ml-score-label').textContent = 'score: ' + score.toFixed(3);
            document.getElementById('ml-state-label').textContent = anomaly ? 'ANOMALY' : 'NORMAL';
            document.getElementById('ml-state-label').style.color = anomaly ? 'var(--red)' : 'var(--green)';

            // --- Feature 값 + 룰 기반 contribution 하이라이트 ---
            const lx = parseFloat(d.linear_x || 0);
            const az = parseFloat(d.angular_z || 0);
            const pr = parseFloat(d.publish_rate || 0);
            const dl = parseFloat(d.delay || 0);
            document.getElementById('feat-linear-v').textContent = lx.toFixed(3);
            document.getElementById('feat-angular-v').textContent = az.toFixed(3);
            document.getElementById('feat-rate-v').textContent = pr.toFixed(1) + '/s';
            document.getElementById('feat-delay-v').textContent = dl.toFixed(3) + 's';
            toggleFlag('feat-linear',  Math.abs(lx) > 0.5);
            toggleFlag('feat-angular', Math.abs(az) > 1.0);
            toggleFlag('feat-rate',    pr > 50);
            toggleFlag('feat-delay',   dl > 1.0);

            // --- 공격 유형 배지 ---
            const at = (d.attack_type || 'none').toLowerCase();
            const badge = document.getElementById('attack-badge');
            badge.className = 'attack-badge ' + at;
            const ATTACK_INFO = {
                none:     ['NORMAL',   'No confirmed attack',         '확정된 공격 없음 (ML이 거부한 규칙 위반 포함)'],
                flooding: ['FLOODING', 'High publish rate',           'publish_rate > 50/s (DoS/플러딩 의심)'],
                spoofing: ['SPOOFING', 'Abnormal velocity command',   '|linear|>0.5 또는 |angular|>1.0 (속도 스푸핑)'],
                delay:    ['DELAY',    'Stale command lag',           'header.stamp 대비 지연 > 1.0s'],
            };
            const info = ATTACK_INFO[at] || ATTACK_INFO.none;
            document.getElementById('ab-tag').textContent = info[0];
            document.getElementById('ab-title').textContent = info[1];
            document.getElementById('ab-desc').textContent = info[2];
        });
}

function updatePose() {
    fetch('/robot_pose').then(r => r.json()).then(p => {
        const badge = document.getElementById('robot-badge');
        const poseInfo = document.getElementById('pose-info');
        if (p.valid) {
            poseInfo.style.display = 'flex';
            document.getElementById('pose-x').textContent = p.x.toFixed(2);
            document.getElementById('pose-y').textContent = p.y.toFixed(2);
            document.getElementById('pose-yaw').textContent = p.yaw_deg.toFixed(1);
            if (p.stale) { badge.className = 'robot-badge stale'; badge.textContent = '● ROBOT (stale)'; }
            else { badge.className = 'robot-badge active'; badge.textContent = '● ROBOT ACTIVE'; }
        } else {
            poseInfo.style.display = 'none';
            badge.className = 'robot-badge offline';
            badge.textContent = '● ROBOT OFFLINE';
        }
    }).catch(() => {});
}

function updateMap() {
    const img = document.getElementById('map-img');
    const newImg = new Image();
    newImg.onload = function() {
        img.src = newImg.src;
        document.getElementById('map-offline').style.display = 'none';
        img.style.display = 'block';
    };
    newImg.onerror = function() {
        document.getElementById('map-offline').style.display = 'block';
        img.style.display = 'none';
    };
    newImg.src = '/map_snapshot?t=' + Date.now();
}

let eventFilter = 'all';

document.querySelectorAll('.log-tabs .tab').forEach(btn => {
    btn.addEventListener('click', (ev) => {
        ev.stopPropagation();
        document.querySelectorAll('.log-tabs .tab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        eventFilter = btn.dataset.filter;
        updateEventLog();
    });
});

function updateEventLog() {
    if (!document.getElementById('panel-events').classList.contains('open')) return;
    fetch('/event_log').then(r => r.json()).then(events => {
        // 탭별 카운트 (전체 기준)
        const counts = { all: events.length, flooding: 0, spoofing: 0, delay: 0 };
        events.forEach(e => {
            const t = (e.attack_type || 'none').toLowerCase();
            if (t in counts) counts[t]++;
        });
        document.getElementById('cnt-all').textContent      = counts.all;
        document.getElementById('cnt-flooding').textContent = counts.flooding;
        document.getElementById('cnt-spoofing').textContent = counts.spoofing;
        document.getElementById('cnt-delay').textContent    = counts.delay;

        const filtered = eventFilter === 'all'
            ? events
            : events.filter(e => (e.attack_type || 'none').toLowerCase() === eventFilter);

        const box = document.getElementById('event-log');
        if (!filtered.length) {
            box.innerHTML = `<div class="event-row empty">No events for "${eventFilter}"</div>`;
            return;
        }
        box.innerHTML = filtered.slice().reverse().map(e => {
            const date = new Date(e.timestamp * 1000);
            const t = date.toTimeString().slice(0, 8);
            const atype = (e.attack_type || 'none').toLowerCase();
            const atypeTag = `<span class="atype-tag ${atype}">${atype.toUpperCase()}</span>`;
            return `<div class="event-row"><span class="tag ${e.tag}">${e.tag}</span><span class="ts">${t}</span>${atypeTag}${e.detail} (score=${e.score})</div>`;
        }).join('');
    });
}

// ---- 차트 (canvas) ----
const chartCanvas = document.getElementById('chart');
const chartCtx = chartCanvas.getContext('2d');

function updateChart() {
    if (!document.getElementById('panel-chart').classList.contains('open')) return;
    fetch('/history').then(r => r.json()).then(h => drawChart(h));
}

function drawChart(h) {
    const W = chartCanvas.width, H = chartCanvas.height;
    const ctx = chartCtx;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#fafafa';
    ctx.fillRect(0, 0, W, H);

    const pad = { l: 36, r: 12, t: 12, b: 24 };
    const innerW = W - pad.l - pad.r;
    const innerH = H - pad.t - pad.b;

    // 그리드
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = pad.t + (innerH / 4) * i;
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    }

    // y레이블
    ctx.fillStyle = '#64748b'; ctx.font = '10px sans-serif';
    ctx.fillText('1.0', 8, pad.t + 4);
    ctx.fillText('0.0', 8, pad.t + innerH / 2 + 4);
    ctx.fillText('-1.0', 4, pad.t + innerH + 2);

    if (!h.timestamps || !h.timestamps.length) {
        ctx.fillStyle = '#94a3b8'; ctx.font = '13px sans-serif';
        ctx.fillText('No data yet — waiting for /security_status', W / 2 - 140, H / 2);
        return;
    }

    const t0 = h.timestamps[0];
    const tLast = h.timestamps[h.timestamps.length - 1];
    const tSpan = Math.max(tLast - t0, 1);

    function xOf(t) { return pad.l + ((t - t0) / tSpan) * innerW; }
    function yOf(v) {
        const c = Math.max(-1, Math.min(1, v));
        return pad.t + innerH - ((c + 1) / 2) * innerH;
    }

    function plotLine(values, color) {
        ctx.strokeStyle = color; ctx.lineWidth = 2;
        ctx.beginPath();
        let first = true;
        for (let i = 0; i < h.timestamps.length; i++) {
            const x = xOf(h.timestamps[i]);
            const y = yOf(values[i]);
            if (first) { ctx.moveTo(x, y); first = false; }
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    plotLine(h.linear_x.map(v => v / 0.22),  '#2563eb');
    plotLine(h.angular_z.map(v => v / 2.84), '#f59e0b');
    plotLine(h.publish_rate.map(v => v / 50), '#10b981');

    // x축 시간 표시
    ctx.fillStyle = '#64748b'; ctx.font = '10px sans-serif';
    ctx.fillText('-' + tSpan.toFixed(0) + 's', pad.l, H - 6);
    ctx.fillText('now', W - pad.r - 22, H - 6);
}

setInterval(updateStatus,   500);
setInterval(updatePose,     500);
setInterval(updateMap,      500);
setInterval(updateEventLog, 1000);
setInterval(updateChart,    500);
updateStatus(); updatePose(); updateMap();
</script>
</body>
</html>
'''


# ──────────────────────────────────────────────────────────────────── #
# Flask 라우트
# ──────────────────────────────────────────────────────────────────── #
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/status')
def status():
    return jsonify(current_status)


@app.route('/robot_pose')
def robot_pose():
    cache = robot_pose_cache
    if not cache['valid']:
        return jsonify({'valid': False})
    age = time.time() - cache['timestamp']
    return jsonify({
        'valid': True,
        'x': round(cache['x'], 3),
        'y': round(cache['y'], 3),
        'yaw_deg': round(cache['yaw_deg'], 1),
        'stale': age > POSE_STALE_SEC
    })


@app.route('/history')
def history_route():
    with history_lock:
        items = list(feature_history)
    return jsonify({
        'timestamps':   [r[0] for r in items],
        'linear_x':     [r[1] for r in items],
        'angular_z':    [r[2] for r in items],
        'publish_rate': [r[3] for r in items],
        'delay':        [r[4] for r in items],
    })


@app.route('/event_log')
def event_log_route():
    with event_lock:
        items = list(event_log)
    return jsonify(items)


@app.route('/map_snapshot')
def map_snapshot():
    with map_lock:
        img = latest_map_image
    if img is None:
        return Response(status=503)
    ret, buffer = cv2.imencode('.png', img)
    if not ret:
        return Response(status=503)
    return Response(buffer.tobytes(), mimetype='image/png')


# ──────────────────────────────────────────────────────────────────── #
# ROS 노드
# ──────────────────────────────────────────────────────────────────── #
class DashboardNode(Node):
    def __init__(self):
        super().__init__('dashboard_node')
        self.sub = self.create_subscription(
            String, '/security_status', self.status_callback, 10)
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_timer(0.2, self.poll_robot_pose)
        self.get_logger().info('Dashboard Node started')

    def status_callback(self, msg):
        global _last_detection_text, _last_ml_anomaly

        parsed = {}
        for pair in msg.data.split('|'):
            if ':' in pair:
                key, value = pair.split(':', 1)
                parsed[key] = value
                current_status[key] = value

        # 히스토리 누적
        try:
            ts = time.time()
            lx = float(parsed.get('linear_x', 0))
            az = float(parsed.get('angular_z', 0))
            pr = float(parsed.get('publish_rate', 0))
            dl = float(parsed.get('delay', 0))
            with history_lock:
                feature_history.append((ts, lx, az, pr, dl))
                while feature_history and ts - feature_history[0][0] > HISTORY_WINDOW:
                    feature_history.popleft()
        except (ValueError, TypeError):
            pass

        # 이벤트 로깅 (rising edge)
        det = parsed.get('detection', '')
        ml_an = parsed.get('ml_anomaly', '0') == '1'
        ml_score = parsed.get('ml_score', '0')
        attack_type = parsed.get('attack_type', 'none')

        if ml_an and not _last_ml_anomaly:
            with event_lock:
                event_log.append({
                    'timestamp': time.time(),
                    'tag': 'ML',
                    'detail': 'ML flagged anomaly',
                    'score': ml_score,
                    'attack_type': attack_type,
                })
        if det and det != _last_detection_text:
            with event_lock:
                event_log.append({
                    'timestamp': time.time(),
                    'tag': 'RULE',
                    'detail': det,
                    'score': ml_score,
                    'attack_type': attack_type,
                })

        _last_detection_text = det
        _last_ml_anomaly = ml_an

    def poll_robot_pose(self):
        global robot_pose_cache
        for child_frame in ['base_footprint', 'base_link']:
            try:
                tf = self.tf_buffer.lookup_transform(
                    'map', child_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.05)
                )
                rx = tf.transform.translation.x
                ry = tf.transform.translation.y
                q = tf.transform.rotation
                yaw = math.atan2(
                    2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                )
                robot_pose_cache.update({
                    'x': rx, 'y': ry,
                    'yaw_deg': math.degrees(yaw),
                    'timestamp': time.time(),
                    'valid': True
                })
                return
            except Exception:
                continue

    def get_robot_pose_from_cache(self):
        cache = robot_pose_cache
        if not cache['valid']:
            return None, None, None, False
        age = time.time() - cache['timestamp']
        stale = age > POSE_STALE_SEC
        return cache['x'], cache['y'], cache['yaw_deg'], stale

    def draw_robot_marker(self, img, px, py_f, yaw_deg, width, height, stale=False):
        if not (0 <= px < width and 0 <= py_f < height):
            return
        arrow_color = (0, 180, 60) if not stale else (0, 140, 200)
        dot_color   = (255, 130, 0) if not stale else (80, 80, 200)
        ring_color  = (0, 220, 255) if not stale else (150, 150, 150)
        arrow_len = 24
        ax = int(px + arrow_len * math.cos(math.radians(yaw_deg)))
        ay = int(py_f - arrow_len * math.sin(math.radians(yaw_deg)))
        cv2.arrowedLine(img, (px, py_f), (ax, ay),
                        arrow_color, 3, tipLength=0.35)
        cv2.circle(img, (px, py_f), 9, (255, 255, 255), -1)
        cv2.circle(img, (px, py_f), 7, dot_color, -1)
        cv2.circle(img, (px, py_f), 13, ring_color, 2)
        if stale:
            cv2.circle(img, (px, py_f), 18, (100, 100, 100), 1)

    def map_callback(self, msg):
        global latest_map_image, map_meta

        width = msg.info.width
        height = msg.info.height
        res = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        map_meta.update({'resolution': res, 'origin_x': origin_x,
                         'origin_y': origin_y, 'width': width, 'height': height})

        data = np.array(msg.data, dtype=np.int8).reshape((height, width))

        img = np.zeros((height, width, 3), dtype=np.uint8)
        # 밝은 테마에 맞춰 미탐색은 옅은 회색으로
        img[data == -1] = [225, 230, 235]
        img[data == 0]  = [255, 255, 255]
        img[data > 50]  = [0, 0, 0]

        img = cv2.flip(img, 0)

        rx, ry, yaw_deg, stale = self.get_robot_pose_from_cache()
        if rx is not None:
            px = int((rx - origin_x) / res)
            py = int((ry - origin_y) / res)
            py_f = height - 1 - py
            self.draw_robot_marker(img, px, py_f, yaw_deg, width, height, stale)

        scale = 2
        img = cv2.resize(img, (width * scale, height * scale),
                         interpolation=cv2.INTER_NEAREST)

        with map_lock:
            latest_map_image = img


def ros_thread():
    rclpy.init()
    node = DashboardNode()
    rclpy.spin(node)


def main():
    t = threading.Thread(target=ros_thread, daemon=True)
    t.start()
    print('Dashboard running at http://localhost:5000')
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


if __name__ == '__main__':
    main()
