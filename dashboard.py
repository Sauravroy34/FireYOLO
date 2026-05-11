"""
FireYOLO Panel Dashboard
========================
Real-time fire detection dashboard combining:
  - ESP32-CAM MJPEG video stream with YOLO inference
  - ESP32 UDP sensor telemetry (temperature, humidity)

Run with:
    panel serve dashboard.py --show --autoreload
"""

import threading
import socket
import time
import io
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import panel as pn
from bokeh.models import ColumnDataSource, DatetimeTickFormatter, NumeralTickFormatter
from bokeh.plotting import figure
from bokeh.palettes import Sunset4
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
MODEL_PATH = "/home/saurav/Desktop/FireYOLO/best.pt"
DEFAULT_ESP32_URL = "http://10.56.153.229"
UDP_PORT = 4210
SENSOR_FIELDS = ["temperature", "humidity"]
CHART_ROLLOVER = 200          # max data-points kept per sensor chart
VIDEO_CALLBACK_MS = 120       # ~8 FPS target for annotated video
SENSOR_CALLBACK_MS = 500      # poll new sensor readings every 500ms
FIRE_ALERT_CALLBACK_MS = 200  # push fire-state to ESP32 this often (ms)
FRAME_WIDTH = 640
FRAME_HEIGHT = 640

# ──────────────────────────────────────────────────────────────────────
# Panel extensions
# ──────────────────────────────────────────────────────────────────────

pn.extension(sizing_mode="stretch_width", notifications=True)


# ======================================================================
#  VideoStream — background thread: capture → YOLO → JPEG bytes
# ======================================================================
class VideoStream:
    """Continuously grabs frames from an ESP32 MJPEG stream,
    runs YOLO inference, and stores the latest annotated JPEG."""

    def __init__(self, model_path: str, esp32_url: str):
        self.model = YOLO(model_path)
        self.esp32_url = esp32_url
        self.cap = None

        self._lock = threading.Lock()
        self._latest_jpg: bytes | None = None
        self._fire_detected: bool = False
        self._detection_count: int = 0
        self._connected: bool = False
        self._running: bool = False
        self._thread: threading.Thread | None = None

    # ---- public API (called from Panel callbacks) --------------------
    def get_frame(self) -> bytes | None:
        with self._lock:
            return self._latest_jpg

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def fire_detected(self) -> bool:
        with self._lock:
            return self._fire_detected

    @property
    def detection_count(self) -> int:
        with self._lock:
            return self._detection_count

    # ---- lifecycle ---------------------------------------------------
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self.cap and self.cap.isOpened():
            self.cap.release()

    # ---- internal loop -----------------------------------------------
    def _run(self):
        while self._running:
            try:
                self.cap = cv2.VideoCapture(self.esp32_url)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                with self._lock:
                    self._connected = self.cap.isOpened()

                while self._running and self.cap.isOpened():
                    ok, frame = self.cap.read()
                    if not ok:
                        break

                    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

                    # Run YOLO (non-streaming for single frame)
                    results = self.model(frame, verbose=False)

                    fire_now = False
                    for r in results:
                        annotated = r.plot()
                        if len(r.boxes) > 0:
                            fire_now = True

                    # Encode annotated frame to JPEG bytes
                    ok_enc, buf = cv2.imencode(
                        ".jpg", annotated,
                        [cv2.IMWRITE_JPEG_QUALITY, 80],
                    )
                    if ok_enc:
                        with self._lock:
                            self._latest_jpg = buf.tobytes()
                            self._fire_detected = fire_now


            except Exception as exc:
                print(f"[VideoStream] Error: {exc}")

            # brief pause before reconnect attempt
            with self._lock:
                self._connected = False
            if self._running:
                time.sleep(2)


# ======================================================================
#  SensorReceiver — background thread: UDP → deque of dicts
# ======================================================================
class SensorReceiver:
    """Listens for UDP packets from the ESP32 sensor board, parses
    the human-readable format ``Temp: X.XXC, Hum: Y.YY%``,
    and buffers readings in a thread-safe deque."""

    def __init__(self, port: int = UDP_PORT, fields: list[str] | None = None):
        self.port = port
        self.fields = fields or SENSOR_FIELDS
        self._buffer: deque[dict] = deque(maxlen=500)
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._receiving = False
        self._last_addr: tuple | None = None   # (ip, port) of the Arduino
        self._sock: socket.socket | None = None  # shared so send_response can use it

    # ---- public API --------------------------------------------------
    def drain(self) -> list[dict]:
        """Atomically retrieve and clear all buffered readings."""
        with self._lock:
            items = list(self._buffer)
            self._buffer.clear()
        return items

    def send_response(self, msg: str) -> None:
        """Send a UDP response back to the last known Arduino address."""
        with self._lock:
            addr = self._last_addr
            sock = self._sock
        if addr and sock:
            try:
                sock.sendto(msg.encode("utf-8"), addr)
            except Exception as exc:
                print(f"[SensorReceiver] send_response error: {exc}")

    @property
    def receiving(self) -> bool:
        with self._lock:
            return self._receiving

    # ---- lifecycle ---------------------------------------------------
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    # ---- internal loop -----------------------------------------------
    def _run(self):
        import re
        # Matches: "Temp: 25.30C, Hum: 60.50%"  (case-insensitive, flexible spacing)
        _PATTERN = re.compile(
            r"Temp:\s*([\d.]+)C,?\s*Hum:\s*([\d.]+)%",
            re.IGNORECASE,
        )

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self.port))
        sock.settimeout(1.0)  # non-blocking-ish so we can check _running
        with self._lock:
            self._sock = sock
        print(f"[SensorReceiver] Listening on UDP port {self.port}")

        while self._running:
            try:
                data, addr = sock.recvfrom(1024)
                msg = data.decode("utf-8").strip()

                m = _PATTERN.search(msg)
                if m:
                    reading = {
                        "time":        datetime.now(),
                        "temperature": float(m.group(1)),
                        "humidity":    float(m.group(2)),
                    }
                    with self._lock:
                        self._buffer.append(reading)
                        self._receiving = True
                        self._last_addr = addr  # remember Arduino's address
                else:
                    print(f"[SensorReceiver] Unexpected format: {msg!r}")

            except socket.timeout:
                continue
            except Exception as exc:
                print(f"[SensorReceiver] Error: {exc}")

        sock.close()


# ======================================================================
#  Bokeh chart helpers
# ======================================================================
SENSOR_COLORS = {
    "temperature": "#FF6B6B",   # warm red
    "humidity":    "#4ECDC4",   # teal
}

SENSOR_UNITS = {
    "temperature": "°C",
    "humidity":    "%",
}


def make_sensor_chart(field: str, source: ColumnDataSource) -> figure:
    """Create a styled Bokeh time-series figure for one sensor."""
    color = SENSOR_COLORS.get(field, "#ffffff")
    unit = SENSOR_UNITS.get(field, "")

    p = figure(
        title=f"{field.capitalize()} ({unit})",
        x_axis_type="datetime",
        height=160,
        sizing_mode="stretch_width",
        toolbar_location=None,
        background_fill_color="#1a1a2e",
        border_fill_color="#16213e",
        outline_line_color="#0f3460",
    )

    p.line(
        x="time", y=field, source=source,
        line_width=2, color=color, alpha=0.9,
    )
    p.circle(
        x="time", y=field, source=source,
        size=3, color=color, alpha=0.5,
    )

    # --- axis styling ---
    p.xaxis.formatter = DatetimeTickFormatter(seconds="%H:%M:%S", minutes="%H:%M")
    p.xaxis.axis_label_text_color = "#a0a0c0"
    p.xaxis.major_label_text_color = "#a0a0c0"
    p.xaxis.major_tick_line_color = "#a0a0c0"
    p.xaxis.minor_tick_line_color = "#a0a0c0"

    p.yaxis.axis_label = unit
    p.yaxis.axis_label_text_color = "#a0a0c0"
    p.yaxis.major_label_text_color = "#a0a0c0"
    p.yaxis.major_tick_line_color = "#a0a0c0"
    p.yaxis.minor_tick_line_color = "#a0a0c0"

    p.title.text_color = color
    p.title.text_font_size = "11pt"
    p.title.text_font_style = "bold"

    p.grid.grid_line_color = "#2a2a4a"
    p.grid.grid_line_alpha = 0.4

    return p


# ======================================================================
#  Dashboard construction
# ======================================================================
def build_dashboard():
    """Create and return the complete Panel dashboard."""

    # ── Start backend services ─────────────────────────────────────
    video_stream = VideoStream(MODEL_PATH, DEFAULT_ESP32_URL)
    sensor_receiver = SensorReceiver(UDP_PORT, SENSOR_FIELDS)

    video_stream.start()
    sensor_receiver.start()

    # ── Video pane ─────────────────────────────────────────────────
    # Create a 1×1 black placeholder JPEG to show on startup
    placeholder = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    cv2.putText(
        placeholder,
        "Connecting to ESP32-CAM...",
        (80, FRAME_HEIGHT // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 200), 2,
    )
    _, placeholder_buf = cv2.imencode(".jpg", placeholder)
    placeholder_bytes = placeholder_buf.tobytes()

    image_pane = pn.pane.JPG(
        placeholder_bytes,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        sizing_mode="fixed",
    )

    # ── Sensor chart data source ───────────────────────────────────
    sensor_source = ColumnDataSource(
        data={"time": [], **{f: [] for f in SENSOR_FIELDS}}
    )

    sensor_charts = pn.Column(
        *(
            pn.pane.Bokeh(make_sensor_chart(f, sensor_source), sizing_mode="stretch_width")
            for f in SENSOR_FIELDS
        ),
        sizing_mode="stretch_width",
    )

    # ── Risk alert banner (below sensor charts) ───────────────────
    alert_banner = pn.pane.HTML(
        """
        <div style="background: rgba(255,255,255,0.04); border-radius: 10px;
                    padding: 12px 16px; margin-top: 6px;
                    border: 1px solid #2a2a4a; color: #888; font-size: 12px;">
            ℹ️    &nbsp; Awaiting sensor data…
        </div>
        """,
        sizing_mode="stretch_width",
    )

    # ── Fire detection banner (below camera feed) ───────────────────
    fire_banner = pn.pane.HTML(
        """
        <div style="background: rgba(255,255,255,0.04); border-radius: 10px;
                    padding: 12px 16px; margin-top: 6px;
                    border: 1px solid #2a2a4a; color: #888; font-size: 12px;">
            &nbsp; No fire detected
        </div>
        """,
        sizing_mode="stretch_width",
    )

    # ── Sidebar widgets & indicators ───────────────────────────────
    camera_status = pn.indicators.BooleanStatus(
        value=False, color="success",
        width=18, height=18,
    )
    udp_status = pn.indicators.BooleanStatus(
        value=False, color="success",
        width=18, height=18,
    )
    fire_status = pn.indicators.BooleanStatus(
        value=False, color="danger",
        width=18, height=18,
    )

    # Latest sensor value indicators
    temp_indicator = pn.indicators.Number(
        name="Temperature", value=0, format="{value:.1f} °C",
        colors=[(30, "green"), (45, "gold"), (100, "red")],
        font_size="22pt", title_size="10pt",
    )
    hum_indicator = pn.indicators.Number(
        name="Humidity", value=0, format="{value:.1f} %",
        colors=[(30, "red"), (60, "gold"), (100, "green")],
        font_size="22pt", title_size="10pt",
    )

    _prev_fire: list[bool] = [False]  # mutable container for closure state

    # ── Periodic callback: VIDEO ───────────────────────────────────
    def update_video():
        jpg = video_stream.get_frame()
        if jpg is not None:
            image_pane.object = jpg

        # Status indicators
        camera_status.value = video_stream.connected
        fire_now = video_stream.fire_detected
        fire_status.value = fire_now

        # Update fire banner on every tick (state-based, not transition-based)
        if fire_now:
            fire_banner.object = """
            <div style="
                background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
                border-radius: 10px; padding: 14px 18px; margin-top: 6px;
                border-left: 5px solid #ef4444;
                display: flex; align-items: center; gap: 12px;
                animation: firepulse 1.0s ease-in-out infinite;
            ">
                <div>
                    <div style="color: #fca5a5; font-weight: 700; font-size: 13px;
                                letter-spacing: 0.5px;">
                        FIRE DETECTED
                    </div>
                    <div style="color: #fecaca; font-size: 11px; margin-top: 3px;">
                        YOLO model has identified fire in the frame
                    </div>
                </div>
            </div>
            <style>
              @keyframes firepulse {
                0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.6); }
                50%      { box-shadow: 0 0 0 10px rgba(239,68,68,0); }
              }
            </style>
            """
        else:
            fire_banner.object = """
            <div style="
                background: rgba(20,83,45,0.5); border-radius: 10px;
                padding: 14px 18px; margin-top: 6px;
                border-left: 5px solid #22c55e;
                display: flex; align-items: center; gap: 12px;
            ">
                <div style="color: #86efac; font-weight: 600; font-size: 13px;">
                    No Fire Detected
                </div>
            </div>
            """

        _prev_fire[0] = fire_now

    # ── Periodic callback: SENSOR ──────────────────────────────────
    def update_sensors():
        readings = sensor_receiver.drain()
        udp_status.value = sensor_receiver.receiving

        if not readings:
            return

        new_data = {
            "time": [r["time"] for r in readings],
            "temperature": [r.get("temperature", 0.0) for r in readings],
            "humidity":    [r.get("humidity",    0.0) for r in readings],
        }

        sensor_source.stream(new_data, rollover=CHART_ROLLOVER)

        # Update numeric indicators with latest value
        latest = readings[-1]
        temp_val = latest.get("temperature", 0)
        hum_val  = latest.get("humidity",    100)
        temp_indicator.value = temp_val
        hum_indicator.value  = hum_val

        # ── Risk alert banner ─────────────────────────────────────
        if temp_val > 35 or hum_val < 70:
            alert_banner.object = """
            <div style="
                background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
                border-radius: 10px; padding: 14px 18px; margin-top: 6px;
                border-left: 5px solid #ef4444;
                display: flex; align-items: center; gap: 12px;
                animation: pulse 1.4s ease-in-out infinite;
            ">
                <span style="font-size: 28px;"></span>
                <div>
                    <div style="color: #fca5a5; font-weight: 700; font-size: 13px;
                                letter-spacing: 0.5px;">
                        RISK OF EARLY FOREST FIRE
                    </div>
                    <div style="color: #fecaca; font-size: 11px; margin-top: 3px;">
                        High temp or low humidity detected
                    </div>
                </div>
            </div>
            <style>
              @keyframes pulse {
                0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.5); }
                50%      { box-shadow: 0 0 0 8px rgba(239,68,68,0); }
              }
            </style>
            """
        else:
            alert_banner.object = """
            <div style="
                background: rgba(20,83,45,0.6); border-radius: 10px;
                padding: 14px 18px; margin-top: 6px;
                border-left: 5px solid #22c55e;
                display: flex; align-items: center; gap: 12px;
            ">
                <span style="font-size: 24px;"></span>
                <div style="color: #86efac; font-weight: 600; font-size: 13px;">
                    Conditions Normal
                </div>
            </div>
            """

    # ── Periodic callback: FIRE ALERT ──────────────────────────────
    # Proactively push fire state to ESP32 every 200 ms so the buzzer
    # reacts immediately — independent of the 2-second DHT read cycle.
    def push_fire_alert():
        response = "2" if video_stream.fire_detected else "0"
        sensor_receiver.send_response(response)

    # ── Register callbacks ─────────────────────────────────────────
    video_cb = pn.state.add_periodic_callback(
        update_video, period=VIDEO_CALLBACK_MS
    )
    sensor_cb = pn.state.add_periodic_callback(
        update_sensors, period=SENSOR_CALLBACK_MS
    )
    fire_alert_cb = pn.state.add_periodic_callback(
        push_fire_alert, period=FIRE_ALERT_CALLBACK_MS
    )

    # ── Cleanup on session destroy ─────────────────────────────────
    def cleanup(session_context):
        video_cb.stop()
        sensor_cb.stop()
        fire_alert_cb.stop()
        video_stream.stop()
        sensor_receiver.stop()

    pn.state.on_session_destroyed(cleanup)

    # ── Assemble sidebar ───────────────────────────────────────────
    status_section = pn.Column(
        pn.pane.Markdown(
            "## Connection Status",
            styles={"color": "#e0e0e0"},
        ),
        pn.Row(camera_status, pn.pane.Markdown("**Camera Stream**", styles={"color": "#ccc", "margin": "0"})),
        pn.Row(udp_status, pn.pane.Markdown("**UDP Sensors**", styles={"color": "#ccc", "margin": "0"})),
        pn.Row(fire_status, pn.pane.Markdown("**Fire Detected**", styles={"color": "#ccc", "margin": "0"})),
        pn.layout.Divider(),
        styles={"padding": "5px"},
    )

    sensor_values_section = pn.Column(
        pn.pane.Markdown(
            "## Latest Readings",
            styles={"color": "#e0e0e0"},
        ),
        pn.Row(temp_indicator, hum_indicator),
        pn.layout.Divider(),
    )

    sidebar_content = pn.Column(
        status_section,
        sensor_values_section,
        sizing_mode="stretch_width",
    )

    # ── Assemble main area ─────────────────────────────────────────
    video_card = pn.Column(
        pn.pane.Markdown(
            "### Live Camera Feed — YOLO Fire Detection",
            styles={"color": "#e0e0e0", "margin-bottom": "5px"},
        ),
        image_pane,
        fire_banner,
        styles={
            "background": "#1a1a2e",
            "border-radius": "12px",
            "padding": "15px",
            "border": "1px solid #0f3460",
        },
    )

    sensor_card = pn.Column(
        pn.pane.Markdown(
            "### Real-Time Sensor Data",
            styles={"color": "#e0e0e0", "margin-bottom": "5px"},
        ),
        sensor_charts,
        alert_banner,
        styles={
            "background": "#1a1a2e",
            "border-radius": "12px",
            "padding": "15px",
            "border": "1px solid #0f3460",
        },
        sizing_mode="stretch_width",
    )

    main_content = pn.Row(
        video_card,
        sensor_card,
        sizing_mode="stretch_width",
    )

    # ── Template ───────────────────────────────────────────────────
    template = pn.template.FastListTemplate(
        title="FireYOLO — Early Forest Fire Detection System",
        sidebar=[sidebar_content],
        main=[main_content],
        theme="dark",
        accent="#e74c3c",
        header_background="#16213e",
        main_layout=None,           # no extra wrapping
        sidebar_width=320,
    )

    return template


# ======================================================================
#  Entry point — `panel serve dashboard.py`
# ======================================================================
build_dashboard().servable()
