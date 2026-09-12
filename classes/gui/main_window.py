"""New QMainWindow — pure Python, no .ui files.

Central widget: log panel (grows to fill the vertical space).
Right dock: tabbed control panels (Mode A, Mode B, Solver, Calibration, Supervisor).
Bottom dock: mode summary (which mode is active, what B_des is commanded).

Video / tracker integration is not wired here yet — the new stack ships
with manual control first; camera and closed-loop follow when the operator
is happy with Modes A/B on the bench.
"""

from __future__ import annotations

from typing import Optional

from PyQt5 import QtCore, QtWidgets

from classes.gui.panels import (
    LogPanel,
    ModeAPanel,
    ModeBPanel,
    SolverPanel,
    SupervisorPanel,
)
from classes.gui.calibration_wizard import CalibrationWizard
from classes.gui.coil_viz import CoilVisualization
from classes.gui.frame_calibration import FrameCalibrationPanel
from classes.gui.joystick_bridge import JoystickBridge
from classes.gui.joystick_panel import JoystickPanel
from classes.gui.live_calibration_panel import LiveCalibrationPanel
from classes.gui.path_follow import PathFollowController
from classes.gui.recording import RecordingController
from classes.gui.tracker_panel import TrackerPanel
from classes.gui.video import CameraPanel, VideoSource, VideoWidget
from classes.gui.widgets import scroll_wrap


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self,
                 config,
                 motion,
                 supervisor,
                 hal,
                 rebuild_solver_cb,
                 parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.config = config
        self.motion = motion
        self.supervisor = supervisor
        self.hal = hal
        # Wrap the caller's rebuild callback so the coil viz's Bmap follows
        # every calibration change without a separate wire from each source.
        self._orig_rebuild_cb = rebuild_solver_cb
        self.rebuild_solver_cb = self._rebuild_and_refresh_viz

        # Coil-power dial (Joystick tab) → HAL, live. The HAL otherwise
        # reads limits.output_scale only once at construction.
        try:
            config.on_change(
                "limits.output_scale",
                lambda *_a: self.hal.set_output_scale(
                    float(config.get("limits.output_scale", 1.0))))
        except Exception:
            pass

        self.setWindowTitle("BigTweezerSystem — Control")
        self.resize(1400, 900)

        # Central widget: vertical splitter, video on top and log on bottom.
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.video = VideoWidget(config=config)
        self.log = LogPanel()
        splitter.addWidget(self.video)
        splitter.addWidget(self.log)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        # Video source thread. Frames land on the widget via signal.
        self.video_source = VideoSource(printer=self._log)
        self.video_source.frameReady.connect(self.video.set_frame)

        # Annotated-view recorder (⏺ button on the video widget).
        self.recorder = RecordingController(
            self.video, config, log_fn=self._log)

        # Right dock: tabbed control panels.
        self.tabs = QtWidgets.QTabWidget()
        self.camera_panel = CameraPanel(self.video_source)
        self.mode_a = ModeAPanel(motion, config)
        self.mode_b = ModeBPanel(motion, config)
        self.solver_panel = SolverPanel(config)
        self.solver_panel.solverChanged.connect(self.rebuild_solver_cb)
        self.calibration = CalibrationWizard(
            hal, config, supervisor=supervisor, log_fn=self._log)
        self.calibration.finished.connect(self.rebuild_solver_cb)
        self.live_calibration = LiveCalibrationPanel(
            hal, config, supervisor=supervisor, log_fn=self._log)
        self.live_calibration.gainsChanged.connect(self.rebuild_solver_cb)
        # Group the two calibration views under one tab.
        self.calibration_tabs = QtWidgets.QTabWidget()
        self.calibration_tabs.addTab(
            scroll_wrap(self.calibration), "Wizard (Hall probe)")
        self.calibration_tabs.addTab(
            scroll_wrap(self.live_calibration), "Live tune (per-coil)")
        self.supervisor_panel = SupervisorPanel(supervisor, config)

        # Tracker + path-follow controller. Runs entirely off video frames;
        # emits state updates to the panel and overlay updates to the video
        # widget. Uses Source.ALGORITHM to talk to MotionController so joystick
        # writes still win.
        self.path_follow = PathFollowController(
            motion, config, log_fn=self._log)
        self.tracker_panel = TrackerPanel(self.path_follow)
        # Frames feed the trackers.
        self.video_source.frameReady.connect(self.path_follow.on_frame)
        # Mouse clicks on the video widget flow into the controller.
        self.video.leftClicked.connect(self._on_video_left_click)
        self.video.rightClicked.connect(self._on_video_right_click)
        self.video.middleClicked.connect(self._on_video_middle_click)
        # Overlay refresh on any state change.
        self.path_follow.overlayChanged.connect(self._refresh_overlay)

        self.joystick_bridge = JoystickBridge(
            motion, supervisor, config, log_fn=self._log)
        # Live coil-firing + pull-direction visualization. Bmap is snapshotted
        # here; if calibration changes later, the rebuild_solver_cb path pushes
        # the new matrix in via set_bmap.
        self.coil_viz = CoilVisualization(
            motion.solver.Bmap, motion=motion, config=config)
        self.joystick_panel = JoystickPanel(
            self.joystick_bridge, coil_viz=self.coil_viz)

        # Frame calibration reuses the same tracker (path_follow.robot) that
        # the Tracker tab set up, and pulses coils via the same MotionController.
        self.frame_cal = FrameCalibrationPanel(
            self.path_follow, motion, config,
            joystick_bridge=self.joystick_bridge, log_fn=self._log)

        self.tabs.addTab(scroll_wrap(self.camera_panel), "Camera")
        # Tracker tab lives right after Camera because the operator uses it
        # against the same video feed.
        self.tabs.addTab(scroll_wrap(self.tracker_panel), "Tracker")
        # Frame calibration sits next to Tracker — same video-click flow.
        self.tabs.addTab(scroll_wrap(self.frame_cal), "Frame Cal")
        self.tabs.addTab(scroll_wrap(self.mode_a), "Mode A — Rotate")
        self.tabs.addTab(scroll_wrap(self.mode_b), "Mode B — Static")
        # Joystick sits between Mode B and Solver — closest to the
        # manual-control tabs it augments.
        self.tabs.addTab(scroll_wrap(self.joystick_panel), "Joystick")
        self.tabs.addTab(scroll_wrap(self.solver_panel), "Solver")
        self.tabs.addTab(self.calibration_tabs, "Calibration")
        self.tabs.addTab(scroll_wrap(self.supervisor_panel), "Supervisor")

        right_dock = QtWidgets.QDockWidget("Control", self)
        right_dock.setWidget(self.tabs)
        right_dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, right_dock)

        # Bottom dock: quick command summary.
        self.summary = QtWidgets.QLabel("Mode OFF — no field commanded")
        self.summary.setStyleSheet("font-family: monospace; padding: 6px;")
        bottom_dock = QtWidgets.QDockWidget("Current command", self)
        bottom_dock.setWidget(self.summary)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, bottom_dock)

        # Menu bar
        m_file = self.menuBar().addMenu("&File")
        act_save = QtWidgets.QAction("Save config", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self._save_config)
        m_file.addAction(act_save)
        act_quit = QtWidgets.QAction("Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        m_file.addAction(act_quit)

        m_safety = self.menuBar().addMenu("&Safety")
        act_estop = QtWidgets.QAction("EMERGENCY STOP", self)
        act_estop.setShortcut("Ctrl+E")
        act_estop.triggered.connect(supervisor.estop)
        m_safety.addAction(act_estop)

        # Wire the motion controller to the supervisor + UI.
        motion.on_solved(self._on_solved)

        # Refresh the summary line at 10 Hz.
        self.sum_timer = QtCore.QTimer(self)
        self.sum_timer.setInterval(100)
        self.sum_timer.timeout.connect(self._refresh_summary)
        self.sum_timer.start()

        # Auto-connect the camera in the background so a live feed is up
        # when the operator first sees the window. If none of the backends
        # succeed the video widget just stays on its "no camera" placeholder;
        # operator can still manually retry from the Camera tab.
        QtCore.QTimer.singleShot(200, self._auto_connect_camera)
        # Same idea for the joystick — try once at startup so a pad plugged
        # in before launch just works. Skipped if the config disables it.
        QtCore.QTimer.singleShot(200, self._auto_connect_joystick)

    def _auto_connect_camera(self) -> None:
        if self.video_source.cap is not None:
            return
        try:
            self.video_source.open(source="auto")
        except Exception as e:
            self._log(f"Camera auto-connect failed: {e}")

    def _auto_connect_joystick(self) -> None:
        if not self.config.get("joystick.enabled", True):
            self._log("Joystick disabled in config; skipping auto-connect")
            return
        try:
            self.joystick_bridge.start()
        except Exception as e:
            self._log(f"Joystick auto-connect failed: {e}")

    # ---- tracker + video mouse routing --------------------------------

    def _on_video_left_click(self, x: int, y: int) -> None:
        # Snapshot the current frame from the widget for the mask init.
        self.path_follow.select_at(x, y, self.video._last_frame)

    def _on_video_right_click(self, x: int, y: int) -> None:
        self.path_follow.add_waypoint(x, y)

    def _on_video_middle_click(self, _x: int, _y: int) -> None:
        self.path_follow.clear()

    def _refresh_overlay(self) -> None:
        try:
            self.video.set_overlay(self.path_follow.overlay_snapshot())
        except Exception as e:
            self._log(f"Overlay refresh error: {e}")

    # ---- callbacks --------------------------------------------------

    def _rebuild_and_refresh_viz(self) -> None:
        """Wrapper around the caller-supplied rebuild callback so the coil
        visualization's Bmap follows every calibration change."""
        try:
            self._orig_rebuild_cb()
        except Exception as e:
            self._log(f"Solver rebuild failed: {e}")
        try:
            self.coil_viz.set_bmap(self.motion.solver.Bmap)
        except Exception:
            pass

    def _on_solved(self, result, acoustic_freq: float) -> None:
        # Route through supervisor first (returns actual sent currents).
        sent = self.supervisor.check_and_forward(result, acoustic_freq)
        self.supervisor_panel.set_currents_display(sent)
        # Push into the coil visualization. The widget throttles its own
        # repaints to ~30 Hz internally, so the 200 Hz inner-loop cadence
        # here is fine to just fire on every tick.
        self.coil_viz.update_state(sent)

    def _refresh_summary(self) -> None:
        m = self.motion
        mode_name = {
            "A": "Rolling pull",
            "B": "Static pull",
            "OFF": "OFF",
        }.get(m.mode.value, "?")
        if m.mode.value == "OFF":
            self.summary.setText("Mode OFF — coils commanded to zero")
        elif m.mode.value == "A":
            self.summary.setText(
                f"Mode A ({mode_name}): |B|={m.magnitude:.2f}, "
                f"f={m.freq_hz:.2f} Hz  (target {m.freq_target_hz:.2f})  "
                f"direction={m.direction.round(2).tolist()}  "
                f"roll_axis={m.roll_axis.round(2).tolist()}"
            )
        else:
            self.summary.setText(
                f"Mode B ({mode_name}): |B|={m.magnitude:.2f}  "
                f"direction={m.direction.round(2).tolist()}"
            )

    def _save_config(self) -> None:
        try:
            self.config.save()
            self._log("Config saved to config.yaml")
        except Exception as e:
            self._log(f"Config save failed: {e}")

    def _log(self, msg: str) -> None:
        self.log.append_line(msg)

    # ---- lifecycle --------------------------------------------------

    def closeEvent(self, event) -> None:
        try:
            self.joystick_bridge.stop()
        except Exception:
            pass
        try:
            # Before video_source.close() — finalizes a running recording.
            self.recorder.shutdown()
        except Exception:
            pass
        try:
            self.video_source.close()
        except Exception:
            pass
        try:
            self.supervisor.estop()
        except Exception:
            pass
        try:
            self.hal.close()
        except Exception:
            pass
        super().closeEvent(event)
