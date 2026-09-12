"""Tracker tab.

Layout adapts to the current operation mode:

* **Follow robot path** (default, "simple mode") — only the robot-relevant
  controls are visible. Operator sees mask sliders, cropped preview, four
  metrics, arrival threshold + magnitude, Start / Stop / Clear.

* **Push cells sequencer** — the cell-specific groups are revealed: cell
  mask parameters, cropped cell preview, cells list with per-status
  colours, push offset and approach threshold spinboxes.

All mask parameter spinboxes bind to ``config.yaml`` paths so per-session
tuning persists across launches. Selection mode radio (Robot / Cell) is only
shown when in Push-cells mode — in Follow-robot mode the click always
selects the robot.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from classes.gui.path_follow import PathFollowController
from classes.gui.tracker import MaskParams
from classes.gui.widgets import make_group


# ------------------------------------------------------------------ helpers

def _bgr_to_pixmap(bgr: np.ndarray) -> QtGui.QPixmap:
    if bgr is None or bgr.size == 0:
        return QtGui.QPixmap()
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    img = QtGui.QImage(
        rgb.tobytes(), w, h, w * 3, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(img)


def _blend_mask(bgr: np.ndarray, mask: np.ndarray,
                bgr_colour: tuple, alpha: float = 0.35) -> np.ndarray:
    """Return ``bgr`` with the mask painted on top at ``alpha`` opacity in
    the given colour. Mask is auto-resized to the crop shape (INTER_NEAREST
    to preserve edges); ``bgr`` isn't modified in place."""
    if mask is None or bgr is None or bgr.size == 0:
        return bgr
    h, w = bgr.shape[:2]
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    m = mask
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    out = bgr.copy()
    on = m > 0
    if not np.any(on):
        return out
    colour = np.array(bgr_colour, dtype=np.uint8)
    out[on] = (
        (1.0 - alpha) * out[on].astype(np.float32)
        + alpha * colour.astype(np.float32)
    ).astype(np.uint8)
    return out


class _MaskGroup(QtWidgets.QGroupBox):
    """Six spinboxes + Invert checkbox, bound to config."""

    changed = QtCore.pyqtSignal()

    def __init__(self, title: str, config, path_prefix: str, parent=None):
        super().__init__(title, parent)
        self.config = config
        self.path_prefix = path_prefix

        form = QtWidgets.QFormLayout(self)
        form.setContentsMargins(8, 12, 8, 8)

        def spin(minv, maxv, key, step=1):
            s = QtWidgets.QSpinBox()
            s.setRange(minv, maxv)
            s.setSingleStep(step)
            s.setValue(int(config.get(f"{path_prefix}.{key}", minv)))
            s.valueChanged.connect(self._on_changed)
            return s

        self.lower = spin(0, 255, "mask_lower")
        self.upper = spin(0, 255, "mask_upper")
        self.blur = spin(0, 30, "blur")
        self.dilation = spin(0, 30, "dilation")
        self.crop_length = spin(10, 400, "crop_length", step=5)
        self.invert = QtWidgets.QCheckBox("invert")
        self.invert.setChecked(
            bool(config.get(f"{path_prefix}.invert", False)))
        self.invert.toggled.connect(self._on_changed)

        form.addRow("Lower threshold", self.lower)
        form.addRow("Upper threshold", self.upper)
        form.addRow("Blur", self.blur)
        form.addRow("Dilation", self.dilation)
        form.addRow("Crop length (px)", self.crop_length)
        form.addRow("", self.invert)

    def _on_changed(self, *_a) -> None:
        # Write back to config so per-session tuning persists.
        self.config.set(f"{self.path_prefix}.mask_lower", self.lower.value())
        self.config.set(f"{self.path_prefix}.mask_upper", self.upper.value())
        self.config.set(f"{self.path_prefix}.blur", self.blur.value())
        self.config.set(f"{self.path_prefix}.dilation", self.dilation.value())
        self.config.set(
            f"{self.path_prefix}.crop_length", self.crop_length.value())
        self.config.set(
            f"{self.path_prefix}.invert", self.invert.isChecked())
        self.changed.emit()

    def params(self) -> MaskParams:
        return MaskParams(
            lower=self.lower.value(),
            upper=self.upper.value(),
            blur=self.blur.value(),
            dilation=self.dilation.value(),
            crop_length=self.crop_length.value(),
            invert=self.invert.isChecked(),
            min_blob_area_px=int(
                self.config.get(f"{self.path_prefix}.min_blob_area_px", 20)),
        )


# ------------------------------------------------------------------ panel

class TrackerPanel(QtWidgets.QWidget):

    def __init__(self, controller: PathFollowController, parent=None):
        super().__init__(parent)
        self.controller = controller
        cfg = controller.config

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)

        # --- Operation mode ------------------------------------------
        op_widget = QtWidgets.QWidget()
        op_row = QtWidgets.QHBoxLayout(op_widget)
        op_row.setContentsMargins(0, 0, 0, 0)
        self.op_simple = QtWidgets.QRadioButton("Follow robot path (simple)")
        self.op_push = QtWidgets.QRadioButton("Push cells sequencer")
        self.op_simple.setChecked(True)
        self.op_simple.toggled.connect(
            lambda on: on and self._set_op_mode(PathFollowController.OP_FOLLOW_ROBOT))
        self.op_push.toggled.connect(
            lambda on: on and self._set_op_mode(PathFollowController.OP_PUSH_CELLS))
        op_row.addWidget(self.op_simple)
        op_row.addWidget(self.op_push)
        outer.addWidget(make_group("Operation mode", op_widget))

        # --- Selection mode radios (only visible in push_cells) ------
        self.sel_widget = QtWidgets.QWidget()
        sel_row = QtWidgets.QHBoxLayout(self.sel_widget)
        sel_row.setContentsMargins(0, 0, 0, 0)
        self.sel_robot = QtWidgets.QRadioButton("Left-click selects robot")
        self.sel_cell = QtWidgets.QRadioButton("Left-click adds cell target")
        self.sel_robot.setChecked(True)
        self.sel_robot.toggled.connect(
            lambda on: on and controller.set_selection_mode("robot"))
        self.sel_cell.toggled.connect(
            lambda on: on and controller.set_selection_mode("cell"))
        sel_row.addWidget(self.sel_robot)
        sel_row.addWidget(self.sel_cell)
        self.sel_group = make_group("Selection mode", self.sel_widget)
        outer.addWidget(self.sel_group)

        # --- Cropped previews ---------------------------------------
        prev_widget = QtWidgets.QWidget()
        prev_row = QtWidgets.QHBoxLayout(prev_widget)
        prev_row.setContentsMargins(0, 0, 0, 0)
        self.robot_preview = QtWidgets.QLabel("no robot")
        self.robot_preview.setFixedSize(180, 180)
        self.robot_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.robot_preview.setStyleSheet(
            "background-color: #202020; color: #888;")
        self.cell_preview = QtWidgets.QLabel("no active cell")
        self.cell_preview.setFixedSize(180, 180)
        self.cell_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.cell_preview.setStyleSheet(
            "background-color: #202020; color: #888;")
        prev_row.addWidget(self.robot_preview)
        prev_row.addWidget(self.cell_preview)
        self.preview_group = make_group("Cropped previews", prev_widget)
        outer.addWidget(self.preview_group)

        # --- Robot metrics ------------------------------------------
        metrics_widget = QtWidgets.QWidget()
        m_lay = QtWidgets.QFormLayout(metrics_widget)
        m_lay.setContentsMargins(8, 8, 8, 8)
        self.lbl_diameter = QtWidgets.QLabel("—")
        self.lbl_speed = QtWidgets.QLabel("—")
        self.lbl_accel = QtWidgets.QLabel("—")
        self.lbl_blur = QtWidgets.QLabel("—")
        self.lbl_ambiguity = QtWidgets.QLabel("—")
        mono = "font-family: monospace;"
        self._mono = mono
        for lbl in (self.lbl_diameter, self.lbl_speed,
                    self.lbl_accel, self.lbl_blur, self.lbl_ambiguity):
            lbl.setStyleSheet(mono)
        m_lay.addRow("Diameter", self.lbl_diameter)
        m_lay.addRow("Speed", self.lbl_speed)
        m_lay.addRow("Acceleration", self.lbl_accel)
        m_lay.addRow("Blur (Laplacian var)", self.lbl_blur)
        # Runner-up/winner ratio of the correlation peaks. Near 1 means a
        # second, equally good candidate — an identical neighbour. This is
        # the failure that match confidence cannot see, because both peaks
        # score high.
        m_lay.addRow("Ambiguity", self.lbl_ambiguity)
        outer.addWidget(make_group("Robot metrics", metrics_widget))

        # --- Mask overlay toggles ----------------------------------
        mask_view_widget = QtWidgets.QWidget()
        mv_row = QtWidgets.QHBoxLayout(mask_view_widget)
        mv_row.setContentsMargins(0, 0, 0, 0)
        self.show_robot_mask_cb = QtWidgets.QCheckBox("Show robot mask")
        self.show_robot_mask_cb.toggled.connect(controller.set_show_robot_mask)
        self.show_cell_mask_cb = QtWidgets.QCheckBox("Show cell mask")
        self.show_cell_mask_cb.toggled.connect(controller.set_show_cell_mask)
        mv_row.addWidget(self.show_robot_mask_cb)
        mv_row.addWidget(self.show_cell_mask_cb)
        outer.addWidget(make_group("Mask overlays", mask_view_widget))

        # --- Mask parameter groups ---------------------------------
        self.robot_mask = _MaskGroup(
            "Robot mask", cfg, "tracker.robot")
        self.robot_mask.changed.connect(
            lambda: controller.set_robot_mask_params(self.robot_mask.params()))
        outer.addWidget(self.robot_mask)

        # --- Neural tracker toggles -----------------------------------
        neural_widget = QtWidgets.QWidget()
        n_lay = QtWidgets.QVBoxLayout(neural_widget)
        n_lay.setContentsMargins(0, 0, 0, 0)
        self.neural_robot_check = QtWidgets.QCheckBox(
            "Use neural tracker for robot (slower, learns object appearance)")
        self.neural_robot_check.setChecked(
            str(cfg.get("tracker.robot_tracker_backend", "template")) ==
            "adaptive")
        self.neural_robot_check.toggled.connect(
            lambda on: controller.set_robot_tracker_backend(
                "adaptive" if on else "template"))
        n_lay.addWidget(self.neural_robot_check)

        self.neural_cell_check = QtWidgets.QCheckBox(
            "Use neural tracker for cells (Push-cells mode)")
        self.neural_cell_check.setChecked(
            str(cfg.get("tracker.cell_tracker_backend", "template")) ==
            "adaptive")
        self.neural_cell_check.toggled.connect(
            lambda on: controller.set_cell_tracker_backend(
                "adaptive" if on else "template"))
        n_lay.addWidget(self.neural_cell_check)

        self.neural_status = QtWidgets.QLabel("running: —")
        self.neural_status.setStyleSheet(
            "color: #888; font-family: monospace; padding-left: 22px;")
        n_lay.addWidget(self.neural_status)
        self.neural_group = make_group("Tracker backend", neural_widget)
        outer.addWidget(self.neural_group)

        self.cell_mask = _MaskGroup(
            "Cell mask", cfg, "tracker.cell")
        self.cell_mask.changed.connect(
            lambda: controller.set_cell_mask_params(self.cell_mask.params()))
        outer.addWidget(self.cell_mask)

        # --- Cells list --------------------------------------------
        cells_widget = QtWidgets.QWidget()
        c_lay = QtWidgets.QVBoxLayout(cells_widget)
        c_lay.setContentsMargins(6, 6, 6, 6)
        self.cells_list = QtWidgets.QListWidget()
        self.cells_list.itemSelectionChanged.connect(self._on_cells_selection)
        c_lay.addWidget(self.cells_list)
        cb_row = QtWidgets.QHBoxLayout()
        self.btn_remove_cell = QtWidgets.QPushButton("Remove cell")
        self.btn_remove_cell.clicked.connect(self._on_remove_cell)
        self.btn_clear_cell_path = QtWidgets.QPushButton("Clear cell path")
        self.btn_clear_cell_path.clicked.connect(
            controller.clear_active_cell_path)
        cb_row.addWidget(self.btn_remove_cell)
        cb_row.addWidget(self.btn_clear_cell_path)
        c_lay.addLayout(cb_row)
        self.cells_group = make_group("Cells", cells_widget)
        outer.addWidget(self.cells_group)

        # --- Automation controls -----------------------------------
        auto_widget = QtWidgets.QWidget()
        a_lay = QtWidgets.QFormLayout(auto_widget)
        a_lay.setContentsMargins(8, 8, 8, 8)
        self.arrived = QtWidgets.QSpinBox()
        self.arrived.setRange(1, 200)
        self.arrived.setValue(int(cfg.get("path_follow.arrived_px", 15)))
        self.arrived.valueChanged.connect(controller.set_arrived_threshold)
        a_lay.addRow("Waypoint arrival (px)", self.arrived)

        self.push_offset = QtWidgets.QSpinBox()
        self.push_offset.setRange(5, 200)
        self.push_offset.setValue(
            int(cfg.get("path_follow.push_offset_px", 25)))
        self.push_offset.valueChanged.connect(controller.set_push_offset)
        a_lay.addRow("Push offset (px)", self.push_offset)

        self.approach = QtWidgets.QSpinBox()
        self.approach.setRange(1, 100)
        self.approach.setValue(
            int(cfg.get("path_follow.approach_arrived_px", 8)))
        self.approach.valueChanged.connect(
            controller.set_approach_arrived_threshold)
        a_lay.addRow("Approach arrival (px)", self.approach)

        # Rolling actually locomotes paramagnetic beads; static pull is the
        # legacy fallback.
        self.drive_combo = QtWidgets.QComboBox()
        self.drive_combo.addItems(["rolling", "static"])
        self.drive_combo.setCurrentText(
            str(cfg.get("path_follow.drive_mode", "rolling")))
        self.drive_combo.currentTextChanged.connect(self._on_drive_mode)
        a_lay.addRow("Drive", self.drive_combo)

        # Frequency + magnitude are the shared knobs owned by the Joystick
        # tab — displayed here read-only.
        self.shared_drive = QtWidgets.QLabel("—")
        self.shared_drive.setStyleSheet("font-family: monospace; color: #888;")
        a_lay.addRow("Roll / magnitude", self.shared_drive)
        self._refresh_shared_drive()
        try:
            cfg.on_change(
                "modes.mode_a", lambda *_a: self._refresh_shared_drive())
        except Exception:
            pass

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_start.clicked.connect(lambda: controller.set_running(True))
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.clicked.connect(lambda: controller.set_running(False))
        self.btn_clear_all = QtWidgets.QPushButton("Clear all")
        self.btn_clear_all.clicked.connect(controller.clear)
        for b in (self.btn_start, self.btn_stop, self.btn_clear_all):
            btn_row.addWidget(b)
        a_lay.addRow(btn_row)

        outer.addWidget(make_group("Automation", auto_widget))

        # --- Status --------------------------------------------------
        self.status = QtWidgets.QLabel("(no robot selected)")
        self.status.setStyleSheet(
            "font-family: monospace; padding: 4px;")
        outer.addWidget(self.status)

        # --- Legend --------------------------------------------------
        legend = QtWidgets.QLabel(
            "<b>Left click</b> selects (robot or cell) · "
            "<b>Right click</b> adds a waypoint to the active target · "
            "<b>Middle click</b> clears."
        )
        legend.setStyleSheet("color: #999;")
        legend.setWordWrap(True)
        outer.addWidget(legend)

        outer.addStretch(1)

        # --- Wire controller → UI ------------------------------------
        controller.stateChanged.connect(self._on_state)
        controller.croppedFrameChanged.connect(self._on_cropped)
        controller.runningChanged.connect(self._on_running)

        # Apply the initial simple-mode layout.
        self._apply_operation_mode_ui()

    # ---- adaptive layout -------------------------------------------

    def _set_op_mode(self, mode: str) -> None:
        self.controller.set_operation_mode(mode)
        self._apply_operation_mode_ui()

    def _on_drive_mode(self, mode: str) -> None:
        self.controller.config.set("path_follow.drive_mode", mode)
        self.controller.set_drive_mode(mode)
        self._refresh_shared_drive()

    def _refresh_shared_drive(self) -> None:
        cfg = self.controller.config
        freq = float(cfg.get("modes.mode_a.freq_default", 1.0))
        mag = float(cfg.get("modes.mode_a.magnitude_default", 1.0))
        text = f"{freq:.1f} Hz @ mag {mag:.2f}  (set on the Joystick tab)"
        if self.drive_combo.currentText() == "static":
            text += "  — freq unused in static drive"
        self.shared_drive.setText(text)

    def _apply_operation_mode_ui(self) -> None:
        is_push = (self.controller.operation_mode ==
                   PathFollowController.OP_PUSH_CELLS)
        # Cell-only widgets — hide in simple mode.
        self.sel_group.setVisible(is_push)
        self.cell_mask.setVisible(is_push)
        self.cells_group.setVisible(is_push)
        self.cell_preview.setVisible(is_push)
        self.push_offset.setVisible(is_push)
        self.approach.setVisible(is_push)
        self.show_cell_mask_cb.setVisible(is_push)
        # Robot neural toggle is always available; cell toggle only in push.
        self.neural_cell_check.setVisible(is_push)
        # If we're leaving push_cells, force selection back to robot and
        # kill the cell mask overlay (nothing to see when there are no cells).
        if not is_push:
            self.sel_robot.setChecked(True)
            self.controller.set_selection_mode("robot")
            if self.show_cell_mask_cb.isChecked():
                self.show_cell_mask_cb.setChecked(False)

    # ---- signal handlers -------------------------------------------

    def _on_state(self, payload: dict) -> None:
        robot = payload.get("robot")
        cells = payload.get("cells") or []
        active_idx = payload.get("active_cell_idx", -1)
        running = payload.get("running", False)
        op = payload.get("operation_mode")

        # Update the neural backend status line.
        r_backend = payload.get("robot_tracker_backend", "template")
        c_backend = payload.get("cell_tracker_backend", "template")
        eff = payload.get("adaptive_effective_backend", "—")
        target = payload.get("adaptive_effective_target", "—")
        any_adaptive = ("adaptive" in (r_backend, c_backend))
        if not any_adaptive:
            self.neural_status.setText("running: template (fast) — both")
        elif eff in ("—", "none"):
            self.neural_status.setText(
                "running: (select a target to load weights)")
        else:
            parts = []
            if r_backend == "adaptive":
                parts.append("robot: " + eff)
            if c_backend == "adaptive":
                parts.append("cell: " + eff)
            self.neural_status.setText(
                f"running: {', '.join(parts)} · {target}")

        # Robot metrics
        if robot is None:
            self.lbl_diameter.setText("—")
            self.lbl_speed.setText("—")
            self.lbl_accel.setText("—")
            self.lbl_blur.setText("—")
            self.lbl_ambiguity.setText("—")
            self.lbl_ambiguity.setStyleSheet(self._mono)
        else:
            d_um = float(robot.get("diameter_um", 0.0))
            s_um = float(robot.get("speed_ums", 0.0))
            a_um = float(robot.get("accel_ums2", 0.0))
            self.lbl_diameter.setText(f"{d_um:.2f} µm")
            self.lbl_speed.setText(f"{s_um:.2f} µm/s")
            self.lbl_accel.setText(f"{a_um:.2f} µm/s²")
            self.lbl_blur.setText(f"{robot.get('blur', 0.0):.1f}")
            amb = float(robot.get("ambiguity", 0.0))
            contested = bool(robot.get("contested"))
            n_dist = int(robot.get("n_distractors", 0))
            self.lbl_ambiguity.setText(
                f"{amb:.2f}" + (f"  CONTESTED — coasting" if contested
                                else f"  ({n_dist} others tracked)"))
            self.lbl_ambiguity.setStyleSheet(
                self._mono + ("color: #f39c12;" if contested else ""))

        # Cells list
        self._refresh_cells_list(cells, active_idx)

        # Status line
        if robot is None:
            self.status.setText("no robot selected — left-click on the bead")
        elif op == PathFollowController.OP_FOLLOW_ROBOT:
            n = int(robot.get("n_waypoints", 0))
            k = int(robot.get("target_idx", 0)) + 1
            if n == 0:
                self.status.setText("robot selected — right-click to draw path")
            elif running:
                self.status.setText(f"following {k} of {n}")
            elif robot.get("lost_streak", 0) > 5:
                self.status.setText("robot tracker lost — clear and reselect")
            else:
                self.status.setText(f"ready ({n} waypoints, Start to follow)")
        else:
            if not cells:
                self.status.setText(
                    "no cells — switch selection to Cell and left-click")
            elif running:
                if 0 <= active_idx < len(cells):
                    c = cells[active_idx]
                    self.status.setText(
                        f"cell {active_idx + 1} of {len(cells)}: "
                        f"{c['status']}, wpt {c['target_idx'] + 1}/{c['n_waypoints']}")
                else:
                    self.status.setText("push sequencer running")
            else:
                self.status.setText(
                    f"{len(cells)} cell(s) armed — Start to run")

    def _refresh_cells_list(self, cells: list, active_idx: int) -> None:
        current = self.cells_list.currentRow()
        self.cells_list.blockSignals(True)
        self.cells_list.clear()
        for i, c in enumerate(cells):
            n = c.get("n_waypoints", 0)
            k = c.get("target_idx", 0)
            status = c.get("status", "PENDING")
            item = QtWidgets.QListWidgetItem(
                f"C{i + 1}  {status:<11} {min(k, n)}/{n} wpts")
            colours = {
                "PENDING": "#888",
                "IN_PROGRESS": "#f0a020",
                "DONE": "#4c8a4c",
                "LOST": "#c04040",
            }
            item.setForeground(QtGui.QColor(colours.get(status, "#ccc")))
            self.cells_list.addItem(item)
        target_row = active_idx if 0 <= active_idx < len(cells) else current
        if 0 <= target_row < self.cells_list.count():
            self.cells_list.setCurrentRow(target_row)
        self.cells_list.blockSignals(False)

    def _on_cells_selection(self) -> None:
        row = self.cells_list.currentRow()
        if row >= 0:
            self.controller.set_active_cell(row)

    def _on_remove_cell(self) -> None:
        row = self.cells_list.currentRow()
        if row >= 0:
            self.controller.remove_cell(row)

    def _on_cropped(self, kind: str, frame: np.ndarray, mask) -> None:
        # Optionally tint the mask over the crop so the operator can see
        # what the current mask picks up in the ROI. Robot preview follows
        # the "Show robot mask" checkbox; cell preview follows the cell one.
        show = False
        rgb = None
        if kind == "robot":
            target = self.robot_preview
            show = self.show_robot_mask_cb.isChecked()
            rgb = (60, 60, 220)  # BGR — matches red tint in the video overlay
        elif kind.startswith("cell:"):
            target = self.cell_preview
            show = self.show_cell_mask_cb.isChecked()
            rgb = (220, 200, 60)  # BGR — cyan-ish
        else:
            return
        if show and mask is not None:
            blended = _blend_mask(frame, mask, rgb)
        else:
            blended = frame
        pix = _bgr_to_pixmap(blended)
        scaled = pix.scaled(
            target.width(), target.height(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation)
        target.setPixmap(scaled)

    def _on_running(self, running: bool) -> None:
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
