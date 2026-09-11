"""PathFollowController — glue between the video source, per-object trackers,
and the MotionController.

Two operation modes:

* **follow_robot** — the paramagnetic robot walks its own trajectory. Simple:
  each frame, compute vector from robot to the current waypoint, push it as
  a Mode-B pull direction. Advance target when |err| < arrived_px.

* **push_cells** — cell push automation. The operator marks non-magnetic
  cells and per-cell paths. The robot goes to each cell, positions itself
  behind the cell relative to the cell's next waypoint (APPROACH), then
  pulls itself toward that waypoint so the cell is mechanically dragged
  along (PUSH). Waypoints advance when the *cell* arrives at each; the
  sequencer moves to the next cell when all of the current cell's
  waypoints are DONE.

APPROACH↔PUSH is a stateless recomputation each frame — if the robot drifts
off the push side of the cell during a push, next frame flips back to
APPROACH automatically.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import threading
import time
from typing import Optional

import cv2
import numpy as np
from PyQt5 import QtCore

from classes.gui.adaptive_tracker import AdaptiveTracker
from classes.gui.distractors import DistractorField, DistractorWorker
from classes.gui.motion_filter import ConstantVelocityKF
from classes.gui.recording import resolve_output_dir
from classes.gui.robot_state import CellState, RobotState, TargetStatus
from classes.gui.tracker import (
    Detection,
    MaskParams,
    MotionPrior,
    ObjectTracker,
    _apply_mask,
)
from classes.motion_controller import (
    Mode,
    MotionController,
    Source,
    roll_axis_for,
)


class _TrackerAsync:
    """Serialises tracker calls onto a background thread so slow inference
    (SAM 2, VitTrack init on cv2 5.0, MIL on huge frames) doesn't freeze
    the Qt event loop.

    * ``submit_init(frame, click, t)`` — runs initialize on the worker. The
      result comes back through ``on_result`` on the main Qt thread via
      the ``resultReady`` signal on the owning controller.
    * ``submit_update(frame, t, prior)`` — like init but for per-frame
      updates. If the worker is still busy, replaces the pending frame
      (drop-old) so we never build a backlog. The capture time and motion
      prior travel with the frame: the prior was computed for that frame
      specifically, and the timestamp is the camera's, not the worker's.
    * Frame updates are dropped silently while init is in flight.

    Everything communicates through the owning controller's Qt signals,
    which are automatically queued across threads.
    """

    def __init__(self, tracker, on_init_done, on_update_done, log_fn):
        self._tracker = tracker
        self._on_init_done = on_init_done
        self._on_update_done = on_update_done
        self._log = log_fn
        self._init_takes_t = self._accepts(tracker.initialize, "t")
        self._update_takes_prior = self._accepts(tracker.update, "prior")

        self._cond = threading.Condition()
        self._pending_update: Optional[np.ndarray] = None
        self._pending_init: Optional[tuple] = None    # (frame, click_xy)
        self._stopped = False
        self._initialised = False
        self._busy_init = False

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ---- external ---------------------------------------------------

    @property
    def initialised(self) -> bool:
        return self._initialised

    def submit_init(self, frame: np.ndarray, click_xy, t: float = 0.0) -> None:
        with self._cond:
            self._pending_init = (frame, click_xy, t)
            self._busy_init = True
            self._cond.notify()

    def submit_update(self, frame: np.ndarray, t: float = 0.0,
                      prior=None) -> None:
        with self._cond:
            if self._busy_init or not self._initialised:
                return
            # Drop-old: overwrite the pending frame if one's already queued.
            # The capture time and prior travel WITH the frame — the prior
            # was computed for this frame specifically, and reusing a
            # stale one against a newer frame would defeat the point.
            self._pending_update = (frame, t, prior)
            self._cond.notify()

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify()
        self._thread.join(timeout=1.0)

    @property
    def raw_tracker(self):
        return self._tracker

    # ---- worker loop ------------------------------------------------

    def _loop(self) -> None:
        while True:
            with self._cond:
                while (self._pending_init is None
                       and self._pending_update is None
                       and not self._stopped):
                    self._cond.wait()
                if self._stopped:
                    return
                if self._pending_init is not None:
                    task = ("init", self._pending_init)
                    self._pending_init = None
                    # Drop any queued updates — they'd be against a
                    # stale-tracker frame anyway.
                    self._pending_update = None
                else:
                    task = ("update", self._pending_update)
                    self._pending_update = None

            kind, payload = task
            try:
                if kind == "init":
                    frame, click_xy, t = payload
                    det = self._call_init(frame, click_xy, t)
                else:
                    frame, t, prior = payload
                    det = self._call_update(frame, t, prior)
            except Exception as e:
                self._log(f"_TrackerAsync: {kind} raised: {e}")
                det = None

            if kind == "init":
                with self._cond:
                    self._busy_init = False
                    self._initialised = det is not None
                try:
                    self._on_init_done(det)
                except Exception:
                    pass
            else:
                try:
                    self._on_update_done(det, t)
                except Exception:
                    pass

    # ---- backend-tolerant invocation ---------------------------------
    # AdaptiveTracker keeps the original (frame) / (frame, click) shape —
    # it has no correlation surface, so priors and capture times mean
    # nothing to it. Capability is resolved from the signature ONCE at
    # construction; catching TypeError at the call site would also
    # swallow TypeErrors raised inside the tracker and silently re-run it.

    @staticmethod
    def _accepts(fn, name: str) -> bool:
        try:
            return name in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return False

    def _call_init(self, frame, click_xy, t):
        if self._init_takes_t:
            return self._tracker.initialize(frame, click_xy, t)
        return self._tracker.initialize(frame, click_xy)

    def _call_update(self, frame, t, prior):
        if self._update_takes_prior:
            return self._tracker.update(frame, t, prior)
        return self._tracker.update(frame)


def _mask_params_from_config(cfg, path: str) -> MaskParams:
    return MaskParams(
        lower=int(cfg.get(f"{path}.mask_lower", 0)),
        upper=int(cfg.get(f"{path}.mask_upper", 128)),
        blur=int(cfg.get(f"{path}.blur", 0)),
        dilation=int(cfg.get(f"{path}.dilation", 0)),
        crop_length=int(cfg.get(f"{path}.crop_length", 40)),
        invert=bool(cfg.get(f"{path}.invert", False)),
        min_blob_area_px=int(cfg.get(f"{path}.min_blob_area_px", 20)),
    )


class PathFollowController(QtCore.QObject):

    stateChanged = QtCore.pyqtSignal(dict)                  # panel readouts
    # (kind, bgr, mask_or_none) — mask lets the panel tint the cropped
    # preview so the operator sees what the current mask picks up in the ROI.
    croppedFrameChanged = QtCore.pyqtSignal(str, np.ndarray, object)

    # Internal — trackers finish async on a background thread. These
    # cross-thread signals marshal detection results back to the main Qt
    # thread so state updates and UI emissions are always thread-safe.
    _initResult = QtCore.pyqtSignal(str, object, object, object)
    # (target_id, det_or_none, click_xy, source_frame)
    # (target_id, det_or_none, capture_time)
    _updateResult = QtCore.pyqtSignal(str, object, float)
    runningChanged = QtCore.pyqtSignal(bool)
    overlayChanged = QtCore.pyqtSignal()

    OP_FOLLOW_ROBOT = "follow_robot"
    OP_PUSH_CELLS = "push_cells"

    def __init__(self, motion: MotionController, config,
                 log_fn=print, parent=None):
        super().__init__(parent)
        self.motion = motion
        self.config = config
        self.log = log_fn

        # State
        self.robot: Optional[RobotState] = None
        self.robot_tracker: Optional[ObjectTracker] = None
        self.cells: list = []
        self.cell_trackers: list = []
        self.active_cell_idx: int = -1
        self.selection_mode: str = "robot"       # "robot" | "cell"
        self.operation_mode: str = self.OP_FOLLOW_ROBOT
        self.running: bool = False

        # Config-driven tunables (cached for the hot loop).
        self.um_per_pixel = float(config.get("tracker.um_per_pixel", 0.335))
        self.memory_frames = int(config.get("tracker.memory_frames", 15))
        self.min_match_confidence = float(
            config.get("tracker.min_match_confidence", 0.5))
        # Seconds, not frames. The async worker drops frames under load, so
        # a frame count is an unpredictable amount of real time — and for
        # the loss timeout that meant the safety stop could fire seconds
        # late with the robot running open-loop the whole while.
        self.lost_timeout_s = float(config.get("tracker.lost_timeout_s", 1.5))
        self.template_refresh_s = float(
            config.get("tracker.template_refresh_s", 1.5))
        self.copy_frames = bool(config.get("tracker.copy_frames", True))

        # Motion prior / ambiguity knobs. `motion_prior` is the master
        # switch: off = the tracker behaves exactly as it did before this
        # feature (search window on the last measured position, plain
        # argmax, every detection fused). First thing to try if the box
        # misbehaves on the rig — it isolates the whole stack in one flag.
        self.motion_prior_enabled = bool(
            config.get("tracker.motion_prior", True))
        self.search_half_px = int(config.get("tracker.search_half_px", 0))
        self.ambiguity_enter = float(
            config.get("tracker.ambiguity_enter", 0.85))
        self.ambiguity_exit = float(config.get("tracker.ambiguity_exit", 0.65))
        self.contested_max_s = float(config.get("tracker.contested_max_s", 1.5))
        self.prior_sigma_floor_px = float(
            config.get("tracker.prior_sigma_floor_px", 3.0))
        self.kf_accel_noise = float(
            config.get("tracker.kf_accel_noise_px_s2", 400.0))
        self.kf_meas_noise = float(config.get("tracker.kf_meas_noise_px", 2.0))
        self.kf_velocity_tau_s = float(
            config.get("tracker.kf_velocity_tau_s", 0.15))
        self.kf_coast_inflate = float(
            config.get("tracker.kf_coast_inflate", 4.0))

        self.distractors_enabled = bool(
            config.get("tracker.distractors.enabled", True))
        self.distractor_rate_hz = float(
            config.get("tracker.distractors.rate_hz", 8.0))
        self.distractor_gate_px = float(
            config.get("tracker.distractors.gate_px", 40.0))
        self.distractor_max_blobs = int(
            config.get("tracker.distractors.max_blobs", 64))

        self.arrived_px = float(config.get("path_follow.arrived_px", 15))
        self.approach_arrived_px = float(
            config.get("path_follow.approach_arrived_px", 8))
        self.push_offset_px = float(config.get("path_follow.push_offset_px", 25))
        self.drive_mode = str(config.get("path_follow.drive_mode", "rolling"))
        self.stall_frames = int(config.get("path_follow.stall_frames", 300))
        self.contested_timeout_s = float(
            config.get("path_follow.contested_timeout_s", 3.0))
        self.event_log_enabled = bool(
            config.get("path_follow.event_log", True))

        # Image → world mapping comes from the Frame Calibration fit — the
        # same matrix the joystick uses. Identity default = plain y-flip.
        self._reload_frame_matrix()
        try:
            config.on_change(
                "calibration.screen_to_world_2x2",
                lambda *_a: self._reload_frame_matrix())
        except Exception:
            pass
        # Raw world→screen response, px per (unit world command × Hz × s).
        # This is the ONLY thing in the codebase carrying an absolute pixel
        # speed scale — screen_to_world_2x2 is normalized to direction-only
        # — so it's what converts a commanded direction into an expected
        # pixel velocity for the filter. Absent (never calibrated) → the
        # filter runs as plain constant velocity, which is still correct,
        # just less sharp.
        self._screen_response_2x2 = None
        self._static_response_2x2 = None
        self._static_response_mag = 1.0
        self._reload_response_matrix()
        for _key in ("calibration.screen_response_2x2",
                     "calibration.static_response_2x2",
                     "calibration.static_response_mag"):
            try:
                config.on_change(
                    _key, lambda *_a: self._reload_response_matrix())
            except Exception:
                pass

        # Snapshot of the current mask params.
        self.robot_mask_params = _mask_params_from_config(config, "tracker.robot")
        self.cell_mask_params = _mask_params_from_config(config, "tracker.cell")

        # Per-object tracker choice. "template" = fast path (ObjectTracker);
        # "adaptive" = neural / learned model (AdaptiveTracker with cv2's
        # Nano/Vit/MIL). Both robot and cells can independently flip.
        self.robot_tracker_backend: str = str(
            config.get("tracker.robot_tracker_backend", "template"))
        self.cell_tracker_backend: str = str(
            config.get("tracker.cell_tracker_backend", "template"))
        self.adaptive_backend_pref: str = str(
            config.get("tracker.adaptive_backend", "auto"))
        self.adaptive_target_pref: str = str(
            config.get("tracker.adaptive_dnn_target", "auto"))
        self.models_dir: str = str(config.get("tracker.models_dir", "models"))
        self.nano_url_prefix: str = str(
            config.get("tracker.nano_weights_url_prefix", ""))
        self.sam2_size: str = str(config.get("tracker.sam2_size", "tiny"))
        # Effective backend / target populated on first successful adaptive
        # init so the panel can display them.
        self.adaptive_effective_backend: str = "—"
        self.adaptive_effective_target: str = "—"

        # Wire the cross-thread result signals to main-thread slots.
        self._initResult.connect(self._on_init_result)
        self._updateResult.connect(self._on_update_result)
        # Pending selection metadata for each in-flight init, keyed by
        # target_id. Populated in select_at; drained in _on_init_result.
        self._pending_selections: dict = {}
        # Generation counters so we can identify stale callbacks from a
        # tracker that's already been superseded. Each new robot selection
        # bumps _robot_gen; each new cell selection bumps _cell_gen. The
        # target_id in the async callbacks carries the current generation.
        self._robot_gen: int = 0
        self._cell_gen: int = 0

        # Mask overlay toggles + latest computed masks (full-frame binary
        # arrays). ``None`` when not shown.
        self.show_robot_mask: bool = False
        self.show_cell_mask: bool = False
        self._latest_robot_mask: Optional[np.ndarray] = None
        self._latest_cell_mask: Optional[np.ndarray] = None

        # Emit throttling for stateChanged
        self._last_state_emit = 0.0

        # --- motion prior / contested state ---------------------------
        # One filter per tracked robot. Predicts where the bead should be
        # on the next frame so the correlation surface can be weighted
        # toward that peak instead of picking the marginally-higher one.
        self.robot_kf: Optional[ConstantVelocityKF] = None
        self._robot_kf_t: float = 0.0
        self.robot_contested: bool = False
        self.robot_ambiguity: float = 0.0
        self._contested_since: Optional[float] = None
        self._contested_logged: bool = False
        self._event_log_path: Optional[str] = None

        # Global low-rate blob tracking, so the per-object tracker has a
        # representation of the OTHER beads and can suppress them.
        self._distractors: Optional[DistractorWorker] = None
        self._distractor_snapshot: list = []

    # ---- selection API ------------------------------------------------

    def set_selection_mode(self, mode: str) -> None:
        if mode not in ("robot", "cell"):
            return
        self.selection_mode = mode

    def set_operation_mode(self, mode: str) -> None:
        if mode not in (self.OP_FOLLOW_ROBOT, self.OP_PUSH_CELLS):
            return
        self.operation_mode = mode
        # Reset running so an operator gesture is required to start.
        if self.running:
            self.set_running(False)

    def set_active_cell(self, idx: int) -> None:
        if 0 <= idx < len(self.cells):
            self.active_cell_idx = idx

    def select_at(self, x: int, y: int, frame: Optional[np.ndarray]) -> None:
        """Kick off an async tracker init. The initial state setup happens
        in ``_on_init_result`` on the main thread once the worker returns
        — that way SAM 2's slow first-inference doesn't freeze the UI."""
        if frame is None:
            self.log("PathFollowController: no frame available for select")
            return
        click = (int(x), int(y))
        # Snapshot the frame — the video source may free / overwrite its
        # buffer before the worker thread gets to use it.
        frame_snap = frame.copy()
        if self.selection_mode == "robot":
            # Stop and drop the previous robot tracker (and its state)
            # immediately so its in-flight worker updates don't land on
            # the new selection and pull the box back to the old position.
            if self.robot_tracker is not None:
                try:
                    self.robot_tracker.stop()
                except Exception:
                    pass
            self.robot_tracker = None
            self.robot = None
            # The filter belongs to the old bead; a fresh selection starts
            # from zero velocity and wide covariance.
            self.robot_kf = None
            self.robot_contested = False
            self.robot_ambiguity = 0.0
            self._contested_since = None
            self._robot_gen += 1
            target_id = f"robot:{self._robot_gen}"
            raw = self._build_robot_tracker()
        else:
            self._cell_gen += 1
            target_id = f"cell:{len(self.cells)}:{self._cell_gen}"
            raw = self._build_cell_tracker()

        self.log(
            f"PathFollowController: selecting {target_id} — loading tracker "
            "(may take a few seconds on first call for neural backends) …")

        def on_init(det, _tid=target_id, _f=frame_snap, _c=click):
            self._initResult.emit(_tid, det, _c, _f)

        def on_update(det, t, _tid=target_id):
            self._updateResult.emit(_tid, det, t)

        async_tracker = _TrackerAsync(raw, on_init, on_update, self.log)
        self._pending_selections[target_id] = {
            "raw": raw,
            "async": async_tracker,
            "selection_mode": self.selection_mode,
        }
        async_tracker.submit_init(frame_snap, click, time.monotonic())
        self._ensure_distractors()

    def _on_init_result(self, target_id: str, det,
                        click_xy, frame_snap) -> None:
        """Main-thread slot — completes the selection begun in ``select_at``."""
        info = self._pending_selections.pop(target_id, None)
        if info is None:
            return
        raw = info["raw"]
        async_tracker = info["async"]

        # Adaptive backend refused (deps missing / weights failed / …).
        # Retry synchronously via the fast template tracker so the click
        # doesn't just disappear.
        if det is None and isinstance(raw, AdaptiveTracker):
            self.log(
                f"PathFollowController: adaptive tracker refused for "
                f"{target_id}; falling back to template")
            async_tracker.stop()
            mask_params = (self.robot_mask_params
                           if info["selection_mode"] == "robot"
                           else self.cell_mask_params)
            fallback = self._build_template_tracker(mask_params)
            det = fallback.initialize(frame_snap, click_xy)
            if det is None:
                self.log(
                    f"PathFollowController: {target_id} selection failed "
                    "on template fallback too")
                return

            def on_update(d, t, _tid=target_id):
                self._updateResult.emit(_tid, d, t)

            async_tracker = _TrackerAsync(
                fallback,
                on_init_done=lambda _d: None,
                on_update_done=on_update,
                log_fn=self.log)
            async_tracker._initialised = True
            async_tracker._busy_init = False
            raw = fallback

        if det is None:
            self.log(
                f"PathFollowController: {target_id} selection: no detection")
            return

        if isinstance(raw, AdaptiveTracker):
            self.adaptive_effective_backend = raw.effective_backend
            self.adaptive_effective_target = raw.effective_target

        if target_id.startswith("robot:"):
            gen = self._parse_gen(target_id, "robot:")
            if gen != self._robot_gen:
                # Superseded by a newer click. Discard.
                async_tracker.stop()
                return
            self.robot_tracker = async_tracker
            self.robot = RobotState(
                crop_length=self.robot_mask_params.crop_length,
                um_per_pixel=self.um_per_pixel,
                memory=self.memory_frames)
            t_sel = float(getattr(det, "t_capture", 0.0)) or time.monotonic()
            self.robot.record_frame(
                t_sel, det.pos, det.area_px, det.blur)
            # Seed the filter at the click. Velocity starts at zero with a
            # wide covariance, so the first few frames are effectively
            # unweighted and the prior only tightens once motion is real.
            self.robot_kf = ConstantVelocityKF(
                det.pos, accel_noise_px_s2=self.kf_accel_noise,
                meas_noise_px=self.kf_meas_noise,
                velocity_tau_s=self.kf_velocity_tau_s)
            self._robot_kf_t = t_sel
            self.robot_contested = False
            self.robot_ambiguity = 0.0
            self.log(
                f"PathFollowController: robot selected at "
                f"{det.pos[0]:.1f}, {det.pos[1]:.1f} via {det.source} "
                f"(area {det.area_px:.0f} px)")
            if det.source == "click_seed":
                self.log(
                    "  ^ WARNING: no mask blob at the click, so the template "
                    "is whatever pixels were under the cursor — possibly "
                    "background. Expect the box to sit still or wander. "
                    "Tune the robot mask until the bead shows as a blob, "
                    "then re-select.")
            self.croppedFrameChanged.emit(
                "robot", det.cropped_bgr, det.cropped_mask)
        elif target_id.startswith("cell:"):
            cell = CellState(
                crop_length=self.cell_mask_params.crop_length,
                um_per_pixel=self.um_per_pixel,
                memory=self.memory_frames,
                uid=self._cell_uid_of(target_id))
            t_sel = float(getattr(det, "t_capture", 0.0)) or time.monotonic()
            cell.record_frame(t_sel, det.pos, det.area_px, det.blur)
            self.cells.append(cell)
            self.cell_trackers.append(async_tracker)
            self.active_cell_idx = len(self.cells) - 1
            self.log(
                f"PathFollowController: cell C{self.active_cell_idx + 1} "
                f"selected at {det.pos[0]:.1f}, {det.pos[1]:.1f} "
                f"via {det.source}")
            self.croppedFrameChanged.emit(
                f"cell:{self.active_cell_idx}",
                det.cropped_bgr, det.cropped_mask)
        self._emit_state()
        self.overlayChanged.emit()

    def _parse_gen(self, target_id: str, prefix: str) -> int:
        try:
            return int(target_id[len(prefix):])
        except (ValueError, IndexError):
            return -1

    @staticmethod
    def _cell_uid_of(target_id: str) -> int:
        """Generation number out of a ``cell:<idx>:<gen>`` id.

        The generation is the cell's stable identity; the index baked into
        the id is only a display hint from click time and goes stale the
        moment a cell is removed.
        """
        parts = target_id.split(":")
        if len(parts) < 3:
            return -1
        try:
            return int(parts[2])
        except ValueError:
            return -1

    def _cell_index_by_uid(self, uid: int) -> int:
        """Current list position of the cell with this uid, or -1 if gone."""
        if uid < 0:
            return -1
        for i, cell in enumerate(self.cells):
            if cell.uid == uid:
                return i
        return -1

    def _last_seen_age(self, target, t: float) -> float:
        """Seconds since this target's last accepted detection."""
        if not target.times:
            return 0.0
        return max(0.0, t - float(target.times[-1]))

    def _on_update_result(self, target_id: str, det, t_capture: float = 0.0
                          ) -> None:
        """Main-thread slot for per-frame async tracker results. Stale
        callbacks from a superseded tracker are recognized by their
        generation counter and dropped."""
        t = float(t_capture) if t_capture else time.monotonic()
        if target_id.startswith("robot:"):
            gen = self._parse_gen(target_id, "robot:")
            if gen != self._robot_gen or self.robot is None:
                return
            if det is not None:
                self._ingest_robot_detection(det, t)
            else:
                self.robot.lost_streak += 1
                # Coast the filter so the next prior stays meaningful and
                # its sigma widens to reflect the growing uncertainty.
                if self.robot_kf is not None:
                    self.robot_kf.step(
                        max(0.0, t - self._robot_kf_t),
                        self._commanded_velocity_px_s(), None,
                        inflate=self.kf_coast_inflate)
                    self._robot_kf_t = t
                if (self._last_seen_age(self.robot, t) > self.lost_timeout_s
                        and self.running):
                    self.log(
                        f"PathFollowController: robot lost for "
                        f"{self.lost_timeout_s:.1f}s — stopping")
                    self.set_running(False)
        elif target_id.startswith("cell:"):
            # target_id is "cell:<idx>:<gen>". Resolve by uid (the gen), not
            # by that index: remove_cell() shifts every later entry, so a
            # callback queued before the removal would otherwise write one
            # cell's position, area and blur onto a different cell — and
            # mark the wrong one LOST. A uid with no match means the cell
            # was removed while this detection was in flight; drop it.
            idx = self._cell_index_by_uid(self._cell_uid_of(target_id))
            if idx < 0:
                return
            cell = self.cells[idx]
            if det is not None:
                cell.record_frame(t, det.pos, det.area_px, det.blur)
                if idx == self.active_cell_idx:
                    self.croppedFrameChanged.emit(
                        f"cell:{idx}", det.cropped_bgr, det.cropped_mask)
            else:
                cell.lost_streak += 1
                if self._last_seen_age(cell, t) > self.lost_timeout_s:
                    cell.status = TargetStatus.LOST
        self._emit_state()
        self.overlayChanged.emit()

    def _ingest_robot_detection(self, det, t: float) -> None:
        """Fuse (or deliberately refuse to fuse) one robot detection."""
        self.robot_ambiguity = float(getattr(det, "ambiguity", 0.0))
        contested = bool(getattr(det, "contested", False))
        dt = max(0.0, t - self._robot_kf_t)
        u = self._commanded_velocity_px_s()

        if self.robot_kf is None:
            self.robot_kf = ConstantVelocityKF(
                det.pos, accel_noise_px_s2=self.kf_accel_noise,
                meas_noise_px=self.kf_meas_noise,
                velocity_tau_s=self.kf_velocity_tau_s)
            pos = det.pos
        elif contested:
            # Two equally good answers. Trusting the marginally-higher peak
            # is exactly the coin-flip that swaps beads, so coast on the
            # prediction instead and let the extra process noise widen the
            # next prior.
            pos = self.robot_kf.step(dt, u, None,
                                     inflate=self.kf_coast_inflate)
            # Push the coasted position back into the tracker: otherwise
            # its search window stays centred on the contested peak and
            # walks onto the neighbour over the following frames even
            # though the position we REPORTED was right.
            try:
                raw = self.robot_tracker.raw_tracker
                if hasattr(raw, "set_position"):
                    raw.set_position(pos)
            except Exception:
                pass
        else:
            pos = self.robot_kf.step(dt, u, det.pos)
        self._robot_kf_t = t

        self._update_contested_state(contested, det, pos, t)
        self.robot.record_frame(t, pos, det.area_px, det.blur)
        self.croppedFrameChanged.emit(
            "robot", det.cropped_bgr, det.cropped_mask)

    def _update_contested_state(self, contested: bool, det, pos,
                                t: float) -> None:
        """Enter/leave the contested state and surface it."""
        if contested and not self.robot_contested:
            self._contested_since = t
            self._log_event({
                "event": "contested_enter", "t": t,
                "ambiguity": self.robot_ambiguity,
                "pos": [float(pos[0]), float(pos[1])],
                "second_pos": (list(det.second_pos)
                               if getattr(det, "second_pos", None) else None),
                "confidence": float(getattr(det, "confidence", 0.0)),
            })
        elif self.robot_contested and not contested:
            self._log_event({
                "event": "contested_exit", "t": t,
                "ambiguity": self.robot_ambiguity,
                "pos": [float(pos[0]), float(pos[1])],
                "second_pos": None,
                "held_s": (t - self._contested_since
                           if self._contested_since else 0.0),
            })
            self._contested_since = None
        self.robot_contested = contested

    def add_waypoint(self, x: int, y: int) -> None:
        if self.operation_mode == self.OP_FOLLOW_ROBOT:
            if self.robot is None:
                return
            self.robot.push_waypoint(x, y)
        else:
            if self.active_cell_idx < 0 or self.active_cell_idx >= len(self.cells):
                return
            self.cells[self.active_cell_idx].push_waypoint(x, y)
        self.overlayChanged.emit()

    def clear(self) -> None:
        self.set_running(False)
        # Stop any active async trackers so their background threads exit.
        if self.robot_tracker is not None:
            try:
                self.robot_tracker.stop()
            except Exception:
                pass
        for tr in self.cell_trackers:
            try:
                tr.stop()
            except Exception:
                pass
        self._stop_distractors()
        self.robot = None
        self.robot_tracker = None
        self.robot_kf = None
        self.robot_contested = False
        self.robot_ambiguity = 0.0
        self._contested_since = None
        self.cells = []
        self.cell_trackers = []
        self.active_cell_idx = -1
        self._pending_selections.clear()
        self.overlayChanged.emit()
        self._emit_state()

    def remove_cell(self, idx: int) -> None:
        if 0 <= idx < len(self.cells):
            try:
                self.cell_trackers[idx].stop()
            except Exception:
                pass
            del self.cells[idx]
            del self.cell_trackers[idx]
            if self.active_cell_idx >= len(self.cells):
                self.active_cell_idx = len(self.cells) - 1
            self.overlayChanged.emit()
            self._emit_state()

    def clear_active_cell_path(self) -> None:
        if 0 <= self.active_cell_idx < len(self.cells):
            self.cells[self.active_cell_idx].clear_waypoints()
            self.overlayChanged.emit()

    # ---- runtime tunables ---------------------------------------------

    def set_running(self, on: bool) -> None:
        if on == self.running:
            return
        self.running = bool(on)
        if not self.running:
            # Yield the field control back to whatever was there.
            self.motion.set_mode(Mode.OFF)
        self.runningChanged.emit(self.running)
        self._emit_state()

    def set_robot_mask_params(self, params: MaskParams) -> None:
        self.robot_mask_params = params
        if self.robot_tracker is not None:
            try:
                self.robot_tracker.raw_tracker.set_mask_params(params)
            except AttributeError:
                pass
        # The global blob pass uses the same mask the operator is tuning,
        # so slider changes have to reach it too or the distractor set
        # silently diverges from what's on screen.
        if self._distractors is not None:
            self._distractors.set_mask_params(params)
        if self.robot is not None:
            self.robot.crop_length = params.crop_length

    def set_cell_mask_params(self, params: MaskParams) -> None:
        self.cell_mask_params = params
        for tr in self.cell_trackers:
            try:
                tr.raw_tracker.set_mask_params(params)
            except AttributeError:
                pass
        for c in self.cells:
            c.crop_length = params.crop_length

    def set_arrived_threshold(self, px: float) -> None:
        self.arrived_px = float(px)

    def set_approach_arrived_threshold(self, px: float) -> None:
        self.approach_arrived_px = float(px)

    def set_push_offset(self, px: float) -> None:
        self.push_offset_px = float(px)

    def set_drive_mode(self, mode: str) -> None:
        if mode in ("rolling", "static"):
            self.drive_mode = mode

    def set_show_robot_mask(self, on: bool) -> None:
        self.show_robot_mask = bool(on)
        if not self.show_robot_mask:
            self._latest_robot_mask = None
        self.overlayChanged.emit()

    def set_show_cell_mask(self, on: bool) -> None:
        self.show_cell_mask = bool(on)
        if not self.show_cell_mask:
            self._latest_cell_mask = None
        self.overlayChanged.emit()

    def set_cell_tracker_backend(self, backend: str) -> None:
        """Applies to the *next* cell selection. Existing cell trackers keep
        using whichever backend they were seeded with; operator can Clear
        + reselect if they want to migrate mid-session."""
        if backend not in ("template", "adaptive"):
            return
        self.cell_tracker_backend = backend
        self._emit_state()

    def set_robot_tracker_backend(self, backend: str) -> None:
        """Same idea for the robot."""
        if backend not in ("template", "adaptive"):
            return
        self.robot_tracker_backend = backend
        self._emit_state()

    def _build_adaptive_tracker(self, mask_params: MaskParams) -> AdaptiveTracker:
        return AdaptiveTracker(
            mask_params,
            backend=self.adaptive_backend_pref,
            preferred_target=self.adaptive_target_pref,
            models_dir=self.models_dir,
            url_prefix=self.nano_url_prefix,
            sam2_size=self.sam2_size,
            log_fn=self.log)

    def _build_template_tracker(self, mask_params: MaskParams) -> ObjectTracker:
        return ObjectTracker(
            mask_params,
            min_match_confidence=self.min_match_confidence,
            template_refresh_s=self.template_refresh_s,
            ambiguity_enter=self.ambiguity_enter,
            ambiguity_exit=self.ambiguity_exit,
            contested_max_s=self.contested_max_s,
            search_half_px=self.search_half_px,
            prior_sigma_floor_px=self.prior_sigma_floor_px)

    def _ensure_distractors(self) -> None:
        """Spin up the global blob pass on first selection."""
        if not self.distractors_enabled or self._distractors is not None:
            return
        try:
            field = DistractorField(
                self.robot_mask_params,
                gate_px=self.distractor_gate_px,
                max_blobs=self.distractor_max_blobs,
                accel_noise_px_s2=self.kf_accel_noise)
            self._distractors = DistractorWorker(
                field, rate_hz=self.distractor_rate_hz, log_fn=self.log)
        except Exception as e:
            self.log(f"PathFollowController: distractor field disabled ({e})")
            self._distractors = None

    def _stop_distractors(self) -> None:
        if self._distractors is not None:
            try:
                self._distractors.stop()
            except Exception:
                pass
        self._distractors = None
        self._distractor_snapshot = []

    def _build_cell_tracker(self):
        if self.cell_tracker_backend == "adaptive":
            return self._build_adaptive_tracker(self.cell_mask_params)
        return self._build_template_tracker(self.cell_mask_params)

    def _build_robot_tracker(self):
        if self.robot_tracker_backend == "adaptive":
            return self._build_adaptive_tracker(self.robot_mask_params)
        return self._build_template_tracker(self.robot_mask_params)

    # ---- per-frame tick ------------------------------------------------

    def on_frame(self, frame: np.ndarray, t_capture: float = 0.0) -> None:
        if frame is None:
            return
        # Capture time comes from the video thread the instant read()
        # returned. Using arrival time here instead would fold queue and
        # inference latency into every dt the filter sees.
        t = float(t_capture) if t_capture else time.monotonic()

        # The aravis backend re-latches the SAME buffer object each grab,
        # so the capture thread can overwrite these pixels while a worker
        # is mid-correlation — which shows up as phantom ambiguity, not as
        # an obvious failure. One copy per frame buys determinism.
        if self.copy_frames:
            frame = frame.copy()

        gray = None
        need_gray = (self.show_robot_mask or self.show_cell_mask
                     or self._distractors is not None)
        if need_gray:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Mask overlays (computed once per frame when enabled, so the
        # operator can see live what the slider changes are doing).
        if self.show_robot_mask or self.show_cell_mask:
            if self.show_robot_mask:
                self._latest_robot_mask = _apply_mask(
                    gray, self.robot_mask_params)
            else:
                self._latest_robot_mask = None
            if self.show_cell_mask:
                self._latest_cell_mask = _apply_mask(
                    gray, self.cell_mask_params)
            else:
                self._latest_cell_mask = None

        # Global blob pass — self-throttling to distractor_rate_hz on its
        # own thread; submit is a no-op between ticks.
        if self._distractors is not None and gray is not None:
            self._distractors.submit(gray, t)
            snap, _snap_t = self._distractors.snapshot()
            self._distractor_snapshot = snap

        # Robot detection — submit async. Results marshal back to the main
        # thread via _updateResult and land in _on_update_result.
        if self.robot_tracker is not None and self.robot is not None:
            self.robot_tracker.submit_update(
                frame, t, self._build_robot_prior(t))

        # Cell detection (skip DONE cells). Cells get the capture time but
        # no prior — they're pushed around by the robot rather than
        # self-propelled, so there's no command to predict from.
        for idx, (cell, tr) in enumerate(zip(self.cells, self.cell_trackers)):
            if cell.status == TargetStatus.DONE:
                continue
            tr.submit_update(frame, t, None)

        # Emit state (throttled) — reflects whatever the trackers last
        # produced (which may be slightly behind the current frame under a
        # slow neural backend, but the control loop uses the same latest
        # state, so this stays coherent).
        self._emit_state()
        self.overlayChanged.emit()

        # Control step still runs every frame using the latest state — the
        # inner motion loop needs steady updates to keep the firmware
        # watchdog fed, and the bead isn't moving fast enough for a few
        # frames of tracker lag to matter.
        if self.running:
            if self.operation_mode == self.OP_FOLLOW_ROBOT:
                self._step_follow_robot()
            else:
                self._step_push_cells()

    # ---- control steps -------------------------------------------------

    def _reload_frame_matrix(self) -> None:
        raw = self.config.get("calibration.screen_to_world_2x2",
                              [[1.0, 0.0], [0.0, 1.0]])
        try:
            self._screen_to_world_2x2 = np.asarray(
                raw, dtype=float).reshape(2, 2)
        except Exception:
            self._screen_to_world_2x2 = np.eye(2)

    def _reload_response_matrix(self) -> None:
        self._screen_response_2x2 = self._load_2x2(
            "calibration.screen_response_2x2")
        # Mode B has its own scale — no frequency term, and speed goes with
        # magnitude squared rather than being magnitude-independent.
        self._static_response_2x2 = self._load_2x2(
            "calibration.static_response_2x2")
        try:
            self._static_response_mag = float(
                self.config.get("calibration.static_response_mag", 1.0))
        except Exception:
            self._static_response_mag = 1.0

    def _load_2x2(self, path: str):
        raw = self.config.get(path, None)
        if raw is None:
            return None
        try:
            M = np.asarray(raw, dtype=float).reshape(2, 2)
            return M if np.all(np.isfinite(M)) else None
        except Exception:
            return None

    def _commanded_velocity_px_s(self) -> Optional[tuple]:
        """Expected bead velocity in raw-image px/s, from the live command.

        Read from :class:`MotionController` rather than from the follower's
        own locals, so this works identically whoever is driving — path
        follower, joystick, or a GUI panel. Returns ``None`` when the drive
        isn't characterized, which makes the filter fall back to plain
        constant velocity.

        The two modes scale differently and each has its own measured
        response matrix from the Frame Cal tab:

        * **Rolling (Mode A)** — displacement scales with freq × time, so
          speed is ``S_roll · ŵ · freq``. Magnitude is deliberately not a
          factor: above the rolling threshold it sets force, not speed.
        * **Static (Mode B)** — no frequency at all. For a paramagnetic
          bead ``F ∝ ∇|B|²`` and ``B ∝ duty``, so ``F ∝ mag²``; the regime
          is overdamped, so terminal velocity ∝ F. Hence
          ``S_static · ŵ · (mag / mag_cal)²``, with ``mag_cal`` the
          magnitude the calibration was measured at.

        Either matrix being absent (never calibrated for that mode) returns
        ``None``, and the filter falls back to constant velocity.
        """
        try:
            mode = self.motion.mode
            if mode == Mode.OFF:
                return (0.0, 0.0)
            if mode == Mode.A_ROTATING:
                S, scale = self._screen_response_2x2, None
            elif mode == Mode.B_STATIC:
                S, scale = self._static_response_2x2, None
            else:
                return None
            if S is None:
                return None
            mag = float(self.motion.magnitude)
            if mag <= 1e-6:
                return (0.0, 0.0)
            d = np.asarray(self.motion.direction, dtype=float).reshape(3)
            w = d[:2]
            n = float(np.linalg.norm(w))
            if n < 1e-9:
                return (0.0, 0.0)
            if mode == Mode.A_ROTATING:
                scale = float(self.motion.freq_hz)
            else:
                mag_cal = max(1e-6, float(self._static_response_mag))
                scale = (mag / mag_cal) ** 2.0
            v_screen = (S @ (w / n)) * scale
            # Screen-up frame → raw image (y grows down).
            return float(v_screen[0]), float(-v_screen[1])
        except Exception:
            return None

    def _build_robot_prior(self, t: float) -> Optional[MotionPrior]:
        """Predict the robot's position on the frame captured at ``t``.

        Pure projection — the filter is NOT advanced here. If this frame
        gets dropped by the worker's drop-old queue, nothing has been
        corrupted; the filter only moves when a detection actually lands.
        """
        if self.robot_kf is None or not self.motion_prior_enabled:
            return None
        dt = max(0.0, t - self._robot_kf_t)
        pred, sigma = self.robot_kf.project(
            dt, self._commanded_velocity_px_s())
        # The global pass has no idea which blob the operator selected, so
        # the robot appears in its own distractor list. Drop anything close
        # enough to be it — suppressing the robot's own peak would push the
        # tracker straight onto a neighbour, the exact failure being fixed.
        dist = ()
        if self._distractor_snapshot:
            self_radius = max(15.0, 0.75 * self.robot_mask_params.crop_length)
            # Exclude against BOTH the prediction and the last measured
            # position. If the filter has drifted, an exclusion keyed only
            # to the prediction stops covering the robot's real blob, and
            # the robot's own peak gets suppressed — which looks exactly
            # like the box refusing to follow and then jumping.
            refs = [pred]
            if self.robot is not None and self.robot.last_pos is not None:
                refs.append(self.robot.last_pos)
            dist = tuple(
                (x, y, s) for (x, y, s) in self._distractor_snapshot
                if all(math.hypot(x - rx, y - ry) > self_radius
                       for rx, ry in refs))
        return MotionPrior(pred_xy=pred, sigma_px=sigma, distractors=dist)

    # ---- contested-state audit trail -----------------------------------

    def _log_event(self, payload: dict) -> None:
        """Append one JSON line so a run can be audited for near-swaps.

        Best-effort: a failure to write must never disturb tracking.
        """
        self.log(f"tracker: {payload.get('event')} "
                 f"ambiguity={payload.get('ambiguity', 0.0):.3f} "
                 f"pos={payload.get('pos')} alt={payload.get('second_pos')}")
        if not self.event_log_enabled:
            return
        try:
            if self._event_log_path is None:
                stamp = time.strftime("%Y%m%d-%H%M%S")
                self._event_log_path = os.path.join(
                    resolve_output_dir(self.config),
                    f"tracking_events_{stamp}.jsonl")
            with open(self._event_log_path, "a") as fh:
                fh.write(json.dumps(payload) + "\n")
        except Exception:
            self.event_log_enabled = False

    def _image_to_world(self, dx: float, dy: float) -> tuple:
        """Raw-image delta → world horizontal delta via the calibrated matrix.

        Image y grows down; the operator/calibration frame is screen-UP,
        matching the joystick bridge's ``by = -ly``. So flip y, then apply
        ``calibration.screen_to_world_2x2`` — the SAME matrix the joystick
        uses, so one Frame Cal run fixes manual and autonomous driving
        together. Deliberately does NOT compose ``camera.view_rotation_deg``:
        the follower works in RAW image coordinates (clicks are un-rotated
        in VideoWidget), so the ⟳ display button must not affect it.
        """
        ux, uy = float(dx), -float(dy)
        M = self._screen_to_world_2x2
        return (float(M[0, 0] * ux + M[0, 1] * uy),
                float(M[1, 0] * ux + M[1, 1] * uy))

    def _push_direction(self, err_img: tuple) -> Optional[tuple]:
        mag = math.hypot(*err_img)
        if mag < 1e-6:
            return None
        wx, wy = self._image_to_world(err_img[0], err_img[1])
        wm = math.hypot(wx, wy)
        if wm < 1e-9:
            return None
        return (wx / wm, wy / wm)

    def _contested_hold(self, t: Optional[float] = None) -> bool:
        """Hold station while the robot's identity is in doubt.

        Driving on a possibly-swapped target is worse than briefly not
        driving at all — a wrong-bead command actively walks the real
        robot somewhere unintended. So the field goes to OFF but
        ``running`` stays set, and the run resumes by itself once the
        contest clears. A permanently-contested scene would otherwise hang
        here forever, hence the timeout.
        """
        if not self.robot_contested:
            return False
        now = t if t is not None else time.monotonic()
        self.motion.set_mode(Mode.OFF)
        if (self._contested_since is not None
                and now - self._contested_since > self.contested_timeout_s):
            self.log(
                f"PathFollowController: target contested for "
                f"{self.contested_timeout_s:.1f}s "
                f"(ambiguity {self.robot_ambiguity:.2f}) — stopping")
            self._log_event({
                "event": "contested_timeout", "t": now,
                "ambiguity": self.robot_ambiguity,
                "pos": list(self.robot.last_pos) if self.robot
                       and self.robot.last_pos else None,
                "second_pos": None,
            })
            self.set_running(False)
        return True

    def _step_follow_robot(self) -> None:
        if self.robot is None or self.robot.last_pos is None:
            self.set_running(False)
            return
        if self._contested_hold():
            return
        target = self.robot.current_target()
        if target is None:
            self.set_running(False)
            return
        r = self.robot.last_pos
        err_img = (target[0] - r[0], target[1] - r[1])
        if math.hypot(*err_img) < self.arrived_px:
            if not self.robot.advance_target():
                self.log("PathFollowController: robot reached final waypoint")
                self.set_running(False)
                return
            return
        d = self._push_direction(err_img)
        if d is None:
            return
        self._command_direction(d)

    def _command_direction(self, d: tuple) -> None:
        """Push a world-frame travel direction to the coils using the
        configured drive mode. ``d`` is unit-length, post _image_to_world.

        Rolling (default) is what actually locomotes paramagnetic beads —
        the field sweeps the vertical plane containing the travel vector.
        Magnitude and frequency are the shared knobs (Joystick tab), read
        at use so mid-run changes apply on the next frame. Frequency is
        constant per run; distance-scaled frequency is future work.
        """
        mag = float(np.clip(
            self.config.get("modes.mode_a.magnitude_default", 1.0), 0.0, 1.0))
        self.motion.set_static_direction([d[0], d[1], 0.0], Source.ALGORITHM)
        self.motion.set_magnitude(mag, Source.ALGORITHM)
        if self.drive_mode == "rolling":
            freq = float(self.config.get("modes.mode_a.freq_default", 1.0))
            self.motion.set_roll_axis(roll_axis_for(d[0], d[1]),
                                      Source.ALGORITHM)
            self.motion.set_frequency(freq, Source.ALGORITHM)
            self.motion.set_mode(Mode.A_ROTATING)
        else:
            self.motion.set_mode(Mode.B_STATIC)

    def _step_push_cells(self) -> None:
        if self.robot is None or self.robot.last_pos is None:
            self.set_running(False)
            return
        if self._contested_hold():
            return

        # Pick next actionable cell.
        active_idx = -1
        for i, cell in enumerate(self.cells):
            if cell.status in (TargetStatus.PENDING, TargetStatus.IN_PROGRESS):
                if cell.last_pos is None:
                    continue
                if cell.current_target() is None:
                    cell.status = TargetStatus.DONE
                    continue
                active_idx = i
                break
        if active_idx < 0:
            self.log("PathFollowController: all cells DONE / LOST")
            self.set_running(False)
            return

        active = self.cells[active_idx]
        if active.status == TargetStatus.PENDING:
            active.status = TargetStatus.IN_PROGRESS
        self.active_cell_idx = active_idx  # keep panel in sync

        goal = active.current_target()
        c = active.last_pos
        r = self.robot.last_pos
        push_vec = (goal[0] - c[0], goal[1] - c[1])
        push_mag = math.hypot(*push_vec)
        if push_mag < 1e-6:
            # Cell already on goal — advance.
            if not active.advance_target():
                active.status = TargetStatus.DONE
            return

        push_dir = (push_vec[0] / push_mag, push_vec[1] / push_mag)
        push_position = (c[0] - push_dir[0] * self.push_offset_px,
                         c[1] - push_dir[1] * self.push_offset_px)

        # APPROACH vs PUSH (recomputed each tick).
        dr = math.hypot(r[0] - push_position[0], r[1] - push_position[1])
        if dr > self.approach_arrived_px:
            robot_target = push_position
        else:
            robot_target = goal

        err_img = (robot_target[0] - r[0], robot_target[1] - r[1])
        d = self._push_direction(err_img)
        if d is not None:
            self._command_direction(d)

        # Advance cell target if cell has arrived at its goal.
        if math.hypot(c[0] - goal[0], c[1] - goal[1]) < self.arrived_px:
            if not active.advance_target():
                active.status = TargetStatus.DONE
                self.log(
                    f"PathFollowController: cell C{active_idx + 1} DONE")

        # Stall watchdog.
        dist_to_goal = math.hypot(c[0] - goal[0], c[1] - goal[1])
        if active.stall_last_distance is None:
            active.stall_last_distance = dist_to_goal
            active.stall_frames = 0
        else:
            if abs(dist_to_goal - active.stall_last_distance) < 0.5:
                active.stall_frames += 1
            else:
                active.stall_frames = 0
                active.stall_last_distance = dist_to_goal
            if active.stall_frames > self.stall_frames:
                self.log(
                    f"PathFollowController: cell C{active_idx + 1} STALL — marking LOST")
                active.status = TargetStatus.LOST

    # ---- state emission + overlay snapshot -----------------------------

    def _emit_state(self) -> None:
        now = time.monotonic()
        if now - self._last_state_emit < (1.0 / 15.0):
            return
        self._last_state_emit = now

        robot_metrics = None
        if self.robot is not None:
            robot_metrics = {
                "diameter_um": self.robot.diameter_px * self.um_per_pixel,
                "speed_ums": self.robot.last_speed_px_s * self.um_per_pixel,
                "accel_ums2": self.robot.last_accel_px_s2 * self.um_per_pixel,
                "blur": self.robot.last_blur,
                "pos": self.robot.last_pos,
                "target_idx": self.robot.target_idx,
                "n_waypoints": len(self.robot.trajectory),
                "lost_streak": self.robot.lost_streak,
                "ambiguity": self.robot_ambiguity,
                "contested": self.robot_contested,
                "n_distractors": len(self._distractor_snapshot),
            }
        cells_summary = []
        for i, cell in enumerate(self.cells):
            cells_summary.append({
                "index": i,
                "status": cell.status.value,
                "n_waypoints": len(cell.trajectory),
                "target_idx": cell.target_idx,
                "lost_streak": cell.lost_streak,
                "pos": cell.last_pos,
            })
        self.stateChanged.emit({
            "operation_mode": self.operation_mode,
            "selection_mode": self.selection_mode,
            "running": self.running,
            "robot": robot_metrics,
            "cells": cells_summary,
            "active_cell_idx": self.active_cell_idx,
            "robot_tracker_backend": self.robot_tracker_backend,
            "cell_tracker_backend": self.cell_tracker_backend,
            "adaptive_effective_backend": self.adaptive_effective_backend,
            "adaptive_effective_target": self.adaptive_effective_target,
        })

    def overlay_snapshot(self) -> dict:
        robot_state = None
        if self.robot is not None:
            robot_state = {
                "pos": self.robot.last_pos,
                "crop_length": self.robot.crop_length,
                "trajectory": list(self.robot.trajectory),
                "target_idx": self.robot.target_idx,
                # Amber box while the identity is in doubt — the operator
                # needs to see a near-swap as it happens, not find it in a
                # log afterwards.
                "contested": self.robot_contested,
            }
        cells_state = []
        for i, cell in enumerate(self.cells):
            cells_state.append({
                "index": i,
                "pos": cell.last_pos,
                "crop_length": cell.crop_length,
                "trajectory": list(cell.trajectory),
                "target_idx": cell.target_idx,
                "status": cell.status.value,
            })
        return {
            "robot": robot_state,
            "cells": cells_state,
            "active_cell_idx": self.active_cell_idx,
            "operation_mode": self.operation_mode,
            "running": self.running,
            "robot_mask": self._latest_robot_mask,
            "cell_mask": self._latest_cell_mask,
        }
