"""Tests for PathFollowController's contested-state handling, time-based
loss detection, and the commanded-velocity control input.

Run with:  python3 -m pytest classes/gui/test_path_follow_contested.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from classes.field_solver import FieldSolver, PARAMAGNETIC
from classes.gui.path_follow import PathFollowController
from classes.gui.robot_state import CellState, RobotState, TargetStatus
from classes.gui.test_joystick_bridge import FakeConfig
from classes.gui.tracker import Detection
from classes.motion_controller import Mode, MotionController
from classes.test_motion_controller import GEOM_BMAP


def make_controller(config_values=None):
    motion = MotionController(FieldSolver(GEOM_BMAP,
                                          robot_type=PARAMAGNETIC))
    cfg = FakeConfig(config_values or {})
    ctrl = PathFollowController(motion, cfg, log_fn=lambda *_: None)
    ctrl.event_log_enabled = False          # no files from a unit test
    return ctrl, motion, cfg


def make_det(pos, ambiguity=0.0, contested=False, t=0.0, area=300.0):
    blank = np.zeros((4, 4, 3), dtype=np.uint8)
    return Detection(pos=pos, area_px=area, blur=10.0, cropped_bgr=blank,
                     cropped_mask=blank[:, :, 0], confidence=0.9,
                     source="template", ambiguity=ambiguity,
                     second_pos=(pos[0] + 30.0, pos[1]) if contested else None,
                     contested=contested, t_capture=t)


def seed_robot(ctrl, pos=(100.0, 100.0), t=0.0):
    ctrl.robot = RobotState(crop_length=40, um_per_pixel=1.0, memory=3)
    ctrl.robot.record_frame(t, pos, 300.0, 10.0)
    ctrl._robot_gen = 1
    ctrl.robot_kf = None
    ctrl._robot_kf_t = t
    return ctrl.robot


# ---- contested handling -------------------------------------------------

def test_clean_detection_is_fused():
    ctrl, _motion, _ = make_controller()
    seed_robot(ctrl)
    ctrl._on_update_result("robot:1", make_det((110.0, 100.0), t=0.05), 0.05)
    assert ctrl.robot_kf is not None
    assert not ctrl.robot_contested
    # First detection seeds the filter, so the recorded position is the
    # measurement itself.
    assert ctrl.robot.last_pos == pytest.approx((110.0, 100.0))


def test_contested_detection_coasts_instead_of_trusting_the_peak():
    ctrl, _motion, _ = make_controller()
    seed_robot(ctrl)
    # Build up a rightward velocity from clean frames.
    for k in range(1, 6):
        t = 0.05 * k
        ctrl._on_update_result(
            "robot:1", make_det((100.0 + 10.0 * k, 100.0), t=t), t)
    assert ctrl.robot_kf.vel[0] > 50.0

    # Now a contested frame whose peak jumps BACKWARD onto a neighbour.
    t = 0.05 * 6
    ctrl._on_update_result(
        "robot:1", make_det((60.0, 100.0), ambiguity=0.95,
                            contested=True, t=t), t)
    assert ctrl.robot_contested
    # The reported position must follow the prediction (still moving
    # right), NOT the contested peak at x=60.
    assert ctrl.robot.last_pos[0] > 145.0


def test_contested_hold_stops_driving_but_stays_armed():
    ctrl, motion, _ = make_controller()
    seed_robot(ctrl)
    ctrl.robot.push_waypoint(400, 100)
    ctrl.set_running(True)
    ctrl.robot_contested = True
    ctrl._contested_since = 1000.0
    ctrl._step_follow_robot()
    assert motion.mode == Mode.OFF
    # Held, not aborted — the run resumes on its own when the contest clears.
    assert ctrl.running


def test_contested_times_out_and_stops():
    ctrl, motion, _ = make_controller(
        {"path_follow.contested_timeout_s": 0.5})
    seed_robot(ctrl)
    ctrl.robot.push_waypoint(400, 100)
    ctrl.set_running(True)
    ctrl.robot_contested = True
    import time as _t
    ctrl._contested_since = _t.monotonic() - 5.0
    ctrl._step_follow_robot()
    assert not ctrl.running


def test_contested_clears_and_run_resumes():
    ctrl, motion, _ = make_controller()
    seed_robot(ctrl)
    ctrl.robot.push_waypoint(400, 100)
    ctrl.set_running(True)
    ctrl.robot_contested = True
    ctrl._contested_since = None
    ctrl._step_follow_robot()
    assert motion.mode == Mode.OFF
    ctrl.robot_contested = False
    ctrl._step_follow_robot()
    assert motion.mode == Mode.A_ROTATING


# ---- time-based loss ----------------------------------------------------

def test_loss_is_timed_not_frame_counted():
    """Many rapid misses inside the timeout must NOT stop the run; a single
    miss after the timeout must. Frame counting got this backwards under
    load, letting the robot run open-loop for seconds."""
    ctrl, _motion, _ = make_controller({"tracker.lost_timeout_s": 1.0})
    seed_robot(ctrl, t=0.0)
    ctrl.robot.push_waypoint(400, 100)
    ctrl.set_running(True)
    for k in range(1, 60):                       # 59 misses in 0.3 s
        ctrl._on_update_result("robot:1", None, 0.005 * k)
    assert ctrl.running
    ctrl._on_update_result("robot:1", None, 2.0)
    assert not ctrl.running


def test_detection_resets_the_loss_clock():
    ctrl, _motion, _ = make_controller({"tracker.lost_timeout_s": 1.0})
    seed_robot(ctrl, t=0.0)
    ctrl.robot.push_waypoint(400, 100)
    ctrl.set_running(True)
    ctrl._on_update_result("robot:1", None, 0.9)
    ctrl._on_update_result("robot:1", make_det((105.0, 100.0), t=1.0), 1.0)
    ctrl._on_update_result("robot:1", None, 1.8)
    assert ctrl.running


# ---- commanded velocity -------------------------------------------------

def test_commanded_velocity_none_without_calibration():
    """Each mode needs its OWN response matrix; without it the filter falls
    back to constant velocity."""
    ctrl, motion, _ = make_controller()
    assert ctrl._screen_response_2x2 is None
    assert ctrl._static_response_2x2 is None
    motion.set_mode(Mode.A_ROTATING)
    assert ctrl._commanded_velocity_px_s() is None
    motion.set_mode(Mode.B_STATIC)
    assert ctrl._commanded_velocity_px_s() is None


def test_commanded_velocity_zero_when_off_even_uncalibrated():
    """Field off means the bead stops — that needs no calibration to know.
    The regime is overdamped, so there is no coast to model."""
    ctrl, motion, _ = make_controller()
    motion.set_mode(Mode.OFF)
    assert ctrl._commanded_velocity_px_s() == (0.0, 0.0)


def test_rolling_calibration_does_not_serve_static():
    """The two scales are different quantities (px per rev vs px/s); using
    one for the other would feed the filter a wrong velocity."""
    ctrl, motion, _ = make_controller({
        "calibration.screen_response_2x2": [[10.0, 0.0], [0.0, 10.0]]})
    motion.set_static_direction([1.0, 0.0, 0.0])
    motion.set_magnitude(1.0)
    motion.set_mode(Mode.B_STATIC)
    assert ctrl._commanded_velocity_px_s() is None


def test_static_velocity_uses_static_matrix():
    ctrl, motion, _ = make_controller({
        "calibration.static_response_2x2": [[8.0, 0.0], [0.0, 8.0]],
        "calibration.static_response_mag": 1.0})
    motion.set_static_direction([1.0, 0.0, 0.0])
    motion.set_magnitude(1.0)
    motion.set_mode(Mode.B_STATIC)
    vx, vy = ctrl._commanded_velocity_px_s()
    assert vx == pytest.approx(8.0)
    assert vy == pytest.approx(0.0)


def test_static_velocity_scales_with_magnitude_squared():
    """F ∝ ∇|B|² and B ∝ duty, so F ∝ mag²; overdamped ⇒ v ∝ F. Unlike
    rolling, where speed tracks frequency and ignores magnitude."""
    ctrl, motion, _ = make_controller({
        "calibration.static_response_2x2": [[8.0, 0.0], [0.0, 8.0]],
        "calibration.static_response_mag": 1.0})
    motion.set_static_direction([1.0, 0.0, 0.0])
    motion.set_mode(Mode.B_STATIC)
    motion.set_magnitude(0.5)
    assert ctrl._commanded_velocity_px_s()[0] == pytest.approx(8.0 * 0.25)
    motion.set_magnitude(1.0)
    assert ctrl._commanded_velocity_px_s()[0] == pytest.approx(8.0)


def test_static_velocity_flips_y_into_image_frame():
    ctrl, motion, _ = make_controller({
        "calibration.static_response_2x2": [[0.0, 0.0], [0.0, 8.0]],
        "calibration.static_response_mag": 1.0})
    motion.set_static_direction([0.0, 1.0, 0.0])
    motion.set_magnitude(1.0)
    motion.set_mode(Mode.B_STATIC)
    assert ctrl._commanded_velocity_px_s()[1] == pytest.approx(-8.0)


def test_commanded_velocity_scales_with_frequency():
    ctrl, motion, _ = make_controller({
        "calibration.screen_response_2x2": [[10.0, 0.0], [0.0, 10.0]]})
    motion.set_static_direction([1.0, 0.0, 0.0])
    motion.set_magnitude(1.0)
    motion.set_mode(Mode.A_ROTATING)
    motion.freq_hz = 2.0
    vx, vy = ctrl._commanded_velocity_px_s()
    # 10 px per (unit world · Hz · s) × 2 Hz = 20 px/s along +x.
    assert vx == pytest.approx(20.0)
    assert vy == pytest.approx(0.0)
    motion.freq_hz = 4.0
    assert ctrl._commanded_velocity_px_s()[0] == pytest.approx(40.0)


def test_commanded_velocity_flips_y_into_image_frame():
    """The response matrix is in the screen-up frame; image y grows down."""
    ctrl, motion, _ = make_controller({
        "calibration.screen_response_2x2": [[0.0, 0.0], [0.0, 10.0]]})
    motion.set_static_direction([0.0, 1.0, 0.0])
    motion.set_magnitude(1.0)
    motion.set_mode(Mode.A_ROTATING)
    motion.freq_hz = 1.0
    _vx, vy = ctrl._commanded_velocity_px_s()
    assert vy == pytest.approx(-10.0)


def test_commanded_velocity_zero_when_off():
    ctrl, motion, _ = make_controller({
        "calibration.screen_response_2x2": [[10.0, 0.0], [0.0, 10.0]]})
    motion.set_mode(Mode.OFF)
    assert ctrl._commanded_velocity_px_s() == (0.0, 0.0)


# ---- cell identity across removals --------------------------------------

def add_cell(ctrl, uid, pos):
    """Mimic _on_init_result's cell branch for a given generation."""
    cell = CellState(crop_length=40, um_per_pixel=1.0, memory=3, uid=uid)
    cell.record_frame(0.0, pos, 300.0, 10.0)
    ctrl.cells.append(cell)
    ctrl.cell_trackers.append(None)
    return cell


def test_cell_detection_follows_uid_not_list_position():
    """A detection queued for cell uid=2 must still reach uid=2 after an
    earlier cell is removed and every later index shifts down."""
    ctrl, _motion, _ = make_controller()
    c1 = add_cell(ctrl, 1, (10.0, 10.0))
    c2 = add_cell(ctrl, 2, (200.0, 200.0))
    c3 = add_cell(ctrl, 3, (400.0, 400.0))
    ctrl.active_cell_idx = 0

    # In-flight callback tagged with c2's click-time index (1) and uid 2.
    in_flight = "cell:1:2"
    ctrl.remove_cell(0)                       # c2 slides from index 1 → 0
    assert [c.uid for c in ctrl.cells] == [2, 3]

    ctrl._on_update_result(in_flight, make_det((205.0, 205.0), t=0.05), 0.05)
    assert c2.last_pos == pytest.approx((205.0, 205.0))
    assert c3.last_pos == pytest.approx((400.0, 400.0))   # untouched


def test_detection_for_removed_cell_is_dropped():
    ctrl, _motion, _ = make_controller()
    add_cell(ctrl, 1, (10.0, 10.0))
    c2 = add_cell(ctrl, 2, (200.0, 200.0))
    ctrl.active_cell_idx = 0
    ctrl.remove_cell(1)                       # c2 is gone
    before = [c.last_pos for c in ctrl.cells]
    ctrl._on_update_result("cell:1:2", make_det((205.0, 205.0), t=0.05), 0.05)
    assert [c.last_pos for c in ctrl.cells] == before


def test_miss_marks_the_right_cell_lost():
    """The mis-routing also corrupted loss: a miss for one cell could mark
    a different, perfectly healthy cell LOST."""
    ctrl, _motion, _ = make_controller({"tracker.lost_timeout_s": 1.0})
    c1 = add_cell(ctrl, 1, (10.0, 10.0))
    c2 = add_cell(ctrl, 2, (200.0, 200.0))
    c3 = add_cell(ctrl, 3, (400.0, 400.0))
    ctrl.active_cell_idx = 0
    ctrl.remove_cell(0)
    ctrl._on_update_result("cell:1:2", None, 5.0)
    assert c2.status == TargetStatus.LOST
    assert c3.status == TargetStatus.PENDING


def test_prior_excludes_the_robots_own_blob():
    ctrl, _motion, _ = make_controller()
    seed_robot(ctrl)
    ctrl._on_update_result("robot:1", make_det((100.0, 100.0), t=0.05), 0.05)
    ctrl._distractor_snapshot = [
        (102.0, 101.0, 4.0),        # the robot itself, seen globally
        (300.0, 100.0, 4.0),        # a genuine neighbour
    ]
    prior = ctrl._build_robot_prior(0.1)
    assert len(prior.distractors) == 1
    assert prior.distractors[0][0] == pytest.approx(300.0)
