"""State containers for the tracker + path-follower.

``RobotState`` holds the paramagnetic bead's rolling per-frame state
(position, velocity, acceleration, blur, area) plus its own trajectory
of user-drawn waypoints. ``CellState`` mirrors it for non-magnetic cells
and adds a ``TargetStatus`` so the push-cells sequencer can track which
cells are pending, in progress, done, or lost.

Velocity and acceleration are computed from a rolling memory window —
finite-difference over ``memory`` frames divided by elapsed time. Uses
image-pixel units; the panel converts to µm/s via ``um_per_pixel``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class TargetStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"
    LOST = "LOST"


@dataclass
class _TrackedObject:
    """Common bits between RobotState and CellState. Not instantiated
    directly — the two concrete classes bring the fields together with
    per-role extras."""

    positions: List[Tuple[float, float]] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    velocities: List[Tuple[float, float, float]] = field(default_factory=list)  # (vx, vy, |v|)
    accelerations: List[Tuple[float, float, float]] = field(default_factory=list)
    blur_history: List[float] = field(default_factory=list)
    area_history: List[float] = field(default_factory=list)
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    target_idx: int = 0
    crop_length: int = 40
    um_per_pixel: float = 1.0
    memory: int = 15
    lost_streak: int = 0

    # -- geometry ----------------------------------------------------

    @property
    def last_pos(self) -> Optional[Tuple[float, float]]:
        return self.positions[-1] if self.positions else None

    @property
    def diameter_px(self) -> float:
        """Approximate diameter from the latest contour area
        (D = sqrt(4A/π))."""
        if not self.area_history:
            return 0.0
        return math.sqrt(max(0.0, 4.0 * self.area_history[-1] / math.pi))

    @property
    def last_speed_px_s(self) -> float:
        return self.velocities[-1][2] if self.velocities else 0.0

    @property
    def last_accel_px_s2(self) -> float:
        return self.accelerations[-1][2] if self.accelerations else 0.0

    @property
    def last_blur(self) -> float:
        return self.blur_history[-1] if self.blur_history else 0.0

    # -- trajectory --------------------------------------------------

    def push_waypoint(self, x: float, y: float) -> None:
        self.trajectory.append((float(x), float(y)))

    def clear_waypoints(self) -> None:
        self.trajectory = []
        self.target_idx = 0

    def current_target(self) -> Optional[Tuple[float, float]]:
        if 0 <= self.target_idx < len(self.trajectory):
            return self.trajectory[self.target_idx]
        return None

    def advance_target(self) -> bool:
        """Return True if there's another waypoint after advancing."""
        self.target_idx += 1
        return self.target_idx < len(self.trajectory)

    # -- per-frame update -------------------------------------------

    def record_frame(self,
                     t: float,
                     pos: Tuple[float, float],
                     area_px: float,
                     blur: float) -> None:
        """Append a new sample and recompute velocity + acceleration.

        Velocity: (pos[t] - pos[t - memory]) / (times[t] - times[t - memory])
        Acceleration: (v[t] - v[t - memory]) / same denominator
        """
        self.positions.append((float(pos[0]), float(pos[1])))
        self.times.append(float(t))
        self.area_history.append(float(area_px))
        self.blur_history.append(float(blur))
        self.lost_streak = 0

        # Velocity from memory window
        vx = vy = 0.0
        if len(self.positions) >= self.memory + 1:
            p0 = self.positions[-1 - self.memory]
            t0 = self.times[-1 - self.memory]
            dt = self.times[-1] - t0
            if dt > 1e-9:
                vx = (self.positions[-1][0] - p0[0]) / dt
                vy = (self.positions[-1][1] - p0[1]) / dt
        vmag = math.hypot(vx, vy)
        self.velocities.append((vx, vy, vmag))

        # Acceleration from same window over velocities
        ax = ay = 0.0
        if len(self.velocities) >= self.memory + 1:
            v0 = self.velocities[-1 - self.memory]
            t0 = self.times[-1 - self.memory]
            dt = self.times[-1] - t0
            if dt > 1e-9:
                ax = (self.velocities[-1][0] - v0[0]) / dt
                ay = (self.velocities[-1][1] - v0[1]) / dt
        amag = math.hypot(ax, ay)
        self.accelerations.append((ax, ay, amag))

        # Cap history so long sessions don't grow unbounded.
        cap = max(self.memory * 4, 200)
        for lst in (self.positions, self.times, self.velocities,
                    self.accelerations, self.blur_history, self.area_history):
            if len(lst) > cap:
                del lst[0:len(lst) - cap]


@dataclass
class RobotState(_TrackedObject):
    """The paramagnetic robot. One per session."""
    pass


@dataclass
class CellState(_TrackedObject):
    """A non-magnetic cell that gets pushed along its own path."""

    status: TargetStatus = TargetStatus.PENDING
    stall_frames: int = 0    # consecutive frames without progress on |pos - goal|
    stall_last_distance: Optional[float] = None
    # Stable identity, assigned once at selection and never reused. List
    # position is NOT identity: removing a cell shifts every later index,
    # so an async detection tagged with a position would land on whichever
    # cell slid into that slot. Callbacks resolve by uid instead.
    uid: int = 0
