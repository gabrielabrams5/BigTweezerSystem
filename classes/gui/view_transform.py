"""view_transform — pure math for the camera view's 90° display rotation.

Convention: ``rot_deg`` ∈ {0, 90, 180, 270} is the CLOCKWISE rotation
applied to the displayed image (90 matches ``cv2.ROTATE_90_CLOCKWISE``).
Coordinates use the pixel-center convention (``w - 1 - x``) so they land
exactly on the pixels a cv2-rotated raster produces. The ``(w, h)``
arguments are always the RAW frame size, regardless of rotation.

No Qt or cv2 imports — fully unit-testable.
"""

from __future__ import annotations


def display_size(w: int, h: int, rot_deg: int) -> tuple:
    """Displayed (width, height) after rotating a w×h frame."""
    return (w, h) if int(rot_deg) % 180 == 0 else (h, w)


def frame_to_display(x: float, y: float, w: int, h: int, rot_deg: int) -> tuple:
    """Raw-frame pixel → displayed-image pixel."""
    r = int(rot_deg) % 360
    if r == 90:
        return (h - 1 - y), x
    if r == 180:
        return (w - 1 - x), (h - 1 - y)
    if r == 270:
        return y, (w - 1 - x)
    return x, y


def display_to_frame(x: float, y: float, w: int, h: int, rot_deg: int) -> tuple:
    """Displayed-image pixel → raw-frame pixel (inverse of frame_to_display).

    ``(w, h)`` is still the RAW frame size.
    """
    r = int(rot_deg) % 360
    if r == 90:
        return y, (h - 1 - x)
    if r == 180:
        return (w - 1 - x), (h - 1 - y)
    if r == 270:
        return (w - 1 - y), x
    return x, y


def stick_display_to_raw(rot_deg: int) -> list:
    """2×2 mapping an operator stick vector in the DISPLAYED view's y-UP
    frame to the raw-image y-UP frame (the frame the fitted
    ``calibration.screen_to_world_2x2`` expects).

    Derivation: in image coords (y down) the display rotation acts on
    vectors as D_r; the stick lives in y-up frames on both sides, so the
    required matrix is the conjugated inverse R = F · D_r⁻¹ · F with
    F = diag(1, −1). Sanity check at 90° (view rotated CW): stick-up
    (0, 1) → (−1, 0) = raw-left in the up-frame, and raw-left is exactly
    what appears as display-up under a 90° CW view rotation.
    """
    return {
        0:   [[1.0, 0.0], [0.0, 1.0]],
        90:  [[0.0, -1.0], [1.0, 0.0]],
        180: [[-1.0, 0.0], [0.0, -1.0]],
        270: [[0.0, 1.0], [-1.0, 0.0]],
    }[int(rot_deg) % 360]
