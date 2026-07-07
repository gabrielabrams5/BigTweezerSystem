"""
Runtime patch for EasyPySpin.VideoCapture.

Symptom without this patch:
    SpinnakerException: Failed to write enumeration value.
        Enum entry is not writable :
        AccessException thrown in node 'PixelFormat' while calling
        'PixelFormat.SetIntValue()'

Cause:
    EasyPySpin's grab() calls
        self.cam.PixelFormat.SetValue(PySpin.PixelFormat_BGR8)
    on every frame. On most FLIR machines the PixelFormat node becomes
    read-only after BeginAcquisition(), so this write raises an
    AccessException the first time grab() runs and the tracker thread dies.

Fix:
    Replace the offending line with a no-op via inspect.getsource +
    string substitution + exec. This is safer than shipping a hand-rolled
    grab() replacement because we reuse EasyPySpin's own logic; we only
    neuter the one line that fails. If EasyPySpin ever fixes this upstream
    the substitution is a no-op (nothing to replace) and this module
    becomes harmless.

Optionally we also inject a one-time
    self.cam.PixelFormat.SetValue(PySpin.PixelFormat_BGR8)
    right before BeginAcquisition() inside open(), so the pixel format is
    still guaranteed BGR8 for downstream code that expects color frames.
"""

import inspect
import textwrap


def _patch_method(cls, method_name, needle, replacement, extra_globals):
    """Rewrite a method's source by str.replace and rebind on the class.

    Returns True on success, False if the source could not be fetched or
    the needle was not present (i.e., nothing to patch).
    """
    try:
        src = inspect.getsource(getattr(cls, method_name))
    except (OSError, TypeError):
        return False
    src = textwrap.dedent(src)
    if needle not in src:
        return False
    new_src = src.replace(needle, replacement)
    ns = dict(extra_globals)
    try:
        exec(new_src, ns)
    except Exception:
        return False
    if method_name not in ns:
        return False
    setattr(cls, method_name, ns[method_name])
    return True


def apply_easypyspin_patch(log=print):
    """Idempotent. Safe to call multiple times. No-ops if EasyPySpin/PySpin
    are not installed (dev machines without the FLIR SDK).
    """
    try:
        import EasyPySpin
        import PySpin
    except Exception:
        return False

    VideoCapture = EasyPySpin.VideoCapture
    extra_globals = {"PySpin": PySpin}

    # Neuter the per-frame PixelFormat.SetValue in grab().
    grab_needle = "self.cam.PixelFormat.SetValue(PySpin.PixelFormat_BGR8)"
    grab_patched = _patch_method(
        VideoCapture,
        "grab",
        grab_needle,
        "pass  # patched: PixelFormat set once in open(), not per-frame",
        extra_globals,
    )

    # Ensure PixelFormat is set BGR8 exactly once, right before BeginAcquisition().
    open_needle = "self.cam.BeginAcquisition()"
    open_patched = _patch_method(
        VideoCapture,
        "open",
        open_needle,
        (
            "try:\n"
            "            self.cam.PixelFormat.SetValue(PySpin.PixelFormat_BGR8)\n"
            "        except PySpin.SpinnakerException:\n"
            "            pass\n"
            "        self.cam.BeginAcquisition()"
        ),
        extra_globals,
    )

    if grab_patched or open_patched:
        log(
            "EasyPySpin patched at runtime "
            "(grab={}, open={})".format(grab_patched, open_patched)
        )
    return grab_patched or open_patched
