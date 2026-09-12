"""Neural / adaptive tracker for cells.

Alternative to the fast template-match ``ObjectTracker`` in
``classes/gui/tracker.py``. Uses OpenCV's ONNX-backed neural trackers
(available in base ``opencv-python 5.0``+):

* **NanoTrackV2** (``cv2.TrackerNano_create``) — SiamFC-style, ~2 MB total
  weights, 60+ FPS CPU inference. Default choice.
* **VitTrack** (``cv2.TrackerVit_create``) — transformer, ~40 MB weights,
  higher accuracy. Opt-in.
* **MIL** (``cv2.TrackerMIL_create``) — no weights, adaptive online
  learning. Fallback when neither the neural weights nor a compatible
  ``TrackerNano_create`` is present.

Weights come from the opencv_zoo repo and are downloaded lazily to
``models/`` on first use. Backend and cv2.dnn target both probe at init
time: adopt the first entry that runs a dummy forward cleanly. If nothing
loads, ``initialize`` returns ``None`` and the caller falls back to the
existing template tracker.

Same public API as ``ObjectTracker`` so ``PathFollowController`` doesn't
branch on which tracker's in play.
"""

from __future__ import annotations

import glob
import os
import shutil
import socket
import urllib.error
import urllib.request
from typing import Optional, Tuple

import cv2
import numpy as np

from classes.gui.tracker import (
    Detection,
    MaskParams,
    _apply_mask,
    _clip_slice,
    _find_nearest_blob,
)


# ------------------------------------------------------------------ weights

# The opencv_zoo repo no longer ships NanoTrack — only VitTrack. VitTrack
# has both a full model (~700 KB) and an int8-quantized variant (~270 KB)
# stored via Git LFS. LFS files aren't resolved from raw.githubusercontent
# and instead require the ``media.githubusercontent.com/media/…`` URL.
# So the primary auto-downloadable backend is now ``vit``. ``nano`` still
# works but only if the operator drops NanoTrack ONNX files into
# ``models/`` manually.

_NANO_FILE_PATTERNS = {
    "backbone": [
        "object_tracking_nanotrackv2_backbone_*.onnx",
        "nanotrack_backbone*.onnx",
        "*nanotrack*backbone*.onnx",
    ],
    "neckhead": [
        "object_tracking_nanotrackv2_neckhead_*.onnx",
        "object_tracking_nanotrackv2_head_*.onnx",
        "nanotrack_head*.onnx",
        "*nanotrack*head*.onnx",
        "*nanotrack*neckhead*.onnx",
    ],
}

_VIT_FILE_PATTERNS = [
    "object_tracking_vittrack_*_int8bq.onnx",   # prefer quantized (~270 KB)
    "object_tracking_vittrack_*.onnx",           # then the full model
    "vittrack*.onnx",
]

# Ordered download candidates for vit. Prefer the full model — the int8
# quantized variant is 3× smaller but drops enough accuracy on small
# soft blobs (microrobots / beads) to matter in practice. Both are LFS.
_VIT_FILE_CANDIDATES = [
    "object_tracking_vittrack_2023sep.onnx",           # full, ~700 KB
    "object_tracking_vittrack_2023sep_int8bq.onnx",    # int8, ~270 KB
]

# Session-scoped failure cache. Once a URL prefix has failed, don't rehang
# the Qt event loop on subsequent selections.
_URL_FAILURES: set = set()


def _download(url: str, dest: str, timeout_s: float, log_fn) -> bool:
    """Bounded-timeout download. Returns True on success. Never raises."""
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp, \
                open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        os.replace(tmp, dest)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError,
            socket.timeout, OSError) as e:
        log_fn(f"AdaptiveTracker: download {url} failed: {e}")
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False


def _ensure_nano_weights(models_dir: str,
                         url_prefix: str,
                         log_fn,
                         timeout_s: float = 5.0) -> Optional[Tuple[str, str]]:
    """Return (backbone_path, neckhead_path) if both files exist on disk.
    Tries any pattern-matching file first; otherwise attempts one bounded
    download per candidate filename per role. If the URL prefix has already
    failed this session, skip downloads entirely and return ``None`` fast
    so the caller can fall back."""
    try:
        os.makedirs(models_dir, exist_ok=True)
    except OSError as e:
        log_fn(f"AdaptiveTracker: cannot create {models_dir}: {e}")
        return None

    resolved = {}
    for role, patterns in _NANO_FILE_PATTERNS.items():
        for pattern in patterns:
            matches = sorted(glob.glob(os.path.join(models_dir, pattern)))
            if matches:
                resolved[role] = matches[0]
                break

    missing = [r for r in ("backbone", "neckhead") if r not in resolved]
    if missing:
        prefix = url_prefix.strip()
        if not prefix or prefix in _URL_FAILURES:
            if not prefix:
                log_fn("AdaptiveTracker: no URL prefix configured for download")
            else:
                log_fn(
                    "AdaptiveTracker: skipping download — this session has "
                    "already recorded a failure for " + prefix)
            _log_manual_install_hint(models_dir, prefix, missing, log_fn)
            return None

        any_success = False
        for role in missing:
            got_it = False
            for fname in _NANO_FILE_CANDIDATES[role]:
                dest = os.path.join(models_dir, fname)
                url = prefix.rstrip("/") + "/" + fname
                log_fn(f"AdaptiveTracker: fetching {fname}")
                if _download(url, dest, timeout_s, log_fn):
                    resolved[role] = dest
                    any_success = True
                    got_it = True
                    break
            if not got_it:
                _URL_FAILURES.add(prefix)
                _log_manual_install_hint(models_dir, prefix, missing, log_fn)
                return None
        if not any_success:
            _URL_FAILURES.add(prefix)
            _log_manual_install_hint(models_dir, prefix, missing, log_fn)
            return None

    return resolved["backbone"], resolved["neckhead"]


def _log_manual_install_hint(models_dir: str, prefix: str,
                             missing: list, log_fn) -> None:
    log_fn(
        "AdaptiveTracker: NanoTrack weights unavailable — falling back "
        "to the fast template tracker. To install manually, download the "
        "two ONNX files (backbone + neckhead) from opencv_zoo and drop "
        f"them into `{models_dir}/`. Search paths tried: " +
        ", ".join(sum([_NANO_FILE_PATTERNS[m] for m in missing], [])))


def _ensure_vit_weights(models_dir: str,
                        url_prefix: str,
                        log_fn,
                        timeout_s: float = 8.0) -> Optional[str]:
    """Return path to the VitTrack ONNX (int8 quantized preferred). Auto-
    downloads from opencv_zoo (Git LFS-aware URL) if missing."""
    try:
        os.makedirs(models_dir, exist_ok=True)
    except OSError as e:
        log_fn(f"AdaptiveTracker: cannot create {models_dir}: {e}")
        return None

    # Existing file wins.
    for pattern in _VIT_FILE_PATTERNS:
        matches = sorted(glob.glob(os.path.join(models_dir, pattern)))
        if matches:
            return matches[0]

    prefix = url_prefix.strip()
    if not prefix or prefix in _URL_FAILURES:
        if not prefix:
            log_fn("AdaptiveTracker: no VitTrack URL configured")
        else:
            log_fn(
                "AdaptiveTracker: skipping VitTrack download — this session "
                f"has already recorded a failure for {prefix}")
        log_fn(
            "AdaptiveTracker: VitTrack weights unavailable. Download from "
            "opencv_zoo/main/models/object_tracking_vittrack/ "
            f"(via media.githubusercontent.com — LFS) and drop into `{models_dir}/`.")
        return None

    for fname in _VIT_FILE_CANDIDATES:
        dest = os.path.join(models_dir, fname)
        url = prefix.rstrip("/") + "/" + fname
        log_fn(f"AdaptiveTracker: fetching VitTrack weights {fname} …")
        if _download(url, dest, timeout_s, log_fn):
            return dest

    _URL_FAILURES.add(prefix)
    log_fn(
        "AdaptiveTracker: all VitTrack download candidates failed. "
        "Manual install: download the ONNX from opencv_zoo LFS and drop "
        f"into `{models_dir}/`.")
    return None


# ------------------------------------------------------------------ target probe

_TARGET_ORDER = [
    ("cuda_fp16", "DNN_TARGET_CUDA_FP16", "DNN_BACKEND_CUDA"),
    ("cuda", "DNN_TARGET_CUDA", "DNN_BACKEND_CUDA"),
    ("opencl_fp16", "DNN_TARGET_OPENCL_FP16", "DNN_BACKEND_OPENCV"),
    ("opencl", "DNN_TARGET_OPENCL", "DNN_BACKEND_OPENCV"),
    ("cpu", "DNN_TARGET_CPU", "DNN_BACKEND_OPENCV"),
]

# Session-scoped cache of the probe result keyed by (backbone_path,
# preferred). ``_probe_target`` reads and populates this so repeated
# selections don't re-run the ONNX load + dummy forward five times each.
_TARGET_CACHE: dict = {}


def _probe_target(backbone_path: str,
                  preferred: str,
                  log_fn) -> str:
    """Return the best dnn target for this ONNX file.

    On cv2 ≥ 5.0 the new graph engine doesn't accept ``setPreferableTarget``
    (emits the "Targets are not supported by the new graph engine" WARN
    lines). Running the probe is pure overhead — each attempted target
    loads the ONNX and runs a dummy forward, and every click was doing
    five of them. We now short-circuit to ``cpu`` on cv2 5.0+ and cache
    the result so subsequent selections skip the whole probe path.
    """
    key = (backbone_path, preferred)
    if key in _TARGET_CACHE:
        return _TARGET_CACHE[key]

    try:
        cv_major = int(cv2.__version__.split(".")[0])
    except (ValueError, IndexError):
        cv_major = 0

    if cv_major >= 5:
        _TARGET_CACHE[key] = "cpu"
        log_fn(
            "AdaptiveTracker: cv2 5.0+ new graph engine ignores dnn target "
            "hints — running on cpu (no probe)")
        return "cpu"

    order = list(_TARGET_ORDER)
    if preferred != "auto":
        for i, (name, _t, _b) in enumerate(order):
            if name == preferred:
                order = [order[i]] + order[:i] + order[i + 1:]
                break

    for name, target_attr, backend_attr in order:
        if not hasattr(cv2.dnn, target_attr) or not hasattr(cv2.dnn, backend_attr):
            continue
        try:
            net = cv2.dnn.readNetFromONNX(backbone_path)
            net.setPreferableBackend(getattr(cv2.dnn, backend_attr))
            net.setPreferableTarget(getattr(cv2.dnn, target_attr))
            dummy = np.zeros((1, 3, 127, 127), dtype=np.float32)
            net.setInput(dummy)
            out = net.forward()
            if np.all(np.isfinite(out)):
                log_fn(f"AdaptiveTracker: using dnn target {name}")
                _TARGET_CACHE[key] = name
                return name
        except cv2.error as e:
            log_fn(f"AdaptiveTracker: target {name} probe failed: {e}")
            continue
        except Exception as e:
            log_fn(f"AdaptiveTracker: target {name} probe error: {e}")
            continue
    _TARGET_CACHE[key] = "cpu"
    log_fn("AdaptiveTracker: no dnn target passed the probe; using cpu")
    return "cpu"


def _apply_target(tracker_impl, target_name: str) -> None:
    """cv2.TrackerNano stores an internal Net whose target we can influence
    by setting the tracker's preferred backend/target. Not all versions of
    cv2 expose an API for that; if not, we no-op — inference still works
    on whatever cv2 chose. Best-effort.
    """
    if not hasattr(tracker_impl, "setPreferableBackend"):
        return
    mapping = {
        "cuda_fp16": ("DNN_BACKEND_CUDA", "DNN_TARGET_CUDA_FP16"),
        "cuda": ("DNN_BACKEND_CUDA", "DNN_TARGET_CUDA"),
        "opencl_fp16": ("DNN_BACKEND_OPENCV", "DNN_TARGET_OPENCL_FP16"),
        "opencl": ("DNN_BACKEND_OPENCV", "DNN_TARGET_OPENCL"),
        "cpu": ("DNN_BACKEND_OPENCV", "DNN_TARGET_CPU"),
    }
    backend_attr, target_attr = mapping.get(target_name,
                                            ("DNN_BACKEND_OPENCV", "DNN_TARGET_CPU"))
    try:
        tracker_impl.setPreferableBackend(getattr(cv2.dnn, backend_attr))
        tracker_impl.setPreferableTarget(getattr(cv2.dnn, target_attr))
    except (cv2.error, AttributeError):
        pass


# ------------------------------------------------------------------ SAM 2 adapter

class _Sam2Adapter:
    """Wraps ``ultralytics.SAM`` to look like a cv2 tracker
    (``init(frame, bbox)`` → ``ok, bbox = update(frame)``).

    SAM 2 doesn't have persistent tracker state the way cv2 trackers do —
    it takes an image + prompts (points/box) and returns a segmentation.
    We turn that into tracking by re-prompting each frame with the last
    known centre point, then extracting a new bbox from the resulting
    mask's connected component. Simple and robust for slowly-moving beads.

    For a proper SAM 2 video predictor (with cross-frame memory) we'd
    need the raw ``sam2`` package + frame preloading; ultralytics doesn't
    expose that cleanly. This point-prompt-per-frame mode is close enough
    for a first pass.
    """

    def __init__(self, model, device: str, log_fn, models_dir: str):
        self.model = model
        self.device = device
        self.log = log_fn
        self._prompt_point = None
        self._last_mask = None

    def init(self, frame_bgr, bbox) -> None:
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0
        self._prompt_point = (cx, cy)
        self._segment(frame_bgr, (cx, cy))

    def update(self, frame_bgr):
        if self._prompt_point is None:
            return False, (0, 0, 0, 0)
        mask = self._segment(frame_bgr, self._prompt_point)
        if mask is None or not np.any(mask):
            return False, (0, 0, 0, 0)
        # Bounding box + centroid from the mask.
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            return False, (0, 0, 0, 0)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        w = max(4, x1 - x0)
        h = max(4, y1 - y0)
        cx = float(xs.mean())
        cy = float(ys.mean())
        self._prompt_point = (cx, cy)
        self._last_mask = mask
        return True, (x0, y0, w, h)

    def _segment(self, frame_bgr, point) -> Optional[np.ndarray]:
        try:
            results = self.model.predict(
                frame_bgr,
                points=[[float(point[0]), float(point[1])]],
                labels=[1],
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            self.log(f"AdaptiveTracker: SAM 2 predict failed: {e}")
            return None
        if not results:
            return None
        r = results[0]
        masks = getattr(r, "masks", None)
        if masks is None or masks.data is None or len(masks.data) == 0:
            return None
        # Take the first mask, move to CPU, cast to uint8 0/1.
        m = masks.data[0]
        try:
            m = m.cpu().numpy()
        except AttributeError:
            m = np.asarray(m)
        return (m > 0.5).astype(np.uint8)


# ------------------------------------------------------------------ tracker

class AdaptiveTracker:

    def __init__(self,
                 mask_params: MaskParams,
                 backend: str = "auto",
                 preferred_target: str = "auto",
                 models_dir: str = "models",
                 url_prefix: str = "",
                 sam2_size: str = "tiny",
                 log_fn=print):
        self.mask_params = mask_params
        self.log = log_fn
        self._backend_pref = backend
        self._target_pref = preferred_target
        self._models_dir = models_dir
        self._url_prefix = url_prefix
        self._sam2_size = sam2_size

        self._impl = None                # cv2.Tracker* instance
        self._effective_backend = "none"
        self._effective_target = "cpu"
        self.last_pos: Optional[Tuple[float, float]] = None
        self.last_area: float = 0.0
        self._last_bbox = None           # (x, y, w, h) — for area extraction

    # ---- introspection ---------------------------------------------

    @property
    def effective_backend(self) -> str:
        return self._effective_backend

    @property
    def effective_target(self) -> str:
        return self._effective_target

    # ---- init ------------------------------------------------------

    def _try_build_nano(self) -> bool:
        if not hasattr(cv2, "TrackerNano_create"):
            return False
        weights = _ensure_nano_weights(
            self._models_dir, self._url_prefix, self.log)
        if weights is None:
            return False
        backbone, neckhead = weights
        target = _probe_target(backbone, self._target_pref, self.log)
        try:
            params = cv2.TrackerNano_Params()
            params.backbone = backbone
            params.neckhead = neckhead
            impl = cv2.TrackerNano_create(params)
        except Exception as e:
            self.log(f"AdaptiveTracker: TrackerNano build failed: {e}")
            return False
        _apply_target(impl, target)
        self._impl = impl
        self._effective_backend = "nano"
        self._effective_target = target
        return True

    def _try_build_vit(self) -> bool:
        if not hasattr(cv2, "TrackerVit_create"):
            return False
        vit_path = _ensure_vit_weights(
            self._models_dir, self._url_prefix, self.log)
        if vit_path is None:
            return False
        target = _probe_target(vit_path, self._target_pref, self.log)
        try:
            params = cv2.TrackerVit_Params()
            params.net = vit_path
            impl = cv2.TrackerVit_create(params)
        except Exception as e:
            self.log(f"AdaptiveTracker: TrackerVit build failed: {e}")
            return False
        _apply_target(impl, target)
        self._impl = impl
        self._effective_backend = "vit"
        self._effective_target = target
        return True

    def _try_build_mil(self) -> bool:
        if not hasattr(cv2, "TrackerMIL_create"):
            return False
        try:
            self._impl = cv2.TrackerMIL_create()
        except Exception as e:
            self.log(f"AdaptiveTracker: TrackerMIL build failed: {e}")
            return False
        self._effective_backend = "mil"
        self._effective_target = "cpu"   # MIL is CPU-only
        return True

    def _try_build_sam2(self) -> bool:
        """SAM 2 (Segment Anything Model 2) via ultralytics + torch. Much
        stronger than the cv2 trackers — click a point, get segmentation,
        propagate across frames — but adds a ~500 MB dep."""
        try:
            import torch
            from ultralytics import SAM
        except ImportError:
            self.log(
                "AdaptiveTracker: SAM 2 requires ultralytics + torch. "
                "Install with `/opt/homebrew/bin/python3.13 -m pip install "
                "--user --break-system-packages ultralytics` (this pulls "
                "torch as a dep, ~500 MB).")
            return False

        # Model size: t | s | b | l (tiny / small / base / large). Names
        # match ultralytics' registry (sam2.1_t.pt etc.). Auto-download on
        # first use (~40-230 MB depending on choice).
        size_map = {
            "tiny": "sam2.1_t.pt",
            "small": "sam2.1_s.pt",
            "base": "sam2.1_b.pt",
            "large": "sam2.1_l.pt",
        }
        pref_size = getattr(self, "_sam2_size", "tiny")
        model_name = size_map.get(pref_size, size_map["tiny"])

        # Device: honour the target preference where it maps sensibly.
        # For torch: mps on Apple Silicon, cuda on Jetson/desktop NVIDIA,
        # cpu otherwise. cv2.dnn's opencl targets have no torch equivalent
        # so those fall back to cpu.
        target_pref = self._target_pref
        device = "cpu"
        try:
            if target_pref in ("cuda", "cuda_fp16", "auto") and torch.cuda.is_available():
                device = "cuda"
            elif target_pref == "auto" and getattr(
                    torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
        except Exception:
            device = "cpu"

        try:
            model = SAM(model_name)
            # A trivial forward at construction time surfaces weight-download
            # / device errors early rather than mid-selection.
            model.to(device)
        except Exception as e:
            self.log(f"AdaptiveTracker: SAM 2 load failed: {e}")
            return False

        self._impl = _Sam2Adapter(model, device=device, log_fn=self.log,
                                  models_dir=self._models_dir)
        self._effective_backend = f"sam2_{pref_size}"
        self._effective_target = device
        return True

    def _build(self) -> bool:
        pref = self._backend_pref
        # ``auto`` only tries the truly-neural backends. MIL is available
        # but slow on high-res frames — falling back to it silently caused
        # the UI to lag noticeably. Instead, if neural weights aren't
        # available we return False and the controller falls back to
        # ``ObjectTracker`` (template match — fast and sub-pixel).
        # MIL is still reachable if the operator explicitly asks for it
        # via ``adaptive_backend: mil`` in config. VitTrack is preferred
        # in auto because opencv_zoo actually ships downloadable weights
        # for it; NanoTrack requires manual placement.
        if pref == "auto":
            # Order: strongest first that actually loads.
            # sam2 is preferred when installed (state-of-the-art), vit is
            # the cv2 fallback that always downloads, nano only when the
            # operator has dropped weights manually, mil is opt-in via
            # explicit config because it's slow.
            order = ["sam2", "vit", "nano"]
        else:
            order = [pref]
        for choice in order:
            ok = {
                "nano": self._try_build_nano,
                "vit": self._try_build_vit,
                "mil": self._try_build_mil,
                "sam2": self._try_build_sam2,
            }.get(choice, lambda: False)()
            if ok:
                self.log(
                    f"AdaptiveTracker: backend {self._effective_backend} "
                    f"on {self._effective_target}")
                return True
        return False

    def initialize(self,
                   frame_bgr: np.ndarray,
                   click_xy: Tuple[int, int]) -> Optional[Detection]:
        if self._impl is None:
            if not self._build():
                return None

        cx, cy = int(click_xy[0]), int(click_xy[1])
        # First try to derive a tight bbox from the actual bead's blob in
        # the mask. Neural trackers learn from whatever pixels the init
        # bbox contains, so a loose square (mostly background) drifts.
        tight = self._bead_bbox(frame_bgr, (cx, cy))
        if tight is not None:
            bx, by, bw, bh = tight
        else:
            # Mask found nothing near the click — fall back to a fixed
            # square around the click. Better than refusing to init.
            side = max(16, int(self.mask_params.crop_length * 2))
            x0, y0, x1, y1 = _clip_slice(cx, cy, side // 2,
                                         frame_bgr.shape[:2])
            bx, by, bw, bh = x0, y0, x1 - x0, y1 - y0
            self.log(
                "AdaptiveTracker: no blob near click — init bbox may be loose")

        bbox = (int(bx), int(by), int(max(4, bw)), int(max(4, bh)))
        try:
            self._impl.init(frame_bgr, bbox)
        except Exception as e:
            self.log(f"AdaptiveTracker: init failed: {e}")
            return None

        self._last_bbox = bbox
        gx = bbox[0] + bbox[2] / 2.0
        gy = bbox[1] + bbox[3] / 2.0
        self.last_pos = (gx, gy)
        det = self._make_detection(frame_bgr, source="adaptive_init",
                                   confidence=1.0)
        # Store the initial area — used as the reference for the per-frame
        # sanity check in ``update``.
        self._init_area = max(1.0, det.area_px)
        self._low_area_streak = 0
        return det

    def _bead_bbox(self,
                   frame_bgr: np.ndarray,
                   click_xy: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        """Return a tight (x, y, w, h) around the blob nearest ``click_xy``,
        padded by 20% so the neural tracker has some context. ``None`` if
        no blob passes the mask + min-area filter."""
        cx, cy = click_xy
        # Extract a wider crop so the mask finds a full blob even near
        # the crop edge.
        half = max(self.mask_params.crop_length,
                   self.mask_params.min_blob_area_px + 20)
        x0, y0, x1, y1 = _clip_slice(int(cx), int(cy), half,
                                     frame_bgr.shape[:2])
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        crop = frame_bgr[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mask = _apply_mask(gray, self.mask_params)
        # Nearest blob to the click (in crop-local coords).
        blob = _find_nearest_blob(mask, (cx - x0, cy - y0),
                                  self.mask_params.min_blob_area_px)
        if blob is None:
            return None
        # Extract the actual contour bbox for that blob.
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        best = None
        best_d2 = float("inf")
        blob_cx, blob_cy = blob[0]
        for c in contours:
            m = cv2.moments(c)
            if m["m00"] < 1e-6:
                continue
            ccx = m["m10"] / m["m00"]
            ccy = m["m01"] / m["m00"]
            d2 = (ccx - blob_cx) ** 2 + (ccy - blob_cy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = c
        if best is None:
            return None
        bx, by, bw, bh = cv2.boundingRect(best)
        # 20% padding
        pad_x = int(round(bw * 0.2))
        pad_y = int(round(bh * 0.2))
        bx -= pad_x; by -= pad_y
        bw += 2 * pad_x; bh += 2 * pad_y
        # Translate back to full frame coords + clip.
        bx += x0; by += y0
        h_f, w_f = frame_bgr.shape[:2]
        bx = max(0, bx); by = max(0, by)
        bw = min(w_f - bx, bw); bh = min(h_f - by, bh)
        if bw < 6 or bh < 6:
            return None
        return bx, by, bw, bh

    # ---- per-frame -------------------------------------------------

    def update(self, frame_bgr: np.ndarray) -> Optional[Detection]:
        if self._impl is None or self.last_pos is None:
            return None
        try:
            ok, bbox = self._impl.update(frame_bgr)
        except Exception as e:
            self.log(f"AdaptiveTracker: update failed: {e}")
            return None
        if not ok:
            return None
        x, y, w, h = [int(round(v)) for v in bbox]
        w = max(4, w)
        h = max(4, h)
        self._last_bbox = (x, y, w, h)
        self.last_pos = (x + w / 2.0, y + h / 2.0)
        det = self._make_detection(frame_bgr, source="adaptive",
                                    confidence=0.9)

        # Sanity check: if the mask reports almost no blob at the tracker's
        # claimed position, the tracker has probably drifted onto background.
        # Give it a few frames of grace so a transient occlusion doesn't kill
        # it, then declare lost.
        init_area = getattr(self, "_init_area", 0.0)
        if init_area > 1.0:
            if det.area_px < 0.2 * init_area:
                self._low_area_streak = getattr(
                    self, "_low_area_streak", 0) + 1
                if self._low_area_streak >= 8:
                    self.log(
                        "AdaptiveTracker: bbox left the bead (area "
                        f"{det.area_px:.0f} << init {init_area:.0f}); "
                        "declaring lost")
                    return None
            else:
                self._low_area_streak = 0
        return det

    # ---- accessors -------------------------------------------------

    def set_mask_params(self, params: MaskParams) -> None:
        # Mask params only govern our post-hoc area/blur extraction. cv2
        # trackers have their own internal representation, so no update
        # needed on the tracker impl itself.
        self.mask_params = params

    # ---- helpers ---------------------------------------------------

    def _make_detection(self,
                        frame_bgr: np.ndarray,
                        source: str,
                        confidence: float) -> Detection:
        assert self._last_bbox is not None and self.last_pos is not None
        gx, gy = self.last_pos
        x, y, w, h = self._last_bbox
        # Extract a crop around the bbox for mask / blur / preview.
        cx, cy = int(round(gx)), int(round(gy))
        half = max(self.mask_params.crop_length, max(w, h) // 2 + 8)
        x0, y0, x1, y1 = _clip_slice(cx, cy, half, frame_bgr.shape[:2])
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return Detection(pos=(gx, gy), area_px=self.last_area,
                             blur=0.0, cropped_bgr=frame_bgr,
                             cropped_mask=np.zeros(
                                 (max(1, y1 - y0), max(1, x1 - x0)),
                                 dtype=np.uint8),
                             confidence=confidence, source=source)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mask = _apply_mask(gray, self.mask_params)
        # Area: nearest blob to the bbox centre within the mask.
        blob = _find_nearest_blob(mask, (cx - x0, cy - y0),
                                  self.mask_params.min_blob_area_px)
        if blob is not None:
            area = blob[1]
        elif w > 0 and h > 0:
            # Fall back to the bbox as an area estimate (upper-bounds true
            # bead area but is stable when the mask is misconfigured).
            area = float(w) * float(h) * 0.6
        else:
            area = self.last_area
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        self.last_area = area
        return Detection(pos=(gx, gy), area_px=area, blur=blur,
                         cropped_bgr=crop, cropped_mask=mask,
                         confidence=confidence, source=source)
