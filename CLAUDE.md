# BigTweezerSystem — Magnetoacoustic Microrobotic Manipulation System

## What this project is

A PyQt5 desktop application that controls a benchtop / portable microrobotic experimentation platform combining **magnetic** and **acoustic** actuation. It ingests a live microscope camera feed (FLIR / EasyPySpin, or a video file), tracks micro-robots and cells in real time via OpenCV, and sends per-coil PWM commands over USB serial to an **Arduino**, which drives the 6-coil H-bridge stack. An AD9850 DDS module on the Arduino generates the acoustic transducer signal.

The Python GUI is the *host-side* software. It runs on macOS, Windows, and Nvidia Jetson AGX Orin.

## Coil geometry and field synthesis (rewritten on `rewrite-3d-control` branch)

The rig is a **6-coil 3D-tweezer** with two aligned rings of three:
- **Top ring**: coils C1, C2, C3 at azimuths 0°, 120°, 240°. Each axis 45° from vertical, tips point inward.
- **Bottom ring**: coils C4, C5, C6 at the same azimuths, mirrored below.

All physics lives in `classes/field_synth.py`. The 3×6 matrix `COIL_AXES` is the single source of truth for geometry. Uniform-field targets are solved via minimum-norm pseudoinverse; the geometry gives `A · Aᵀ = diag(3/2, 3/2, 3)` exactly, so the pseudoinverse is a two-matmul closed form. Gradient synthesis uses `I_i = g · (n̂_i · d̂)`. Rolling produces a zero-mean current delta on top of the static uniform contribution.

The Arduino is a **dumb 6-channel PWM writer**: it reads a 7-float packet `[I1..I6, acoustic_freq]` and calls `set1(I1)..set6(I6)` directly. No coil geometry lives in the firmware. Firmware file: `arduino_files/main_3DTweezers.ino`. ~170 lines.

## Coil calibration

Each rig's six coils inevitably deliver slightly different fields for identical PWM duty (winding variation, driver-channel differences, wiring path lengths). Per-coil gains live in `calibration.json` at the repo root — gitignored, so operator gains stay local. A neutral template lives in `calibration_example.json`.

Operator flow:
1. Launch the app. The **Field Controls** dock (top of the window) has a **Calibration** tab.
2. Adjust "Test drive strength" (default 0.3 = 30% PWM).
3. For each coil in turn: click **Test**. The Arduino fires only that coil at that strength. Read the magnetometer at a fixed reference position.
4. Adjust that coil's **Gain** spinbox until the magnetometer reads the target field (e.g. 2 mT).
5. When all six read the same, click **Save + Apply**. Values persist to `calibration.json` and take effect on the next serial send.
6. On next launch the app auto-loads `calibration.json` if present.

Negative gains are legal — they invert an individual coil's H-bridge polarity for coils wound backward at build time, without a code change.

## Entry point

- `main.py` — creates a `QApplication` and instantiates `classes.gui_functions.MainWindow`. That's the whole launcher.

Run locally:
```
python3 main.py
```

Building a Mac app bundle (from the README):
```
/opt/homebrew/bin/python3.10 -m PyInstaller --onedir --windowed --icon MagScopeBox.icns --name MagScope main.py
```

Regenerating the Qt UI Python from the `.ui` file:
```
pyuic5 uis/GUI.ui -o gui_widgets.py
```

## Repository layout

```
main.py                          # Qt entry point
reqs.txt / reqswindows.txt       # pip requirements (reqs.txt is UTF-16-ish; treat carefully)
README.md                        # hardware/OS setup instructions for Jetson Orin
magscopeUIs.pptx                 # slide deck of UI variants

uis/                             # Qt Designer .ui files (GUI.ui + platform variants)
imgs/                            # icons, screenshots, a sample test video (test2.mp4)
action_data/control_actions.xlsx # example imported "Excel actions" script
Data/                            # runtime output directory (tracking data, recordings)
Microrobots/Tracking Data/       # alt output path used on Windows (D:/Microrobots/...)
old/                             # legacy / archived versions of files — do not edit
classes/                         # all application logic (see below)
```

## `classes/` — application code

Two large modules make up the app:

- `gui_functions.py` (~1300 lines) — `MainWindow(QMainWindow)`. All Qt signal wiring, event handlers, mouse/wheel/keyboard interaction with the video display, joystick polling, calibration, Excel-action playback, and coordination between the tracker thread, the Arduino, and the acoustic hardware.
- `gui_widgets.py` (~1300 lines) — **generated** by `pyuic5` from `uis/GUI.ui`. Do not hand-edit; regenerate from the `.ui` file instead.

Domain / hardware modules:

- `tracker_class.py` — `VideoThread(QThread)`. Owns the OpenCV capture loop, per-frame robot/cell masking (HSV thresholds, dilation, blur, invert), cropped-frame extraction, FPS counting, and emits `change_pixmap_signal`, `cropped_frame_signal`, `actions_signal` back to the GUI. Delegates control decisions to `algorithm`.
- `algorithm_class.py` — `algorithm`. Path-following, orientation, and auto-acoustic frequency-sweep logic. Given a robot's current state and a trajectory, produces the `[Bx, By, Bz, alpha, gamma, freq, psi, acoustic_freq]` action tuple to send to the Arduino. Uses `mpc/rrtstar.py` for path planning.
- `mpc/` — planning + control research code: `rrtstar.py`, `RRT.py`, `MPC.py`, `MR_simulator.py`, `Learning_module_2d.py`, `mpc_algorithm_class.py`, `p_algorithm_class.py`.
- `robot_class.py` / `cell_class.py` — per-tracked-object state containers (position list, velocity list, blur/z estimate, cropped frames, trajectory, timing, µm/pixel conversion).
- `arduino_class.py` — `ArduinoHandler`. Thin wrapper over `pySerialTransfer`; `send()` packs the 10-float control message `[Bx, By, Bz, alpha, gamma, freq, psi, gradient_status, equal_field_status, acoustic_freq]` for the Arduino firmware.
- `acoustic_class.py` — `AcousticClass`. Drives the AD9850 DDS + X9C104 digital pot over `RPi.GPIO`; import is guarded so the module is a no-op on non-Pi platforms.
- `halleffect_class.py` — Hall-effect field-sensor readout (also RPi.GPIO / I2C).
- `joystick_class.py` — `Mac_Controller`, `Linux_Controller`, `Windows_Controller` (pygame-based) — map Xbox controller axes/buttons to field/frequency commands with per-OS button remapping and a deadzone helper.
- `simulation_class.py` — `HelmholtzSimulator(FigureCanvas)`. Embedded matplotlib 3D animation of the rotating B-field vector for live visualization inside the Qt UI.
- `projection_class.py` — `AxisProjection`. Utility for projecting joystick / on-screen 2D input onto the physical coil axes given a calibrated rotation.
- `record_class.py` — `RecordThread`. Separate QThread that writes video recordings without blocking the tracker.
- `fps_class.py` — `FPSCounter` used by `tracker_class`.

Firmware sources (kept alongside the Python for convenience):

- `classes/main.ino` — Arduino firmware for the coil driver.
- `classes/main_Bigtweezers/main_Bigtweezers.ino` — variant firmware for the "Big Tweezers" hardware.
- `classes/main/` — legacy `main.ino` archive.
- `classes/arduino_correct.txt` — notes / snippet reference.

## Key runtime behaviors worth knowing

- The GUI resizes itself against the actual screen size on startup (`resize_widgets()` and the `displayheightratio` / `aspectratio` fields on `MainWindow`); layout is not fixed pixel-perfect.
- Output directory: on Windows the app writes to `D:/Microrobots/Tracking Data`; on macOS/Linux it uses the repo-local `Data/` folder. Both are created on startup if missing.
- Micrometers per pixel is derived from objective magnification: `um2pixel = 3.35 / objective`.
- "Apply Excel Actions" reads `action_data/control_actions.xlsx` and applies one row of actions per camera frame — a scripted-playback mode for reproducible experiments.
- Camera acquisition uses `EasyPySpin` (FLIR Spinnaker); the import is wrapped in try/except so dev machines without the SDK still load. See README step 7 for a required in-place patch to `EasyPySpin/videocapture.py`.
- Hardware imports (`RPi.GPIO`, FLIR SDK, serial ports) are all guarded — cross-platform is a first-class requirement, do not remove those guards.

## Editing conventions

- **Do not hand-edit `classes/gui_widgets.py`.** Regenerate from `uis/GUI.ui` with `pyuic5`. The four `Magscope*.ui` files are per-OS layout variants of the same UI.
- Legacy code lives in `old/` — treat as read-only reference.
- The Arduino action packet is a fixed 10-float schema (see `arduino_class.ArduinoHandler.send`). If you add a field, both firmware (`main.ino`) and every call site of `send(...)` must change together.
- `algorithm.reset()` in `algorithm_class.py` is the canonical list of controller state — new controller fields should be initialized there.
