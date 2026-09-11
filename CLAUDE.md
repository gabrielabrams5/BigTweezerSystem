# BigTweezerSystem — Magnetoacoustic Microrobotic Manipulation System

## What this project is

A PyQt5 desktop application that controls a benchtop / portable microrobotic experimentation platform combining **magnetic** and **acoustic** actuation. It ingests a live microscope camera feed (FLIR / EasyPySpin, or a video file), tracks micro-robots and cells in real time via OpenCV, and sends per-coil PWM commands over USB serial to an **Arduino**, which drives the 6-coil H-bridge stack. An AD9850 DDS module on the Arduino generates the acoustic transducer signal.

The Python GUI is the *host-side* software. It runs on macOS, Windows, and Nvidia Jetson AGX Orin.

## Architecture (post-overhaul, current)

The Python side is split into five modules with clean interfaces. Everything
above HAL is hardware-free and unit-testable; the Supervisor sits between
the solver and HAL with veto authority.

- `classes/config_loader.py` — YAML config as single source of truth
  (`config.yaml`; template committed as `config_example.yaml`).
  Dotted-path access, atomic save, listener callbacks for live UI reload.
- `classes/robot_model.py` — three physics models (paramagnetic, soft, hard);
  selected by `robot.type`.
- `classes/field_solver.py` — pure-math solver. Paramagnetic default uses
  the closed-form `max(0, Bmap.T @ B_des)` with peak normalization so the
  strongest coil always saturates at `|B_des|`. Signed-drive rigs use DLS
  (Tikhonov damped least squares) with optional null-space objective and a
  configurable saturation policy.
- `classes/motion_controller.py` — Mode A (rotating pull, phase accumulator
  in a specified `(û, v̂)` plane at a ramped frequency) or Mode B (static
  pull, direction × magnitude). `tick(dt)` runs the solver and emits solved
  currents to listeners.
- `classes/supervisor.py` — independent safety authority. Watchdog on last
  solve, total-current budget, rolling I²R per-coil power estimate
  (thermistor stand-in), and a latched E-stop. Every current sent to the
  HAL passes through it.
- `classes/hal.py` — `ArduinoHAL` (real serial via `ArduinoHandler`) or
  `PrintHAL` (no hardware; logs packets). Auto-discovers the serial port
  from `serial.port_glob`.
- `classes/gui/*` — pure Python PyQt5 with **no Qt Designer `.ui` files**.
  Panels: Mode A, Mode B, Solver, Calibration wizard (guided gaussmeter
  entry), Supervisor (live coil bars + I²R + E-stop).

Entry point:
```
python3 main.py                 # new stack
python3 main.py --print-only    # no serial; PrintHAL for GUI development
python3 main_legacy.py          # legacy stack (rollback path)
```

## Coil geometry

The rig is a **6-coil 3D tweezer** with two aligned rings of three:
- **Top ring**: coils C1, C2, C3 at azimuths 0°, 120°, 240°. Each axis 45° from vertical, tips point inward.
- **Bottom ring**: coils C4, C5, C6 at the same azimuths, mirrored below.

Geometry-derived defaults for `Bmap` live in `config_example.yaml`; measured values are written back into `config.yaml` by the Calibration wizard.

The Arduino remains a **dumb 6-channel PWM writer**: it reads a 7-float packet `[I1..I6, acoustic_freq]` and calls `set1(I1)..set6(I6)` directly. Firmware file: `arduino_files/main_3DTweezers/main_3DTweezers.ino`. Now runs at **500000 baud** with a **500 ms soft-watchdog** that zeroes all coils and stops the DDS if the host stops sending packets — belt-and-braces with the Python-side Supervisor.

## Physics model

**Static pull (Mode B) uses an exact solve.** The pull-only rule `max(0, n̂·B)` is a *projection heuristic*, not a solve: it fires each coil by how well its axis projects onto the command, which does **not** reproduce the commanded direction. With three coil azimuths (0/120/240°) the realized horizontal field is a **±30° sawtooth** about the command — exact only at multiples of 60°, and with **flat dead zones** where a whole 30° span of commands maps to one realized direction (the operator sees the bead refuse to turn, then jump). Mode A is unaffected: it precesses through a full cycle so the per-instant distortion averages out, and Frame Cal measures rolling end-to-end anyway, absorbing the residue. Mode B steers directly by the realized direction, so there it is the dominant error.

The commanded direction *is* reachable with non-negative currents — the heuristic just doesn't find it. `FieldSolver._solve_paramagnetic_exact` solves `min ‖[Bmap; √λ·I]·i − [B_des; 0]‖` subject to `i ≥ 0` (scipy NNLS, ~63 µs against a 5000 µs budget at 200 Hz), giving **0.004°** error across the full azimuth sweep while keeping pull-only physics. The ridge λ (`solver.static_ridge`, 1e-4) is not for conditioning: plain NNLS returns a minimal-support solution that breaks top/bottom ring symmetry and tilts the |B| landscape out of plane; a small λ selects the minimum-norm exact solution, restoring symmetric firing. `MotionController.tick` passes `static=(mode == Mode.B_STATIC)`, so **Mode A output is bit-identical to before and the existing Frame Cal stays valid**. Escape hatch: `solver.static_synthesis: projection`.

*Caveat that no solver fixes:* paramagnetic force follows `∇|B|²` **at the bead's position**, and `solve()` takes no position argument — nothing models the gradient (`Gmap` is all zeros and unreachable on the paramagnetic path). Static pull therefore stays inherently position-dependent: the bead is drawn toward the fired coil cores, so the direction varies across the field of view. The exact solve removes a large *systematic* error; it does not make static as spatially robust as rolling. **Rolling remains the default for locomotion.**

- **Robot type**: `paramagnetic` by default. Configured Spherotech CFM magnetic beads are superparamagnetic — force is always attractive (`F ∝ ∇|B|²`), reversing coil polarity does nothing useful. Under this model the solver uses the closed-form pull rule.
- The `soft` and `hard` robot types exist in code but aren't the current hardware target. If you enable one, the DLS solver activates.

## Calibration workflow

`config.yaml` is gitignored — operator tuning stays local. `config_example.yaml` is committed as a neutral template with geometry-derived defaults.

**Probe assumed: TD8620** (or any scalar / single-axis Hall probe). Measurement model: the operator physically places the probe tip on the **back of each coil's core** in turn. That gives one scalar per (coil, duty) — the field strength at that coil's own reference point. Direction comes from the rig's known geometry (`_GEOM_AXES` in `calibration_wizard.py`, matching `config_example.yaml`'s Bmap direction convention). Bmap column j = `slope_j · n̂_j`. This captures per-coil strength variation exactly; it does *not* validate physical alignment (rebuilt/moved coils need their geometry updated in the source).

Operator flow (Calibration tab):
1. Pick a run mode:
   - **Quick** (default, 6 measurements): fire each coil at 100% duty, one scalar. `slope_j = reading_j / 1.0`.
   - **Full sweep** (24 measurements): four duties per coil with a through-origin linear regression, plus a saturation-onset prompt per coil.
2. Wizard walks per coil: "move probe to back of C1's core" → measure(s) → "move to C2" → measure(s) → … → C6 → optional saturation prompts → summary with per-coil fit residuals → **Save** writes to `config.yaml`.
3. While a measurement card is up the wizard **re-fires the same coil packet every 200 ms** (heartbeat). Without this, the Arduino soft-watchdog (`main_3DTweezers.ino`, 500 ms) would zero the coils while the operator reads the probe and every measurement would come back near zero.
4. If a coil is wound backward relative to the intended geometry, flip its `calibration.per_coil_gains[j]` sign in `config.yaml` — the wizard uses `|slope|` (probe readings are magnitudes; the wizard can't tell "wired backwards" from "wound backwards" on its own).

Legacy `calibration.json` (per-coil scalar gains + channel_map) is auto-migrated into `config.yaml` on first load if `config.yaml` doesn't yet exist. Legacy gain-only entries remain honored under `calibration.per_coil_gains` and `calibration.channel_map`.

**Power defaults.** As of the TD8620 pass, `modes.mode_a.magnitude_default` and `modes.mode_b.magnitude_default` are `1.0` (full-scale) and `solver.saturation_policy` is `flag` — each coil is clipped independently instead of the whole vector being rescaled. Paramagnetic pull benefits: total pull force is higher and one gain > 1 doesn't drag the other coils down. Switch back to `rescale_preserve_direction` when moving to a hard/soft-magnetic robot where field direction matters.

## Joystick control

An Xbox / PS-style pad drives the same `MotionController` API as the GUI panels. `classes/gui/joystick_bridge.py` polls pygame at ~33 Hz on the Qt event loop and pushes commands tagged `Source.JOYSTICK`. The Joystick tab (between Mode B and Solver) shows connection status, a live active-mode indicator (OFF / A — Rolling / B — Static pull), a stick-engage mode selector, live axis bars, live button LEDs, and the mapping legend. It is also the **single owner of the two shared drive knobs** — `modes.mode_a.freq_default` ("Base roll frequency") and `modes.mode_a.magnitude_default` ("Magnitude") — read by the joystick (master gain), path follower, Frame Cal pulses, and the Mode A tab; every other tab shows them read-only.

**Default mapping** (Xbox layout — X=0, B=1, X=2, Y=3, Back=6, Start=7):

| Input | Effect |
|---|---|
| Left stick X / Y | Travel direction. Rolling mode: the bridge recomputes the roll axis every poll as horizontal ⊥ travel, so the field sweeps the vertical plane containing the travel vector and the bead rolls toward the stick. Static mode: plain Bx / By pull. |
| RT / LT triggers | +Bz / −Bz (vertical pull). Triggers alone (sticks centered) in rolling mode → zero roll axis → static vertical pull. |
| Right stick X | Signed roll frequency (rolling mode): centered = `modes.mode_a.freq_default` (base, also settable from the tab spinbox), full right = `modes.mode_a.freq_max_hz`, left = same curve reversed (negative frequency = roll away). |
| A (0) | Toggle Mode B ↔ OFF |
| B (1) | Toggle Mode A ↔ OFF |
| X (2) / Y (3) | Shared Magnitude −0.1 / +0.1 (writes `modes.mode_a.magnitude_default`) |
| Back (6) | E-STOP (latches; needs Start to reset) |
| Start (7) | Reset E-STOP |

**Invert toggles.** "Invert X" / "Invert Y" checkboxes on the Joystick tab (`joystick.invert_x`/`invert_y`) flip the stick's display-frame intent *before* the view rotation and calibration matrix — operator-preference flips that always mean screen-left/right / screen-up/down as displayed.

The bridge auto-connects at startup (200 ms after MainWindow shows), or the operator can Reconnect manually from the tab. Sticks at rest → mode drops to OFF (only when the joystick was the last writer); grab the stick again → mode returns to the operator-selected stick-engage mode: rolling Mode A by default, or static Mode B via `config.yaml → joystick.stick_mode: static` (switchable live from the tab's radio buttons).

**Per-rig remap** — edit `config.yaml → joystick.*` and press Reconnect. `axis_left_x`, `axis_left_y`, `axis_right_x`, `axis_right_y`, `axis_lt`, `axis_rt` accept pygame axis indices. `buttons.mode_a`, `buttons.mode_b`, `buttons.mag_up`, `buttons.mag_down`, `buttons.estop`, `buttons.reset_estop` accept pygame button indices (set to `-1` to unbind).

**Cross-platform axis note.** Defaults target macOS + Windows. Linux's SDL layout is different (right stick at (3,4), triggers at (2,5)) — the bridge picks the Linux mapping automatically from `platform.system()`, and the config knobs still take precedence.

**macOS 15 caveat.** SDL / pygame need "Input Monitoring" permission for the terminal (or app bundle) launching Python. If a pad is clearly plugged in but the panel says "no joystick", check System Settings → Privacy & Security → Input Monitoring.

## Tracker + path following (Tracker tab)

Click-to-select a microrobot on the video, right-click to draw a path, Start to have the robot autonomously walk that path. Two operation modes on the Tracker tab:

- **Follow robot path** (default, "simple mode") — only robot controls are visible. Robot mask sliders, cropped preview, five metrics (diameter / speed / acceleration / blur / ambiguity), arrival threshold, magnitude, Start / Stop / Clear.
- **Push cells sequencer** — the panel expands to reveal the cell mask group, a cells list, and push-specific spinboxes (push offset, approach threshold). Operator marks non-magnetic cells and draws a path per cell; the robot visits each cell in turn, positions behind it relative to the cell's next waypoint (APPROACH), then pulls itself toward the waypoint so the cell is mechanically dragged along (PUSH). APPROACH ↔ PUSH is a stateless per-frame recomputation, so mid-push drift auto-corrects.

**Vision model.** `classes/gui/tracker.py::ObjectTracker` initialises via grayscale-mask detection at click time (same paradigm as legacy: `lower / upper / blur / dilation / crop_length / invert`), extracts a template patch, then per frame runs `cv2.matchTemplate(..., TM_CCOEFF_NORMED)` in a search window around the **predicted** position. Sub-pixel refined via 3-point parabolic fit. Falls back to mask relocalization inside the same window when correlation drops below `tracker.min_match_confidence` (default 0.5). Template refreshes every `tracker.template_refresh_s` seconds to guard slow appearance drift.

**Identical-neighbour defence (the anti-swap stack).** Plain `minMaxLoc` is a maximum-likelihood pick: with two near-identical beads in the window it produces two nearly-equal peaks and returns whichever is marginally higher, so *sensor noise* decides which bead is being tracked. `min_match_confidence` is structurally blind to this — in a contested frame **both** peaks score high; that gate only catches low-*absolute* correlation (bead out of focus, left frame). Four mechanisms address it:

1. **Motion prior (MAP, not ML).** `classes/gui/motion_filter.py::ConstantVelocityKF` is a 4-state `[x, y, vx, vy]` filter with a first-order-lag process model (microrobots are overdamped — velocity relaxes toward the commanded drive with time constant `tracker.kf_velocity_tau_s` rather than persisting). Its prediction becomes a Gaussian that multiplies the correlation surface before the argmax, so the peak consistent with where the bead was heading wins. `project()` is pure (used to build a prior for a frame that may later be dropped); `step()` mutates (called only when a detection lands) — dropped frames therefore cannot corrupt the filter.
2. **Commanded velocity as the filter's control input.** Read from `MotionController` state, not the follower's locals, so it works identically whoever is driving — path follower, joystick, or a GUI panel. Scale comes from `calibration.screen_response_2x2`, the **raw** Frame Cal response matrix (px per unit world command per Hz·s). `screen_to_world_2x2` cannot serve here: it is normalized to direction-only and deliberately discards the pixel scale. Absent (never calibrated) → the filter degrades to plain constant velocity, still correct, just less sharp. Only rolling (Mode A) is modelled; magnitude is deliberately not a factor since above the rolling threshold it sets force, not speed.
3. **Ambiguity detection + contested state machine.** `peak_ambiguity()` suppresses a disc around the winner on the *unweighted* surface and re-runs `minMaxLoc`; `ambiguity = peak2/peak1`. Above `tracker.ambiguity_enter` (0.85) the tracker latches **contested**, releasing below `ambiguity_exit` (0.65). While contested: the template is **frozen** (neither the periodic refresh nor the mask fallback re-cuts — a refresh taken while the match has half-slid onto a neighbour makes the drift permanent and unrecoverable), the controller **coasts** on the filter instead of trusting the peak and pushes the coasted position back via `set_position()` so the search window doesn't walk onto the neighbour, the overlay box turns **amber** and bold, and the follower **holds station** (field OFF, `running` still armed) until the contest clears or `path_follow.contested_timeout_s` elapses. Enter/exit events with ambiguity, both candidate positions, and timestamps append to `tracking_events_<stamp>.jsonl` in the recording output dir.
   *Merge caveat:* two beads that overlap make **one** blob and **one** peak, so the peak ratio collapses exactly at closest approach — the moment of maximum danger reads as safe. Blob-area inflation (`merge_area_ratio`) plus a minimum dwell hold the latch through that dip.
4. **Distractor tracking.** `classes/gui/distractors.py` runs the same operator-tuned mask over the **whole** frame at `tracker.distractors.rate_hz` (8 Hz default) on its own drop-old thread, using `cv2.connectedComponentsWithStats` (scipy's `ndimage.label` is far too slow full-frame), and keeps a light constant-velocity track per blob. Their predicted positions are subtracted as Gaussian bumps from the correlation surface before the argmax — you cannot exclude a neighbour you aren't modelling. Per-track running means of blob area and Laplacian variance serve as association tie-breakers. The controller filters the robot's own blob out of that list by radius; suppressing it would push the tracker straight onto a neighbour.

**Search-window sizing tension.** The ambiguity detector can only see a neighbour within `(search_half − template_half)` px of the prediction — ~32 px at the defaults, because the correlation surface is only `window − template + 1` across. Widening `tracker.search_half_px` (0 = derive from `crop_length` × 1.5, the historical behaviour) buys earlier contest warning at the cost of a bigger `matchTemplate`. The suppression disc is clamped to 0.35 × the surface so it can never swallow the map and silently report "no runner-up".

**Timing is wall-clock, not frame counts.** `_TrackerAsync` uses drop-old semantics, so the frame interval is variable and a frame count is an unpredictable amount of real time. `VideoSource.frameReady` carries `(frame, capture_time)` stamped the instant `read()` returns — arrival time on the main thread would fold queue and inference latency into every dt. Both thresholds are seconds: `tracker.lost_timeout_s` (1.5) and `tracker.template_refresh_s` (1.5). This matters most for the safety stop: under load, `lost_frames_stop` consecutive misses could be *seconds* of the robot running open-loop.

**Frame copying.** `tracker.copy_frames` (default true) copies each frame before handing it to the worker. The aravis backend re-latches the same buffer object every grab, so without this the capture thread can overwrite pixels mid-correlation — which surfaces as phantom ambiguity rather than an obvious failure.

**Interaction on the video widget** (see `classes/gui/video.py::VideoWidget`):

| Click | Effect |
|---|---|
| Left click | Select robot (Robot mode) or append new cell target (Cell mode, push-cells only) |
| Right click | Append waypoint to the active target's trajectory |
| Middle click | Clear robot + all cells + paths; motion returns to OFF |

The widget owns a custom `paintEvent` that draws the pixmap letterboxed then adds the overlay: green box for the robot, per-status-coloured box for each cell (gray=PENDING, orange=IN_PROGRESS, green=DONE, red=LOST), trajectories in per-object hues, yellow dot on the current target waypoint, and magenta arrows during Push-cells mode showing robot→cell and cell→goal.

**Control ingress.** `PathFollowController` commands motion tagged `Source.ALGORITHM` through `_command_direction`: by default the **rolling recipe** (Mode A, roll axis horizontal ⊥ travel via `roll_axis_for`), switchable to legacy static pull with `path_follow.drive_mode: static` (a "Drive" combo on the Tracker tab). Magnitude and frequency come from the shared `modes.mode_a.*` knobs, read at use so mid-run Joystick-tab changes apply on the next frame; the Tracker tab shows them read-only. Joystick's `Source.JOYSTICK` still preempts via last-writer-wins if the operator grabs manual control mid-run.

**Config paths** — all mask sliders bind to `tracker.robot.*` and `tracker.cell.*`; automation knobs bind to `path_follow.*`. Per-session tuning persists via `config_loader.Config.set`.

**Image → workspace mapping.** `_image_to_world` flips image-y to the screen-up frame and applies `calibration.screen_to_world_2x2` — the same matrix the joystick uses, live-reloaded on config change. One Frame Cal run fixes manual and autonomous driving together; if the bead follows paths in the wrong direction, re-run Frame Calibration (never hand-edit the matrix). The legacy `image_rotation_deg`/`image_flip_y` knobs are gone.

**Recording (⏺ button).** Top-left of the camera view. Records the *annotated view* — displayed frame, overlays, and current ⟳ rotation, exactly as on screen — to `rec_YYYYmmdd-HHMMSS.mp4` under `recording.output_dir` (empty = `D:/Microrobots/Tracking Data` on Windows, `<repo>/Data` elsewhere). A fixed-rate grab timer at `recording.fps` (default 20) doubles as the writer fps, so files play at wall-clock speed by construction; encoding runs on a dedicated QThread (`classes/gui/recording.py`) with a bounded drop-when-behind backlog, and `MainWindow.closeEvent` finalizes a running recording before the video source closes.

**View rotation (⟳ button).** The camera view has a top-right ⟳ button that rotates the *displayed* image 90° per click (persisted as `camera.view_rotation_deg`). Display-only: the tracker, path following, and frame calibration always operate in raw camera coordinates (`classes/gui/view_transform.py` holds the pure coordinate math; clicks are un-rotated back to raw pixels). The joystick bridge composes the rotation so stick-up always means displayed-up.

**Frame calibration (Frame Cal tab).** A **recipe selector** picks what is being measured. **Rolling** (default) is the calibration that matters for driving: it fits `calibration.screen_to_world_2x2` (direction — shared by the joystick, the path follower, *and* static pull, since it is just the camera-vs-coil frame rotation) plus `screen_response_2x2` (rolling speed). **Static pull** measures Mode B and writes **only** `calibration.static_response_2x2` + `static_response_mag`; it deliberately does not touch the direction matrix, or the two recipes would overwrite each other. Measurements are kept per-recipe — the units differ (px per revolution vs px/s), so mixing them into one fit would be meaningless. `_measurement_delta` divides rolling by `freq × dur` and static by `dur` alone (a Mode B pulse has no frequency). The static run also reports the angle between its own direction fit and the saved rolling matrix: after the exact Mode B synthesis landed these should agree, so a large disagreement is a real signal (bad `Bmap`, moved coil) rather than something to average away.

The tracker's motion filter consumes both: rolling velocity is `S_roll · ŵ · freq`, static is `S_static · ŵ · (mag/mag_cal)²` — magnitude *squared* because `F ∝ ∇|B|²` and `B ∝ duty`, and the overdamped regime makes terminal velocity ∝ F. That is genuinely unlike rolling, where speed tracks frequency and is roughly magnitude-independent above threshold. Either matrix absent → that mode falls back to constant velocity.

Rolling pulses each world direction with the *rolling* recipe (Mode A, roll axis ⊥ direction; magnitude + frequency from the shared `modes.mode_a.*` knobs, hold duration panel-local), measures the tracked bead's raw-image displacement, and fits `calibration.screen_to_world_2x2` consumed by both the joystick and the path follower. The fit is the **raw inverse** — `M_sw ∝ S⁻¹` with a single overall scalar (largest column norm → 1.0). Never per-column-normalize before inverting: that silently distorts directions on skewed/anisotropic rigs (`S·D·S⁻¹·u ≠ u`); `classes/gui/test_frame_calibration_fit.py` pins this. Consumers use the matrix for **direction only** — its column magnitudes encode measured rolling distances, and roll displacement scales with freq×time, not linearly with `|B|`, so letting them modulate drive strength starves the weak screen axis (joystick: magnitude comes from stick deflection; follower: from the shared magnitude knob). All four pulses must share magnitude/freq/duration (the panel records each pulse's parameters, compensates hold-length differences, and warns on magnitude mismatch). The panel reports fit quality (column separation angle, mirror-handedness, anisotropy). Re-running calibration and pressing Apply is the intended fix whenever stick or follower directions are wrong — never hand-edit the matrix. The joystick bridge is disabled during each pulse so stick drift can't corrupt the measurement, and calibration refuses to start while path following runs.

### Neural backend for cell tracking

The robot defaults to the fast template-match tracker (`tracker.robot_tracker_backend: template`) and should stay there: VitTrack/NanoTrack are trained on data where the target is visually distinct from its surroundings, and identical spheres on a uniform background is close to their worst case — they fail the same way template matching would, but slower and less inspectably. SAM 2 has real cross-frame memory but is heavy for a closed-loop rate and gives up the correlation surface, which is the best ambiguity signal available. The learned trackers solve appearance drift; appearance drift is not the failure mode here. Note the motion prior and ambiguity detection are `ObjectTracker`-only — flipping the robot to `adaptive` silently forfeits the whole anti-swap stack. Cells (Push-cells mode only) can optionally use a **neural learned tracker** — turn on the "Use neural tracker for cells" checkbox on the Tracker tab. Implemented in `classes/gui/adaptive_tracker.py::AdaptiveTracker`; same `Detection` shape as `ObjectTracker` so `PathFollowController.cell_trackers` holds either polymorphically.

**Backend resolution** (in order — `auto` mode):

1. **SAM 2** (Segment Anything Model 2) — `ultralytics.SAM("sam2.1_t.pt")` etc. State-of-the-art click-to-segment-and-track. Requires `pip install ultralytics` (pulls torch, ~500 MB install). Weights auto-download on first use (~40 MB tiny → ~230 MB large). Uses **Apple MPS** on Mac Apple Silicon (real GPU/ANE acceleration — no more CPU-only cap!) and **CUDA** on Jetson. Size configured via `tracker.sam2_size` (`tiny | small | base | large`).
2. **VitTrack** — `cv2.TrackerVit_create`, transformer tracker, ~700 KB (full) / ~270 KB (int8) weights. Auto-downloaded from opencv_zoo (via `media.githubusercontent.com` for Git LFS resolution) to `models/` on first use. Config: `tracker.nano_weights_url_prefix`. The reliable fallback if SAM 2 deps aren't installed.
3. **NanoTrack** — `cv2.TrackerNano_create`, SiamRPN-style. opencv_zoo dropped these weights — the operator must place them in `models/` manually. Auto mode falls through if absent.
4. **Template match** — silent final fallback if no neural backend is available. The controller's `select_at` catches the `AdaptiveTracker` refusal and instantiates an `ObjectTracker` instead. Sub-pixel accurate, no download needed.
5. **MIL** (`cv2.TrackerMIL_create`) is **not** in the auto fallback chain because it's genuinely slow on multi-megapixel frames and caused UI lag. It's still reachable by explicitly setting `tracker.adaptive_backend: mil`.

**Init robustness.** `AdaptiveTracker.initialize` uses the mask to find the target's tight bounding box (via `cv2.findContours` + `boundingRect` around the blob nearest the click, padded 20%). This is critical: cv2's neural trackers learn features from whatever pixels the init bbox contains, so a loose square (mostly background) causes drift. Every frame post-init, the mask verifies the reported bbox still contains a blob comparable in area to the initial one — 8 consecutive frames of `area < 20% × init_area` = declared lost.

**SAM 2 vs cv2 backends.** SAM 2 is a whole different paradigm: it takes a *point prompt* + image and returns a segmentation mask. `_Sam2Adapter` wraps it to look like a cv2 tracker (init(bbox) → ok, new_bbox = update()). Each frame it re-prompts SAM 2 with the last known centre point, extracts the resulting mask's bounding rect, and updates the prompt point for the next tick. Simple point-propagation tracking — good for slowly-moving beads. If the operator wants full SAM 2 video-mode with cross-frame memory, that's a follow-up (needs Meta's raw `sam2` package + frame preloading; ultralytics' current API doesn't expose the video predictor cleanly).

**Fast-fail + failure cache.** Every download attempt has a 5–8 s timeout (`urllib.request.urlopen(url, timeout=…)`, not the default 60 s). Once a URL prefix has failed within a session, subsequent selections skip the download entirely and go straight to template — no re-hang on every click. `_URL_FAILURES` is a module-level set cleared only by process restart. Manual install path (as documented in the log message on failure) is: download the ONNX from opencv_zoo's `object_tracking_vittrack/` folder via a web browser and drop into `models/`.

**GPU / dnn target.** `AdaptiveTracker` probes `DNN_TARGET_CUDA_FP16 → CUDA → OPENCL_FP16 → OPENCL → CPU` at init time, adopting the first one that runs a dummy forward cleanly. Override via `tracker.adaptive_dnn_target`. On the operator's Apple Silicon Mac, cv2.dnn has no Metal target so this resolves to CPU (NanoTrack at 60+ FPS on modern CPUs). On the Jetson AGX Orin with a CUDA-enabled opencv build it should light up CUDA_FP16 automatically. Real GPU/ANE acceleration on Mac would require onnxruntime + CoreMLExecutionProvider — deliberately not built this pass; the architecture leaves room for a second backend behind the same `AdaptiveTracker` interface.

**Panel status readout** (Tracker tab, under the neural checkbox):

| State | Reads |
|---|---|
| Neural off | `running: template (fast)` |
| Neural on, no cells yet | `running: (select a cell to load)` |
| Neural on, weights + cv2 ok | `running: nano · cuda_fp16` (or CPU) |
| Neural on, weights fetch failed | `running: mil · cpu` |

Toggling the checkbox affects the *next* cell selection. Existing cells stay on whatever tracker seeded them; Clear + reselect to migrate mid-session.

## Entry points

- `main.py` — new stack (this is what operators launch).
- `main_legacy.py` — legacy stack. Rollback path only; do not evolve.

Run locally:
```
python3 main.py                 # opens serial to Arduino automatically
python3 main.py --print-only    # PrintHAL — for GUI dev without a rig
```

Building a Mac app bundle:
```
/opt/homebrew/bin/python3.10 -m PyInstaller --onedir --windowed --icon MagScopeBox.icns --name MagScope main.py
```

## Repository layout

```
main.py                          # new-stack entry point
main_legacy.py                   # legacy entry point (rollback)
config.yaml                      # gitignored, operator-local config
config_example.yaml              # committed template
calibration.json                 # legacy per-coil gains; auto-migrated
reqs.txt / reqswindows.txt       # pip requirements
README.md                        # hardware/OS setup

arduino_files/                   # firmware sources (dumb PWM writer)
uis/                             # legacy Qt Designer .ui files
imgs/                            # icons, screenshots, sample test video
action_data/                     # legacy Excel-action scripts
Data/                            # runtime output (tracking data, recordings)
old/                             # legacy / archived versions — do not edit
classes/                         # all application logic (see below)
```

## `classes/` — application code

The new stack (post-overhaul) — small, single-responsibility modules:

- `config_loader.py` — YAML config; dotted-path get/set; auto-migrate from `calibration.json`.
- `robot_model.py`  — `RobotModel` for paramagnetic / soft / hard.
- `field_solver.py` — pure-math solver + tests.
- `motion_filter.py` — pure-numpy constant-velocity Kalman filter with an
  optional commanded-velocity control input; supplies the tracker's motion
  prior. No Qt / OpenCV.
- `distractors.py` — low-rate global blob tracking of every object that
  isn't the selected robot, so their correlation peaks can be suppressed.
- `motion_controller.py` — mode state machine, phase accumulator, tick.
- `supervisor.py`   — watchdog, current budget, I²R estimate, e-stop.
- `hal.py`          — `ArduinoHAL` and `PrintHAL`; both expose `set_currents`, `read_temps`, `estop`, `close`.
- `arduino_class.py`— low-level `ArduinoHandler` (kept from legacy; now baud-parameterized).
- `gui/`            — pure-Python PyQt5 panels (no `.ui` files):
  `main_window.py`, `panels.py` (Mode A/B/Solver/Supervisor), `calibration_wizard.py`, `widgets.py`, `app.py`.

Legacy stack (only used by `main_legacy.py`) — untouched, kept as rollback:

- `gui_functions.py` — `MainWindow(QMainWindow)` (~1300 lines). Old signal wiring.
- `gui_widgets.py`   — generated from `uis/GUI.ui` via `pyuic5`. Do not hand-edit.
- `field_synth.py`   — legacy field synthesis (COIL_AXES geometry, pull-only closed form).
- `field_tabs.py`    — legacy Field Controls dock.

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

## Editing conventions (post-overhaul)

- **New GUI is pure Python — no `.ui` files, no `pyuic5`.** Edit `classes/gui/*.py` directly. `classes/gui_widgets.py` and `uis/*.ui` belong to the legacy stack only.
- Config values are the source of truth. Widgets bind to `config.yaml` paths via `LabeledDoubleSpinBox(config=cfg, config_path="solver.lambda", ...)`. Changes flow through `Config.set()` and its `on_change` listeners.
- The Arduino packet is a **7-float** schema: `[I1..I6, acoustic_freq]`. Firmware zero-fills on watchdog trip; the Python-side Supervisor does the same. Any packet-shape change must land in `arduino_files/main_3DTweezers/main_3DTweezers.ino`, `classes/arduino_class.py`, and every call site of `HAL.set_currents()` together.
- Baud rate is **500000**. Match in `config.yaml -> serial.baud` and firmware `Serial.begin(500000)`. If a rig can't hold 500000 cleanly, drop to 250000 in both.
- **Do not touch `classes/gui_functions.py`, `classes/gui_widgets.py`, `classes/field_synth.py`, `classes/field_tabs.py`, or `uis/*.ui`.** These belong to the legacy stack (`main_legacy.py`) and are frozen. Once the new stack is verified on the rig, they move to `old/` and are eventually deleted.
- `algorithm.reset()` in `algorithm_class.py` is legacy — the new stack has no equivalent yet. When the new stack integrates path following, the analog will live in the outer PID + `MotionController` command bindings.
