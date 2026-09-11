"""Small reusable widgets for the new control GUI.

All widgets are pure Python — no .ui files. They plug into ``config.yaml``
via dotted paths so edits push back to the config immediately.
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt5 import QtCore, QtWidgets


class LabeledDoubleSpinBox(QtWidgets.QWidget):
    """A `label — spinbox` pair. Emits ``valueChanged(float)``.

    If ``config`` and ``config_path`` are provided, the initial value is
    read from that path and edits are written back and pushed through the
    Config's on_change listeners.
    """

    valueChanged = QtCore.pyqtSignal(float)

    def __init__(self,
                 label: str,
                 minimum: float = -1.0,
                 maximum: float = 1.0,
                 step: float = 0.01,
                 decimals: int = 3,
                 initial: float = 0.0,
                 suffix: str = "",
                 config=None,
                 config_path: Optional[str] = None,
                 parent=None):
        super().__init__(parent)
        self.config = config
        self.config_path = config_path
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.label = QtWidgets.QLabel(label)
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setSingleStep(step)
        self.spin.setDecimals(decimals)
        if suffix:
            self.spin.setSuffix(f" {suffix}")
        if config is not None and config_path is not None:
            v = config.get(config_path, initial)
            self.spin.setValue(float(v))
        else:
            self.spin.setValue(initial)
        self.spin.valueChanged.connect(self._on_changed)
        row.addWidget(self.label)
        row.addStretch(1)
        row.addWidget(self.spin)

    def _on_changed(self, v: float) -> None:
        if self.config is not None and self.config_path is not None:
            self.config.set(self.config_path, float(v))
        self.valueChanged.emit(v)

    def value(self) -> float:
        return float(self.spin.value())

    def setValue(self, v: float) -> None:
        self.spin.setValue(float(v))


class Vec3Editor(QtWidgets.QWidget):
    """Three labeled spinboxes for a 3-vector. Emits ``valueChanged(list)``."""

    valueChanged = QtCore.pyqtSignal(list)

    def __init__(self,
                 title: str,
                 minimum: float = -1.0,
                 maximum: float = 1.0,
                 step: float = 0.05,
                 decimals: int = 3,
                 initial=(0.0, 0.0, 0.0),
                 suffix: str = "",
                 parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if title:
            layout.addWidget(QtWidgets.QLabel(title))
        row = QtWidgets.QHBoxLayout()
        self.spins = []
        for axis, init in zip(("x", "y", "z"), initial):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setPrefix(f"{axis}: ")
            spin.setRange(minimum, maximum)
            spin.setSingleStep(step)
            spin.setDecimals(decimals)
            if suffix:
                spin.setSuffix(f" {suffix}")
            spin.setValue(float(init))
            spin.valueChanged.connect(self._emit)
            row.addWidget(spin)
            self.spins.append(spin)
        layout.addLayout(row)

    def _emit(self, _v: float) -> None:
        self.valueChanged.emit([float(s.value()) for s in self.spins])

    def value(self) -> list:
        return [float(s.value()) for s in self.spins]

    def setValue(self, v) -> None:
        for spin, x in zip(self.spins, v):
            spin.blockSignals(True)
            spin.setValue(float(x))
            spin.blockSignals(False)
        self._emit(0.0)


class CoilBar(QtWidgets.QWidget):
    """Six horizontal bars, one per coil, showing current duty as a filled
    rectangle. Used by the Supervisor panel for live feedback."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self.bars = []
        self.labels = []
        for k in range(6):
            row = QtWidgets.QHBoxLayout()
            lab = QtWidgets.QLabel(f"C{k + 1}")
            lab.setFixedWidth(28)
            row.addWidget(lab)
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat("%p%")
            bar.setFixedHeight(16)
            row.addWidget(bar)
            layout.addLayout(row)
            self.bars.append(bar)
            self.labels.append(lab)

    def update_from(self, currents) -> None:
        for k in range(6):
            pct = max(0, min(100, int(round(abs(float(currents[k])) * 100))))
            self.bars[k].setValue(pct)


def make_group(title: str, inner: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
    box = QtWidgets.QGroupBox(title)
    layout = QtWidgets.QVBoxLayout(box)
    layout.setContentsMargins(6, 12, 6, 6)
    layout.addWidget(inner)
    return box


def scroll_wrap(inner: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
    """Wrap a widget so the dock stays usable even when the window is short."""
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(inner)
    return scroll
