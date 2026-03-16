"""Floating countdown timer that follows the pony during enforcement mode."""

from __future__ import annotations

import logging

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QApplication, QWidget

logger = logging.getLogger(__name__)


class CountdownTimer(QWidget):
    """Small floating countdown that appears near the pony."""

    def __init__(self) -> None:
        super().__init__(None)
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowTransparentForInput
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setFixedSize(100, 36)

        self._remaining_s: int = 0
        self._anchor_widget = None

        self._font = QFont("Consolas", 14, QFont.Bold)
        self._font.setStyleStrategy(QFont.PreferAntialias)

        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(1000)
        self._tick_timer.timeout.connect(self._tick)

        self._follow_timer = QTimer(self)
        self._follow_timer.setInterval(33)  # ~30fps
        self._follow_timer.timeout.connect(self._follow_tick)

    def set_anchor_widget(self, widget) -> None:
        self._anchor_widget = widget

    def start_countdown(self, seconds: int) -> None:
        """Show and start the countdown."""
        self._remaining_s = max(0, seconds)
        self._reposition()
        self.show()
        self.raise_()
        self._tick_timer.start()
        self._follow_timer.start()
        self.update()
        logger.info("Countdown started: %ds", seconds)

    def stop_countdown(self) -> None:
        """Hide the countdown."""
        self._tick_timer.stop()
        self._follow_timer.stop()
        self.hide()

    def _tick(self) -> None:
        if self._remaining_s > 0:
            self._remaining_s -= 1
            self.update()
        else:
            self._tick_timer.stop()
            # Keep showing 0:00 — enforcement handles hiding

    def _follow_tick(self) -> None:
        if self.isVisible() and self._anchor_widget:
            self._reposition()

    def _reposition(self) -> None:
        if not self._anchor_widget:
            return
        w = self._anchor_widget
        # Position below and to the right of the pony
        px = w.x() + w.width() // 2 - self.width() // 2
        py = w.y() + w.height() + 4

        from PyQt5.QtCore import QPoint
        screen = QApplication.screenAt(QPoint(w.x() + w.width() // 2, w.y()))
        if screen is None:
            screen = QApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            px = max(geom.left(), min(px, geom.right() - self.width()))
            py = min(py, geom.bottom() - self.height())
        self.move(px, py)

    def paintEvent(self, event) -> None:
        if self._remaining_s < 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background pill
        bg = QColor(30, 30, 30, 200)
        painter.setPen(Qt.NoPen)
        painter.setBrush(bg)
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 10, 10)

        # Border
        painter.setPen(QPen(QColor(255, 255, 255, 80), 1.5))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(1, 1, self.width() - 2, self.height() - 2, 9, 9)

        # Format time
        mins = self._remaining_s // 60
        secs = self._remaining_s % 60
        if mins >= 60:
            hrs = mins // 60
            mins = mins % 60
            time_str = f"{hrs}:{mins:02d}:{secs:02d}"
        else:
            time_str = f"{mins}:{secs:02d}"

        # Pick color based on time remaining
        if self._remaining_s <= 30:
            text_color = QColor(255, 80, 80)  # red when almost done
        elif self._remaining_s <= 120:
            text_color = QColor(255, 200, 60)  # yellow
        else:
            text_color = QColor(255, 255, 255)  # white

        painter.setPen(text_color)
        painter.setFont(self._font)
        painter.drawText(self.rect(), Qt.AlignCenter, time_str)

        painter.end()
