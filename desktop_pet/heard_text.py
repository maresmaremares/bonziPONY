"""Small overlay below the pony showing what the STT heard."""

from __future__ import annotations

from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import QApplication, QWidget

_MAX_WIDTH = 350
_PADDING = 8
_RADIUS = 8
_POINTER_SIZE = 8


class HeardText(QWidget):
    """Translucent overlay showing what the STT transcribed."""

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

        self._text = ""
        self._anchor_widget = None

        self._font = QFont("Segoe UI", 9)
        self._font.setItalic(True)
        self._font.setStyleStrategy(QFont.PreferAntialias)

        self._follow_timer = QTimer(self)
        self._follow_timer.setInterval(33)
        self._follow_timer.timeout.connect(self._follow_tick)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide_heard)

    def set_anchor_widget(self, widget) -> None:
        self._anchor_widget = widget

    def show_heard(self, text: str) -> None:
        """Show what the STT heard below the pony."""
        self._hide_timer.stop()
        self._text = text.strip()
        if not self._text:
            self.hide_heard()
            return
        self._resize_and_position()
        self.show()
        self.raise_()
        if self._anchor_widget:
            self._follow_timer.start()
        # Auto-hide after a few seconds (will be replaced by speech bubble anyway)
        self._hide_timer.start(6000)

    def hide_heard(self) -> None:
        self._hide_timer.stop()
        self._follow_timer.stop()
        self.hide()

    def _follow_tick(self) -> None:
        if not self.isVisible() or self._anchor_widget is None:
            return
        self._reposition()

    def _resize_and_position(self) -> None:
        fm = QFontMetrics(self._font)
        text_rect = fm.boundingRect(
            0, 0, _MAX_WIDTH - 2 * _PADDING, 1000,
            Qt.TextWordWrap, self._text or " ",
        )
        w = min(max(text_rect.width() + 2 * _PADDING, 60), _MAX_WIDTH)
        h = text_rect.height() + 2 * _PADDING + _POINTER_SIZE
        self.setFixedSize(int(w), int(h))
        self._reposition()

    def _reposition(self) -> None:
        if self._anchor_widget is None:
            return
        w = self._anchor_widget
        anchor_x = w.x() + w.width() // 2
        anchor_y = w.y() + w.height()

        bx = anchor_x - self.width() // 2
        by = anchor_y + 4  # small gap below sprite

        # Clamp to screen
        from PyQt5.QtCore import QPoint
        screen = QApplication.screenAt(QPoint(anchor_x, anchor_y))
        if screen is None:
            screen = QApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            bx = max(geom.left(), min(bx, geom.right() - self.width()))
            by_clamped = min(by, geom.bottom() - self.height())
            # If clamped position overlaps anchor, flip above the pony
            if by_clamped < anchor_y + 4:
                by = w.y() - self.height() - 4
            else:
                by = by_clamped

        self.move(bx, by)

    def paintEvent(self, event) -> None:
        if not self._text:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        fm = QFontMetrics(self._font)
        text_rect = fm.boundingRect(
            0, 0, _MAX_WIDTH - 2 * _PADDING, 1000,
            Qt.TextWordWrap, self._text,
        )
        bubble_w = min(max(text_rect.width() + 2 * _PADDING, 60), _MAX_WIDTH)
        bubble_h = text_rect.height() + 2 * _PADDING
        bubble_y = _POINTER_SIZE

        # Bubble background
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, bubble_y, bubble_w, bubble_h), _RADIUS, _RADIUS)
        painter.setPen(QPen(QColor(100, 100, 100, 180), 1))
        painter.setBrush(QColor(40, 40, 40, 200))
        painter.drawPath(path)

        # Pointer triangle pointing up toward pony
        ptr_path = QPainterPath()
        cx = bubble_w // 2
        ptr_path.moveTo(cx - 5, bubble_y)
        ptr_path.lineTo(cx, 1)
        ptr_path.lineTo(cx + 5, bubble_y)
        ptr_path.closeSubpath()
        painter.setBrush(QColor(40, 40, 40, 200))
        painter.setPen(QPen(QColor(100, 100, 100, 180), 1))
        painter.drawPath(ptr_path)

        # Fill seam
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(40, 40, 40, 200))
        painter.drawRect(int(cx) - 4, int(bubble_y), 8, 3)

        # Text
        painter.setPen(QColor(220, 220, 220))
        painter.setFont(self._font)
        painter.drawText(
            QRectF(_PADDING, bubble_y + _PADDING,
                   bubble_w - 2 * _PADDING, bubble_h - 2 * _PADDING),
            Qt.TextWordWrap, self._text,
        )

        painter.end()
