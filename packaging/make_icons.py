"""Draw the app icon (packaging/linux/songpolish.png) and convert it for Windows/macOS."""

import sys
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QGuiApplication,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)

HERE = Path(__file__).parent
SIZE = 1024


def draw(path: Path) -> None:
    img = QImage(SIZE, SIZE, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    grad = QLinearGradient(0, 0, SIZE, SIZE)
    grad.setColorAt(0, QColor("#f59e0b"))
    grad.setColorAt(1, QColor("#b45309"))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(grad))
    m = SIZE * 0.06
    p.drawRoundedRect(QRectF(m, m, SIZE - 2 * m, SIZE - 2 * m), SIZE * 0.2, SIZE * 0.2)
    # Guitar soundhole with strings, and a sparkle: "polish".
    c = QPointF(SIZE * 0.45, SIZE * 0.55)
    p.setBrush(QColor("#3b1d06"))
    p.drawEllipse(c, SIZE * 0.24, SIZE * 0.24)
    p.setPen(QPen(QColor("#fde68a"), SIZE * 0.022))
    for i in range(-2, 4):
        x = c.x() + i * SIZE * 0.06 - SIZE * 0.03
        p.drawLine(QPointF(x, SIZE * 0.2), QPointF(x, SIZE * 0.88))
    star = QPainterPath()
    cx, cy, r = SIZE * 0.74, SIZE * 0.27, SIZE * 0.15
    star.moveTo(cx, cy - r)
    for x, y in ((0.18, -0.18), (1, 0), (0.18, 0.18), (0, 1), (-0.18, 0.18), (-1, 0),
                 (-0.18, -0.18), (0, -1)):  # fmt: skip
        star.lineTo(cx + x * r, cy + y * r)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#fffbeb"))
    p.drawPath(star)
    p.end()
    img.save(str(path))


def main() -> int:
    _app = QGuiApplication(sys.argv)
    png = HERE / "linux" / "songpolish.png"
    draw(png)
    im = Image.open(png)
    im.resize((512, 512), Image.LANCZOS).save(png)
    im.save(HERE / "windows" / "songpolish.ico", sizes=[(s, s) for s in (16, 32, 48, 64, 128, 256)])
    im.save(HERE / "macos" / "songpolish.icns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
