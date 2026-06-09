import sys
from PySide6.QtCore import Qt, QPoint, QUrl, QEvent, QTimer
from PySide6.QtGui import QIcon, QAction
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QSystemTrayIcon, QMenu, QStyle
from PySide6.QtWebEngineWidgets import QWebEngineView

ICON_PATH = "icon_inv.png"
BACKEND_URL = "http://127.0.0.1:8000/tray"

class ModernTrayApp(QWidget):
    """Borderless, frameless web view container matching modern tray tools."""
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(450, 600)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.webview = QWebEngineView()
        self.webview.page().setBackgroundColor(Qt.transparent)
        self.webview.loadFinished.connect(self.on_load_finished)
        self.load_backend()
        layout.addWidget(self.webview)
        
        # Variable to track mouse position for window dragging
        self._drag_pos = None

    def load_backend(self):
        self.webview.setUrl(QUrl(BACKEND_URL))

    def on_load_finished(self, ok):
        if not ok:
            # Retry mechanism if backend is slow to boot
            QTimer.singleShot(1000, self.load_backend)

    def event(self, e):
        """Automatically hide widget on focus loss."""
        if e.type() == QEvent.Type.WindowDeactivate:
            self.hide()
        return super().event(e)

    # --- Methods to make the frameless window movable ---
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Capture the initial click position
            self._drag_pos = event.globalPosition().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None:
            # Calculate how far the mouse has moved
            delta = event.globalPosition().toPoint() - self._drag_pos
            self.move(self.pos() + delta)
            # Update the drag position
            self._drag_pos = event.globalPosition().toPoint()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Stop dragging when the mouse is released
            self._drag_pos = None
        super().mouseReleaseEvent(event)


class TrayController:
    """Manages the system tray icon, context menu, and visibility of the main app."""
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)
        self.window = ModernTrayApp()
        self._setup_tray_icon()

    def _setup_tray_icon(self):
        default_icon = self.app.style().standardIcon(QStyle.SP_ComputerIcon)
        self.tray_icon = QSystemTrayIcon(default_icon, self.app)
        
        menu = QMenu()
        toggle_action = QAction("Show/Hide Interface", self.app)
        toggle_action.triggered.connect(self.toggle_window)
        menu.addAction(toggle_action)
        menu.addSeparator()
        
        quit_action = QAction("Quit Vision Agent", self.app)
        quit_action.triggered.connect(self.app.quit)
        menu.addAction(quit_action)
        
        self.tray_icon.setContextMenu(menu)
        self.tray_icon.activated.connect(self.on_tray_activated)
        self.tray_icon.show()

    def on_tray_activated(self, reason):
        valid_triggers = (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick
        )
        if reason in valid_triggers:
            self.toggle_window()

    def toggle_window(self):
        if self.window.isVisible():
            self.window.hide()
            return
            
        screen_rect = QApplication.primaryScreen().availableGeometry()
        tray_rect = self.tray_icon.geometry()
        
        if tray_rect.isValid() and tray_rect.width() > 0:
            x = tray_rect.center().x() - (self.window.width() // 2)
            y = (tray_rect.top() - self.window.height() - 10
                 if tray_rect.top() > screen_rect.center().y()
                 else tray_rect.bottom() + 10)
        else:
            x = screen_rect.right() - self.window.width() - 10
            y = screen_rect.top() + 10
            
        x = max(screen_rect.left(), min(x, screen_rect.right() - self.window.width()))
        
        self.window.move(QPoint(x, y))
        self.window.showNormal()
        self.window.activateWindow()
        self.window.raise_()

    def run(self):
        sys.exit(self.app.exec())

if __name__ == "__main__":
    controller = TrayController()
    controller.run()