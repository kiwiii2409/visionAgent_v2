import sys
import os
from PySide6.QtCore import Qt, QPoint, QUrl, QEvent, QTimer
from PySide6.QtGui import QIcon, QAction, QPixmap, QPainter, QColor
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QSystemTrayIcon, QMenu, QStyle
from PySide6.QtWebEngineWidgets import QWebEngineView

ICON_PATH = "icon_robot.png"
BACKEND_URL = "http://127.0.0.1:8000/tray"

def get_icon_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), ICON_PATH)

def get_base_icon(app):
    path = get_icon_path()
    if os.path.exists(path):
        return QIcon(path)
    return app.style().standardIcon(QStyle.SP_ComputerIcon)

def get_active_icon():
    path = get_icon_path()
    if not os.path.exists(path):
        return None
    
    pixmap = QPixmap(path)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    
    # Draw red dot in bottom right
    painter.setBrush(QColor(239, 68, 68)) 
    painter.setPen(Qt.PenStyle.NoPen)
    
    size = pixmap.width()
    dot_size = int(size * 0.35) # 35% of icon size
    # Draw at bottom left (x=0, y=size - dot_size)
    painter.drawEllipse(size-dot_size, size - dot_size, dot_size, dot_size)
    painter.end()
    
    return QIcon(pixmap)


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
        
        self._drag_pos = None

    def load_backend(self):
        self.webview.setUrl(QUrl(BACKEND_URL))

    def on_load_finished(self, ok):
        if not ok:
            QTimer.singleShot(1000, self.load_backend)

    def event(self, e):
        if e.type() == QEvent.Type.WindowDeactivate:
            self.hide()
        return super().event(e)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None:
            delta = event.globalPosition().toPoint() - self._drag_pos
            self.move(self.pos() + delta)
            self._drag_pos = event.globalPosition().toPoint()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = None
        super().mouseReleaseEvent(event)


class TrayController:
    """Manages the system tray icon, context menu, and visibility of the main app."""
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)
        self.window = ModernTrayApp()
        
        # Listen to title changes to toggle the active tray icon state
        self.window.webview.titleChanged.connect(self.on_title_changed)
        
        self._setup_tray_icon()

    def _setup_tray_icon(self):
        self.tray_icon = QSystemTrayIcon(get_base_icon(self.app), self.app)
        
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

    def on_title_changed(self, title):
        if "Active" in title:
            active_icon = get_active_icon()
            if active_icon:
                self.tray_icon.setIcon(active_icon)
        else:
            self.tray_icon.setIcon(get_base_icon(self.app))

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