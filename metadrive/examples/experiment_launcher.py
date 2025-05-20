import sys, os, subprocess
from PyQt5 import QtWidgets, QtCore, QtGui

class Launcher(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Experiment Launcher")
        self.setFixedSize(460, 300)

        # Set modern font
        font = QtGui.QFont("Segoe UI", 11)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        QtWidgets.QApplication.setFont(font)

        # Custom dark theme
        QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(24, 24, 24))
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(240, 240, 240))
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 30, 30))
        palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(35, 35, 35))
        palette.setColor(QtGui.QPalette.Text, QtGui.QColor(220, 220, 220))
        palette.setColor(QtGui.QPalette.Button, QtGui.QColor(36, 36, 36))
        palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(230, 230, 230))
        palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(60, 120, 200))
        palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(255, 255, 255))
        QtWidgets.QApplication.setPalette(palette)

        self.setStyleSheet("""
            QPushButton {
                padding: 8px 20px;
                border-radius: 6px;
                background-color: #444;
                color: #eee;
                font-size: 14px;
                letter-spacing: 0.4px;
            }
            QPushButton:hover {
                background-color: #525252;
            }
            QLabel {
                font-size: 14px;
                color: #dddddd;
            }
            QLineEdit, QComboBox {
                padding: 6px;
                border-radius: 6px;
                border: 1px solid #555;
                background-color: #2a2a2a;
                color: #f1f1f1;
                font-size: 14px;
            }
            QComboBox QAbstractItemView {
                background-color: #2a2a2a;
                selection-background-color: #444;
                color: #f1f1f1;
            }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        title = QtWidgets.QLabel("Experiment Launcher")
        title.setStyleSheet("font-size: 18px; font-weight: 600; letter-spacing: 0.5px;")
        layout.addWidget(title, alignment=QtCore.Qt.AlignHCenter)

        form_layout = QtWidgets.QFormLayout()
        form_layout.setLabelAlignment(QtCore.Qt.AlignRight)
        form_layout.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        form_layout.setHorizontalSpacing(12)
        form_layout.setVerticalSpacing(20)

        self.id_input = QtWidgets.QLineEdit()
        self.id_input.setPlaceholderText("e.g. 001")
        self.id_input.setFixedHeight(32)
        form_layout.addRow("Participant ID:", self.id_input)

        self.trial_combo = QtWidgets.QComboBox()
        self.trial_combo.addItems(["Practice", "Experiment"])
        self.trial_combo.setFixedHeight(32)
        form_layout.addRow("Trial Type:", self.trial_combo)

        self.cond_combo = QtWidgets.QComboBox()
        self.cond_combo.addItems(["None", "Visual", "Audio", "Haptic"])
        self.cond_combo.setFixedHeight(32)
        form_layout.addRow("Condition:", self.cond_combo)

        layout.addLayout(form_layout)

        self.launch_btn = QtWidgets.QPushButton("Start Trial")
        self.launch_btn.clicked.connect(self.launch)
        layout.addWidget(self.launch_btn, alignment=QtCore.Qt.AlignCenter)

    def launch(self):
        pid = self.id_input.text().strip()
        trial = self.trial_combo.currentText().lower()
        cond = self.cond_combo.currentText().lower()

        if not pid:
            self.id_input.setStyleSheet("border: 1px solid red;")
            QtWidgets.QMessageBox.warning(self, "Missing ID", "Please enter a participant number")
            return
        else:
            self.id_input.setStyleSheet("border: 1px solid #555;")

        env = os.environ.copy()
        env["EXPERIMENTER_ID"] = pid
        env["CONDITION"] = cond
        env["MAP_BLOCKS"] = "12" if trial == "practice" else "8"
        env["LOG_DATA"] = "0" if trial == "practice" else "1"

        script_path = "metadrive/examples/drive_in_safe_with_math.py"
        subprocess.Popen([sys.executable, script_path], env=env)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Launcher()
    window.show()
    sys.exit(app.exec_())