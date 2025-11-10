from UI.gui import NeuroVoiceWindow
from PyQt6.QtWidgets import QApplication
import sys

def main():
    app = QApplication(sys.argv)
    w = NeuroVoiceWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()