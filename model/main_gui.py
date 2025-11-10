from gui import NeuroVoiceWindow
from PyQt6.QtWidgets import QApplication
import sys
from predict_fixed import load_default_config

def main():
    app = QApplication(sys.argv)
    config = load_default_config()
    w = NeuroVoiceWindow(config)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()