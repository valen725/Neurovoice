import os
import librosa
import numpy as np
import soundfile as sf
from io import BytesIO

from audio_io import Recorder, Player, save_wav_16k

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QHBoxLayout, QCheckBox, QSlider, QGroupBox,
    QMessageBox, QSplitter, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
import matplotlib.pyplot as plt

from preprocess import preprocess_wav_for_model, mel_spectrogram_db
from inference import predict_from_file_or_array
from gradcam import gradcam_overlay_placeholder


DATA_DIR = os.path.join("data", "recordings")
os.makedirs(DATA_DIR, exist_ok=True)


class RecordThread(QThread):
    finished = pyqtSignal(np.ndarray, int)  # audio, samplerate

    def __init__(self, seconds: int = 3, samplerate: int = 44100, parent=None):
        super().__init__(parent)
        self.seconds = seconds
        self.samplerate = samplerate
        self._rec = Recorder(samplerate=self.samplerate)

    def run(self):
        audio = self._rec.record(self.seconds)
        self.finished.emit(audio, self.samplerate)


class NeuroVoiceWindow(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.setWindowTitle("NeuroVoice — Parkinson Voice Screening (Demo)")
        self.resize(1280, 800)

        # Estado
        self.raw_audio = None
        self.raw_sr = None
        self.last_saved_path = None
        self.config = config

        # tema: False = claro (por defecto), True = oscuro
        self.dark_mode = False

        # almacenar últimos datos de gráficas para re-dibujar al cambiar tema
        self.last_wave_data = None      # (y, sr, title)
        self.last_spec_data = None      # (mel_db, overlay)
        self.spec_colorbar = None       # barra de color del espectrograma

        # Colores actuales de ejes (se llenan desde _style_*_axes)
        self._wave_axis_color = "#3a3f66"
        self._spec_axis_color = "#3a3f66"

        # Ruta al mejor modelo disponible (igual que predict_fixed.py)
        from inference import find_best_model
        self.model_path = find_best_model()
        if not self.model_path:
            QMessageBox.critical(self, "Modelo no encontrado", "No se encontró un modelo entrenado. Entrena uno primero.")

        # UI
        self._build_ui()
        self._apply_styles()
        self._init_empty_plots()
        self._update_theme_button_text()

    def _build_ui(self):
        # ----------- Splitter principal (controles | gráficas) -----------
        main_split = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(main_split)

        # ----------- Panel de controles (izquierda) -----------
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)

        # Grupo: Record / Load
        g_rec = QGroupBox("Entrada de audio")
        l_rec = QVBoxLayout(g_rec)

        self.seconds = QSlider(Qt.Orientation.Horizontal)
        self.seconds.setMinimum(1)
        self.seconds.setMaximum(10)
        self.seconds.setValue(3)
        self.seconds_lbl = QLabel("Duración: 3 s")
        self.seconds.valueChanged.connect(lambda v: self.seconds_lbl.setText(f"Duración: {v} s"))

        self.btn_record = QPushButton("● Grabar audio")
        self.btn_record.clicked.connect(self.on_record)

        self.btn_play = QPushButton("▶ Reproducir audio")
        self.btn_play.clicked.connect(self.on_play_raw)

        self.btn_load = QPushButton("📂 Cargar audio (WAV)")
        self.btn_load.clicked.connect(self.on_load_wav)

        self.btn_save = QPushButton("💾 Guardar audio (WAV)")
        self.btn_save.clicked.connect(self.on_save)

        l_rec.addWidget(self.seconds_lbl)
        l_rec.addWidget(self.seconds)
        l_rec.addWidget(self.btn_record)
        l_rec.addWidget(self.btn_play)
        l_rec.addWidget(self.btn_load)
        l_rec.addWidget(self.btn_save)

        # Grupo: Análisis
        g_inf = QGroupBox("Análisis de voz")
        l_inf = QVBoxLayout(g_inf)

        self.threshold_parkinson = 0.3

        self.robust_checkbox = QCheckBox("Modo robusto (promediar)")
        self.robust_checkbox.setChecked(False)

        self.btn_predict = QPushButton("🔎 Analizar y Predecir")
        self.btn_predict.setObjectName("AccentButton")
        self.btn_predict.clicked.connect(self.on_predict)

        self.prob_lbl = QLabel("Probabilidad (PD): —")
        self.prob_lbl.setObjectName("ResultLabel")
        self.prob_lbl.setWordWrap(True)

        self.btn_export_pdf = QPushButton("📄 Exportar informe PDF")
        self.btn_export_pdf.clicked.connect(self.on_export_pdf)
        self.btn_export_pdf.setEnabled(False)

        self.btn_gradcam = QPushButton("🔥 Grad-CAM overlay")
        self.btn_gradcam.clicked.connect(self.on_gradcam)

        # l_inf.addWidget(self.robust_checkbox)  # si lo quieres visible
        l_inf.addWidget(self.btn_predict)
        l_inf.addWidget(self.prob_lbl)
        l_inf.addWidget(self.btn_export_pdf)
        l_inf.addWidget(self.btn_gradcam)

        left_layout.addWidget(g_rec)
        left_layout.addWidget(g_inf)

        # Botón de tema (claro/oscuro)
        self.theme_btn = QPushButton()
        self.theme_btn.clicked.connect(self.toggle_theme)
        left_layout.addWidget(self.theme_btn)

        left_layout.addStretch(1)

        # ----------- Panel de gráficas (derecha, con splitter vertical) -----------
        right_split = QSplitter(Qt.Orientation.Vertical)

        # Waveform panel
        wave_panel = QWidget()
        wave_layout = QVBoxLayout(wave_panel)
        wave_layout.setContentsMargins(6, 6, 6, 6)
        wave_layout.setSpacing(6)

        wave_title = QLabel("Señal de audio")
        wave_title.setObjectName("SectionTitle")

        self.fig_wave, self.ax_wave = plt.subplots(figsize=(8, 3), dpi=120)
        self.canvas_wave = FigureCanvas(self.fig_wave)
        self.toolbar_wave = NavigationToolbar(self.canvas_wave, self)
        self.canvas_wave.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        wave_layout.addWidget(wave_title)
        wave_layout.addWidget(self.toolbar_wave)
        wave_layout.addWidget(self.canvas_wave)

        # Mel-spectrogram panel
        spec_panel = QWidget()
        spec_layout = QVBoxLayout(spec_panel)
        spec_layout.setContentsMargins(6, 6, 6, 6)
        spec_layout.setSpacing(6)

        spec_title = QLabel("Espectrograma Mel (dB)")
        spec_title.setObjectName("SectionTitle")

        self.fig_spec, self.ax_spec = plt.subplots(figsize=(8, 4), dpi=120)
        self.canvas_spec = FigureCanvas(self.fig_spec)
        self.toolbar_spec = NavigationToolbar(self.canvas_spec, self)
        self.canvas_spec.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        spec_layout.addWidget(spec_title)
        spec_layout.addWidget(self.toolbar_spec)
        spec_layout.addWidget(self.canvas_spec)

        # Añadir paneles al splitter vertical
        right_split.addWidget(wave_panel)
        right_split.addWidget(spec_panel)
        right_split.setSizes([400, 500])

        # Añadir paneles al splitter principal
        main_split.addWidget(left_panel)
        main_split.addWidget(right_split)
        main_split.setSizes([420, 860])

    # ---------- Tema ----------

    def _update_theme_button_text(self):
        if self.dark_mode:
            self.theme_btn.setText("☀️ Cambiar a modo claro")
        else:
            self.theme_btn.setText("🌙 Cambiar a modo oscuro")

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self._apply_styles()
        self._update_theme_button_text()

        # Re-dibujar gráficas según el tema (solo colores de ejes, no cmap)
        if self.last_wave_data is not None:
            y, sr, title = self.last_wave_data
            self._plot_wave(y, sr, title)
        else:
            self._init_empty_wave_plot()

        if self.last_spec_data is not None:
            mel_db, overlay = self.last_spec_data
            self._plot_spec(mel_db, overlay)
        else:
            self._init_empty_spec_plot()

    def _apply_styles(self):
        if self.dark_mode:
            # ======== MODO OSCURO ========
            self.setStyleSheet("""
            QWidget {
                background-color: #0b0f19;
                color: #e5e9ff;
                font-family: 'Segoe UI', 'Inter', 'Arial';
                font-size: 15px;
            }

            QGroupBox {
                border: 1px solid #2c355d;
                border-radius: 12px;
                margin-top: 10px;
                padding: 12px;
                font-weight: 600;
                background-color: #12182b;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                color: #00c2ff;
                font-size: 16px;
            }

            QLabel {
                padding: 2px 0;
            }

            QLabel#SectionTitle {
                font-size: 17px;
                font-weight: 700;
                color: #00c2ff;
                padding: 2px 0 4px 2px;
            }

            QLabel#ResultLabel {
                background-color: rgba(0, 0, 0, 0.25);
                border: 1px solid #00c2ff;
                border-radius: 12px;
                padding: 10px;
                font-size: 14px;
            }

            QPushButton {
                background-color: #1a223a;
                color: #e5e9ff;
                border: 1px solid #333f70;
                border-radius: 10px;
                padding: 10px 14px;
                font-weight: 600;
                min-height: 42px;
                font-size: 15px;
            }
            QPushButton:hover {
                background-color: #232f55;
                border-color: #00c2ff;
            }
            QPushButton:pressed {
                background-color: #171e33;
                border-color: #ff0080;
            }
            QPushButton:disabled {
                color: #777a9f;
                border-color: #262b49;
                background-color: #14182a;
            }

            QPushButton#AccentButton {
                background-color: #ff0080;
                border: 1px solid #ff4da6;
                color: white;
                font-size: 17px;
                min-height: 48px;
            }
            QPushButton#AccentButton:hover {
                background-color: #ff4da6;
                border-color: #ffffff;
            }
            QPushButton#AccentButton:pressed {
                background-color: #cc0066;
            }

            QCheckBox {
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                border-radius: 4px;
                border: 1px solid #555b88;
                background-color: #151a2d;
            }
            QCheckBox::indicator:checked {
                border-radius: 4px;
                border: 1px solid #00f5a0;
                background-color: #00c2ff;
            }

            QSlider::groove:horizontal {
                background: #2e365a;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #00f5a0;
                width: 20px;
                height: 20px;
                margin: -6px 0;
                border-radius: 10px;
                border: 1px solid #00c2ff;
            }
            QSlider::handle:horizontal:hover {
                background: #00c2ff;
            }

            QToolBar {
                background: transparent;
                border: none;
                padding: 0;
                margin: 0;
            }
            QToolBar QToolButton {
                background-color: #1a223a;
                border: 1px solid #333f70;
                border-radius: 8px;
                padding: 5px;
                margin-right: 4px;
            }
            QToolBar QToolButton:hover {
                background-color: #232f55;
                border-color: #00f5a0;
            }
            """)
        else:
            # ======== MODO CLARO ========
            self.setStyleSheet("""
            QWidget {
                background-color: #f5f7fb;
                color: #1f2544;
                font-family: 'Segoe UI', 'Inter', 'Arial';
                font-size: 15px;
            }

            QGroupBox {
                border: 1px solid #d0d7f0;
                border-radius: 12px;
                margin-top: 10px;
                padding: 12px;
                font-weight: 600;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                color: #2f6fed;
                font-size: 16px;
            }

            QLabel {
                padding: 2px 0;
            }

            QLabel#SectionTitle {
                font-size: 17px;
                font-weight: 700;
                color: #2f6fed;
                padding: 2px 0 4px 2px;
            }

            QLabel#ResultLabel {
                background-color: #eef2ff;
                border: 1px solid #c2cdfa;
                border-radius: 12px;
                padding: 10px;
                font-size: 14px;
            }

            QPushButton {
                background-color: #ffffff;
                color: #1f2544;
                border: 1px solid #c7cee8;
                border-radius: 10px;
                padding: 10px 14px;
                font-weight: 600;
                min-height: 42px;
                font-size: 15px;
            }
            QPushButton:hover {
                background-color: #e4ebff;
                border-color: #2f6fed;
            }
            QPushButton:pressed {
                background-color: #d2ddff;
                border-color: #1c4fd1;
            }
            QPushButton:disabled {
                color: #a0a7c0;
                border-color: #dde2f4;
                background-color: #f1f3fb;
            }

            QPushButton#AccentButton {
                background-color: #2f6fed;
                border: 1px solid #1c4fd1;
                color: white;
                font-size: 17px;
                min-height: 48px;
            }
            QPushButton#AccentButton:hover {
                background-color: #4b82ff;
                border-color: #102f8a;
            }
            QPushButton#AccentButton:pressed {
                background-color: #1c4fd1;
            }

            QCheckBox {
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                border-radius: 4px;
                border: 1px solid #b7bfdc;
                background-color: #ffffff;
            }
            QCheckBox::indicator:checked {
                border-radius: 4px;
                border: 1px solid #2f6fed;
                background-color: #4b82ff;
            }

            QSlider::groove:horizontal {
                background: #dde2f4;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2f6fed;
                width: 20px;
                height: 20px;
                margin: -6px 0;
                border-radius: 10px;
                border: 1px solid #1c4fd1;
            }
            QSlider::handle:horizontal:hover {
                background: #4b82ff;
            }

            QToolBar {
                background: transparent;
                border: none;
                padding: 0;
                margin: 0;
            }
            QToolBar QToolButton {
                background-color: #dde2f4;
                border: 1px solid #c7cee8;
                border-radius: 8px;
                padding: 5px;
                margin-right: 4px;
            }
            QToolBar QToolButton:hover {
                background-color: #c7d2ff;
                border-color: #2f6fed;
            }
            """)

    # -------- Helpers de estilo para gráficas --------

    def _style_wave_axes(self):
        if self.dark_mode:
            self.fig_wave.patch.set_facecolor("#0b0f19")
            self.ax_wave.set_facecolor("#111522")
            axis_color = "#f5f7ff"
            spine_color = "#848ac6"
            grid_color = "#2e365a"
        else:
            self.fig_wave.patch.set_facecolor("#f5f7fb")
            self.ax_wave.set_facecolor("#ffffff")
            axis_color = "#20243c"  # más oscuro en modo claro
            spine_color = "#b7bfdc"
            grid_color = "#dde2f4"

        self._wave_axis_color = axis_color

        for spine in self.ax_wave.spines.values():
            spine.set_color(spine_color)
        self.ax_wave.tick_params(colors=axis_color, labelsize=9)
        self.ax_wave.grid(alpha=0.4, color=grid_color)

    def _style_spec_axes(self):
        if self.dark_mode:
            self.fig_spec.patch.set_facecolor("#0b0f19")
            self.ax_spec.set_facecolor("#111522")
            axis_color = "#f5f7ff"
            spine_color = "#848ac6"
        else:
            self.fig_spec.patch.set_facecolor("#f5f7fb")
            self.ax_spec.set_facecolor("#ffffff")
            axis_color = "#20243c"  # más oscuro en modo claro
            spine_color = "#b7bfdc"

        self._spec_axis_color = axis_color

        for spine in self.ax_spec.spines.values():
            spine.set_color(spine_color)
        self.ax_spec.tick_params(colors=axis_color, labelsize=9)

    def _restyle_colorbar(self):
        if self.spec_colorbar is None:
            return
        if self.dark_mode:
            axis_color = "#f5f7ff"
            spine_color = "#848ac6"
        else:
            axis_color = "#20243c"
            spine_color = "#b7bfdc"
        self.spec_colorbar.outline.set_edgecolor(spine_color)
        self.spec_colorbar.ax.tick_params(colors=axis_color, labelsize=8)

    # -------- Estado inicial de las gráficas --------

    def _init_empty_wave_plot(self):
        self.last_wave_data = None
        self.ax_wave.clear()
        self._style_wave_axes()
        self.ax_wave.set_title("Señal de audio", fontsize=11, color=self._wave_axis_color)
        self.ax_wave.set_xlabel("Time [s]", fontsize=10, color=self._wave_axis_color)
        self.ax_wave.set_ylabel("Amplitude", fontsize=10, color=self._wave_axis_color)
        self.ax_wave.text(
            0.5, 0.5,
            "Sin audio cargado o grabado",
            transform=self.ax_wave.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            alpha=0.7,
            color=self._wave_axis_color
        )
        # <-- para que no se corte el xlabel
        self.fig_wave.tight_layout()
        self.canvas_wave.draw()

    def _init_empty_spec_plot(self):
        self.last_spec_data = None
        self.ax_spec.clear()
        self._style_spec_axes()
        self.ax_spec.set_title("Espectrograma Mel (dB)", fontsize=11, color=self._spec_axis_color)
        self.ax_spec.set_xlabel("Frames", fontsize=10, color=self._spec_axis_color)
        self.ax_spec.set_ylabel("Mel bins", fontsize=10, color=self._spec_axis_color)
        self.ax_spec.text(
            0.5, 0.5,
            "El espectrograma se mostrará tras el análisis",
            transform=self.ax_spec.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            alpha=0.7,
            color=self._spec_axis_color,
            wrap=True
        )
        self.canvas_spec.draw()

    def _init_empty_plots(self):
        self._init_empty_wave_plot()
        self._init_empty_spec_plot()

    # -------- Plot helpers con datos --------

    def _plot_wave(self, y, sr, title=""):
        self.last_wave_data = (y, sr, title)
        self.ax_wave.clear()
        self._style_wave_axes()

        t = np.arange(len(y)) / sr if len(y) > 0 else np.arange(1) / 1
        color = "#00f5a0" if self.dark_mode else "#2f6fed"
        self.ax_wave.plot(t, y, linewidth=1.0, color=color)
        if len(t) > 0:
            self.ax_wave.set_xlim(0, t[-1])
        self.ax_wave.set_title(title, fontsize=11, color=self._wave_axis_color)
        self.ax_wave.set_xlabel("Time [s]", fontsize=10, color=self._wave_axis_color)
        self.ax_wave.set_ylabel("Amplitude", fontsize=10, color=self._wave_axis_color)

        # <-- asegurar que el xlabel no se corte cuando hay datos
        self.fig_wave.tight_layout()
        self.canvas_wave.draw()

    def _plot_spec(self, mel_db, overlay=None):
        """Espectrograma con un SOLO cmap fijo (magma) para ambos modos."""
        self.last_spec_data = (mel_db, overlay)
        self.ax_spec.clear()
        self._style_spec_axes()

        # Usar siempre el mismo mapa de colores para ambos modos
        im = self.ax_spec.imshow(
            mel_db,
            origin='lower',
            aspect='auto',
            cmap='magma',
            interpolation='nearest'
        )
        self.ax_spec.set_title("Espectrograma Mel (dB)", fontsize=11, color=self._spec_axis_color)
        self.ax_spec.set_xlabel("Frames", fontsize=10, color=self._spec_axis_color)
        self.ax_spec.set_ylabel("Mel bins", fontsize=10, color=self._spec_axis_color)

        if overlay is not None:
            self.ax_spec.imshow(overlay, origin='lower', aspect='auto', alpha=0.45, cmap='jet')

        # Barra de color: se crea solo una vez y luego solo se actualiza
        if self.spec_colorbar is None:
            self.spec_colorbar = self.fig_spec.colorbar(im, fraction=0.046, pad=0.04)
        else:
            self.spec_colorbar.update_normal(im)

        self._restyle_colorbar()
        self.canvas_spec.draw()

    # -------- Handlers --------

    def on_load_wav(self):
        try:
            dlg = QFileDialog(self, "Open WAV", "data", "WAV files (*.wav)")
            if dlg.exec():
                path = dlg.selectedFiles()[0]
                config_sample_rate = self.config.get('sample_rate', 16000)
                audio, sr = librosa.load(path, sr=config_sample_rate)

                self.raw_audio, self.raw_sr = audio, sr
                self._plot_wave(audio, sr, title="Raw audio (loaded)")
                QMessageBox.information(
                    self, "Loaded",
                    f"Loaded {os.path.basename(path)}\nSamplerate: {sr} Hz\nDuration: {len(audio)/sr:.2f} s"
                )
        except Exception as e:
            QMessageBox.critical(self, "Error loading WAV", str(e))

    def on_record(self):
        secs = self.seconds.value()
        self.btn_record.setEnabled(False)
        self.thread = RecordThread(seconds=secs, samplerate=44100)
        self.thread.finished.connect(self._on_record_done)
        self.thread.start()

    def _on_record_done(self, audio, sr):
        self.btn_record.setEnabled(True)
        self.raw_audio = audio
        self.raw_sr = sr
        self._plot_wave(audio, sr, title="Raw audio (recorded)")
        QMessageBox.information(self, "Recorded", f"Captured {audio.shape[0]/sr:.2f} s at {sr} Hz.")

    def on_play_raw(self):
        if self.raw_audio is None:
            QMessageBox.warning(self, "No audio", "Graba o carga un audio primero.")
            return
        Player().play(self.raw_audio, self.raw_sr)

    def on_save(self):
        if self.raw_audio is None:
            QMessageBox.warning(self, "Audio no encontrado", "Graba o carga un audio primero.")
            return
        dlg = QFileDialog(self, "Save WAV", DATA_DIR, "WAV files (*.wav)")
        dlg.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        if dlg.exec():
            path = dlg.selectedFiles()[0]
            try:
                save_wav_16k(path, self.raw_audio, self.raw_sr)
                self.last_saved_path = path
                QMessageBox.information(self, "Guardado", f"Guardado en {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error de guardado", str(e))

    def on_predict(self):
        if self.raw_audio is None:
            QMessageBox.warning(self, "No audio", "Graba o carga un audio primero.")
            return
        try:
            threshold = None
            robust = self.robust_checkbox.isChecked()
            self.last_result = predict_from_file_or_array(
                self.model_path, self.raw_audio, self.config,
                threshold_parkinson=threshold, robust=robust
            )
            result = self.last_result
            if result is None:
                QMessageBox.warning(self, "Predicción fallida", "No se obtuvo resultado.")
                self.btn_export_pdf.setEnabled(False)
                return

            color = "#2ecc40" if "SALUDABLE" in result['prediction'] else (
                    "#ffb300" if "MODERADO" in result['prediction']
                    else ("#e74c3c" if "ALTO" in result['prediction'] else "#888")
            )
            html = f"""
            <div style='font-size:16pt; font-weight:bold; color:{color}; margin-bottom:6px;'>
                {result['prediction']}
            </div>
            <div style='font-size:11.5pt; margin-bottom:6px;'>
                <b>Explicación:</b> {result['explanation']}
            </div>
            <div style='font-size:11pt; line-height:1.5;'>
                <b>Confianza:</b> {result['confidence']:.1%} ({result['confidence_level']})<br>
                <b>Score de riesgo Parkinson:</b> {result['parkinson_risk_score']:.1%}<br>
                <b>Diferencia entre clases:</b> {result['probability_difference']:.1%}<br>
                <b>Probabilidad de estar sano:</b> {result['probabilities']['Healthy']:.1%}<br>
                <b>Probabilidad de Parkinson:</b> {result['probabilities']['Parkinson']:.1%}<br>
            </div>
            <div style='font-size:9.5pt; color:#b0b3d9; margin-top:8px;'>
                IMPORTANTE: Este resultado es solo una herramienta de apoyo y no reemplaza la valoración médica profesional.
            </div>
            """
            self.prob_lbl.setText(html)
            self.prob_lbl.setTextFormat(Qt.TextFormat.RichText)
            self.btn_export_pdf.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error de predicción", str(e))
            self.btn_export_pdf.setEnabled(False)

    def on_export_pdf(self):
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, Image
            from datetime import datetime
            from io import BytesIO

            if not hasattr(self, 'last_result') or self.last_result is None:
                QMessageBox.warning(self, "Sin resultado", "Primero realiza una predicción.")
                return

            dlg = QFileDialog(self, "Guardar informe PDF", DATA_DIR, "PDF files (*.pdf)")
            dlg.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
            if not dlg.exec():
                return
            path = dlg.selectedFiles()[0]
            if not path.lower().endswith('.pdf'):
                path += '.pdf'

            # ---------------- CONFIGURACIÓN DEL DOCUMENTO ----------------
            doc = SimpleDocTemplate(
                path,
                pagesize=A4,
                rightMargin=40,
                leftMargin=40,
                topMargin=40,
                bottomMargin=40
            )
            styles = getSampleStyleSheet()

            # Estilos personalizados
            title_style = ParagraphStyle(
                'TitleCentered',
                parent=styles['Title'],
                fontName='Helvetica-Bold',
                fontSize=20,
                alignment=1,  # centrado
                spaceAfter=10
            )

            subtitle_style = ParagraphStyle(
                'Subtitle',
                parent=styles['Normal'],
                fontSize=11,
                leading=14,
                alignment=1,  # centrado
                textColor=colors.HexColor("#444a6d")
            )

            normal_style = ParagraphStyle(
                'Body',
                parent=styles['Normal'],
                fontSize=11,
                leading=14,
                alignment=4  # justificado
            )

            heading_style = ParagraphStyle(
                'SectionHeading',
                parent=styles['Heading2'],
                fontName='Helvetica-Bold',
                fontSize=13,
                textColor=colors.HexColor("#1f3b73"),
                spaceBefore=14,
                spaceAfter=6
            )

            small_grey = ParagraphStyle(
                'SmallGrey',
                parent=styles['Normal'],
                fontSize=9,
                textColor=colors.grey,
                leading=11
            )

            story = []

            # ---------------- ENCABEZADO ----------------
            story.append(Paragraph("Informe de Análisis de Voz — NeuroVoice", title_style))
            story.append(Paragraph(
                f"Fecha y hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
                subtitle_style
            ))
            story.append(Spacer(1, 12))

            # Línea separadora sutil
            line = Table(
                [['']],
                colWidths=[doc.width]
            )
            line.setStyle(TableStyle([
                ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor("#c3cae8"))
            ]))
            story.append(line)
            story.append(Spacer(1, 12))

            # ---------------- BLOQUE DE DIAGNÓSTICO ----------------
            color = colors.green if "SALUDABLE" in self.last_result['prediction'] else (
                colors.orange if "MODERADO" in self.last_result['prediction'] else (
                    colors.red if "ALTO" in self.last_result['prediction'] else colors.grey
                )
            )

            diag_table = Table(
                [[self.last_result['prediction']]],
                colWidths=[doc.width],
                hAlign='CENTER',
                style=[
                    ('BACKGROUND', (0, 0), (-1, -1), color),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 16),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                    ('TOPPADDING', (0, 0), (-1, -1), 10),
                    ('BOX', (0, 0), (-1, -1), 0.5, colors.white),
                ]
            )
            story.append(diag_table)
            story.append(Spacer(1, 16))

            # ---------------- 1. EXPLICACIÓN ----------------
            story.append(Paragraph("1. Explicación del resultado", heading_style))
            story.append(Paragraph(self.last_result['explanation'], normal_style))
            story.append(Spacer(1, 10))

            # ---------------- 2. RESUMEN NUMÉRICO ----------------
            story.append(Paragraph("2. Resumen numérico del modelo", heading_style))

            data = [
                ["Métrica", "Valor"],
                ["Confianza", f"{self.last_result['confidence']:.1%} ({self.last_result['confidence_level']})"],
                ["Score de riesgo Parkinson", f"{self.last_result['parkinson_risk_score']:.1%}"],
                ["Diferencia entre clases", f"{self.last_result['probability_difference']:.1%}"],
                ["Probabilidad de estar sano", f"{self.last_result['probabilities']['Healthy']:.1%}"],
                ["Probabilidad de Parkinson", f"{self.last_result['probabilities']['Parkinson']:.1%}"],
            ]

            table_style = TableStyle([
                # Encabezado
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#d9e3ff")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#1f3b73")),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),

                # Cuerpo
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),

                # Bordes
                ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.HexColor("#a0accf")),
                ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor("#4f5d94")),

                # Espaciado
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
            ])

            t = Table(
                data,
                style=table_style,
                hAlign='CENTER',
                colWidths=[doc.width * 0.45, doc.width * 0.45]
            )
            story.append(t)
            story.append(Spacer(1, 10))

            # Advertencia
            story.append(Paragraph(
                "IMPORTANTE: Este resultado es solo una herramienta de apoyo y no reemplaza la valoración médica profesional.",
                small_grey
            ))
            story.append(Spacer(1, 12))

            # ---------------- 3. VISUALIZACIÓN DE LA SEÑAL ----------------
            story.append(Paragraph("3. Visualización de la señal", heading_style))

            # Utilidad para insertar figuras
            def fig_to_image(figure, max_width, max_height):
                buf = BytesIO()
                figure.savefig(buf, format='png', dpi=150)
                buf.seek(0)
                img = Image(buf)
                img._restrictSize(max_width, max_height)
                return img

            from reportlab.lib.pagesizes import A4 as RL_A4
            page_width, page_height = RL_A4
            max_width = page_width - doc.leftMargin - doc.rightMargin
            max_height = page_height / 3

            # 3.1 Señal de audio en el tiempo
            if hasattr(self, 'fig_wave') and self.fig_wave is not None:
                story.append(Paragraph("3.1 Señal de audio en el tiempo", normal_style))
                story.append(Spacer(1, 6))
                story.append(fig_to_image(self.fig_wave, max_width, max_height))
                story.append(Spacer(1, 12))

            # 3.2 Espectrograma Mel (dB)
            if hasattr(self, 'fig_spec') and self.fig_spec is not None:
                story.append(Paragraph("3.2 Espectrograma Mel (dB)", normal_style))
                story.append(Spacer(1, 6))
                story.append(fig_to_image(self.fig_spec, max_width, max_height))

            # ---------------- CONSTRUCCIÓN DEL PDF ----------------
            doc.build(story)
            QMessageBox.information(self, "PDF generado", f"Informe guardado en:\n{path}")

        except Exception as e:
            QMessageBox.critical(self, "Error al exportar PDF", str(e))


    def on_gradcam(self):
        if self.raw_audio is None:
            QMessageBox.warning(self, "No input", "Preprocess first.")
            return
        try:
            mel_db = mel_spectrogram_db(self.raw_audio, self.raw_sr)
            self._plot_spec(mel_db, overlay=None)
        except Exception as e:
            QMessageBox.critical(self, "Grad-CAM error", str(e))
