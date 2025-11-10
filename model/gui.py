import os
import numpy as np
import soundfile as sf
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

from audio_io import Recorder, Player, save_wav_16k
from preprocess import preprocess_wav_for_model, mel_spectrogram_db
from inference import predict_from_file_or_array
from gradcam import gradcam_overlay_placeholder

# mplcursors es opcional (para ver valores al pasar el ratón)
try:
    import mplcursors
    MPLCURSORS_AVAILABLE = True
except Exception:
    MPLCURSORS_AVAILABLE = False

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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroVoice — Parkinson Voice Screening (Demo)")
        self.resize(1280, 800)

        # Estado
        self.raw_audio = None
        self.raw_sr = None
        self.last_saved_path = None

        # Ruta al mejor modelo disponible (igual que predict_fixed.py)
        from inference import find_best_model
        self.model_path = find_best_model()
        if not self.model_path:
            QMessageBox.critical(self, "Modelo no encontrado", "No se encontró un modelo entrenado. Entrena uno primero.")
        self.config = None  # O carga config si lo deseas

        # UI
        self._build_ui()
        self._apply_styles()

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
        g_rec = QGroupBox("Record / Load")
        l_rec = QVBoxLayout(g_rec)

        self.seconds = QSlider(Qt.Orientation.Horizontal)
        self.seconds.setMinimum(1)
        self.seconds.setMaximum(10)
        self.seconds.setValue(3)
        self.seconds_lbl = QLabel("Duration: 3 s")
        self.seconds.valueChanged.connect(lambda v: self.seconds_lbl.setText(f"Duration: {v} s"))

        self.btn_record = QPushButton("● Record")
        self.btn_record.clicked.connect(self.on_record)

        self.btn_play = QPushButton("▶ Play (raw)")
        self.btn_play.clicked.connect(self.on_play_raw)

        self.btn_load = QPushButton("📂 Load WAV")
        self.btn_load.clicked.connect(self.on_load_wav)

        l_rec.addWidget(self.seconds_lbl)
        l_rec.addWidget(self.seconds)
        l_rec.addWidget(self.btn_record)
        l_rec.addWidget(self.btn_play)
        l_rec.addWidget(self.btn_load)

        # Grupo: Inference (sin controles avanzados para el usuario)
        g_inf = QGroupBox("Análisis de Voz")
        l_inf = QVBoxLayout(g_inf)

        # Umbral fijo interno (el usuario no lo modifica) ajustado a 0.3 para mayor sensibilidad
        self.threshold_parkinson = 0.3

        self.robust_checkbox = QCheckBox("Modo robusto (promediar)")
        self.robust_checkbox.setChecked(True)

        self.btn_predict = QPushButton("🔎 Analizar y Predecir")
        self.btn_predict.clicked.connect(self.on_predict)

        self.prob_lbl = QLabel("Probabilidad (PD): —")

        self.btn_export_pdf = QPushButton("📄 Exportar PDF")
        self.btn_export_pdf.clicked.connect(self.on_export_pdf)
        self.btn_export_pdf.setEnabled(False)

        self.btn_gradcam = QPushButton("🔥 Grad-CAM overlay")
        self.btn_gradcam.clicked.connect(self.on_gradcam)

        l_inf.addWidget(self.robust_checkbox)
        l_inf.addWidget(self.btn_predict)
        l_inf.addWidget(self.prob_lbl)
        l_inf.addWidget(self.btn_export_pdf)
        l_inf.addWidget(self.btn_gradcam)

        # Añadir grupos al panel izquierdo (solo record/load y análisis)
        for g in (g_rec, g_inf):
            left_layout.addWidget(g)
        left_layout.addStretch(1)

        # ----------- Panel de gráficas (derecha, con splitter vertical) -----------
        right_split = QSplitter(Qt.Orientation.Vertical)

        # Waveform panel
        wave_panel = QWidget()
        wave_layout = QVBoxLayout(wave_panel)
        wave_layout.setContentsMargins(6, 6, 6, 6)
        wave_layout.setSpacing(6)

        wave_title = QLabel("Waveform")
        self.fig_wave, self.ax_wave = plt.subplots(figsize=(8, 3), dpi=120)
        self.canvas_wave = FigureCanvas(self.fig_wave)
        self.toolbar_wave = NavigationToolbar(self.canvas_wave, self)
        # Mejor ajuste de tamaño para que crezca con la ventana
        self.canvas_wave.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        wave_layout.addWidget(wave_title)
        wave_layout.addWidget(self.toolbar_wave)
        wave_layout.addWidget(self.canvas_wave)

        # Mel-spectrogram panel
        spec_panel = QWidget()
        spec_layout = QVBoxLayout(spec_panel)
        spec_layout.setContentsMargins(6, 6, 6, 6)
        spec_layout.setSpacing(6)

        spec_title = QLabel("Mel-spectrogram (dB)")
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
        right_split.setSizes([400, 500])  # proporción inicial

        # Añadir paneles al splitter principal
        main_split.addWidget(left_panel)
        main_split.addWidget(right_split)
        main_split.setSizes([420, 860])  # ancho inicial (controles | gráficas)

    def _apply_styles(self):
        # Estética simple (oscuro / padding / bordes suaves)
        self.setStyleSheet("""
            QWidget { font-size: 11pt; }
            QGroupBox {
                font-weight: 600;
                border: 1px solid #444;
                border-radius: 8px;
                margin-top: 10px;
                padding: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 4px;
            }
            QPushButton {
                padding: 6px 10px;
            }
            QLabel { padding: 2px 0; }
        """)

    # -------- Handlers --------

    def on_load_wav(self):
        try:
            dlg = QFileDialog(self, "Open WAV", "data", "WAV files (*.wav)")
            if dlg.exec():
                path = dlg.selectedFiles()[0]
                y, sr = sf.read(path, dtype="float32", always_2d=False)
                if y.ndim > 1:
                    y = y.mean(axis=1)
                self.raw_audio, self.raw_sr = y, sr
                self._plot_wave(self.raw_audio, self.raw_sr, title=f"Raw audio (loaded)")
                QMessageBox.information(
                    self, "Loaded",
                    f"Loaded {os.path.basename(path)}\nSamplerate: {sr} Hz\nDuration: {len(y)/sr:.2f} s"
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
            QMessageBox.warning(self, "No audio", "Record or load audio first.")
            return
        Player().play(self.raw_audio, self.raw_sr)

    # Eliminado: preprocesamiento manual, ahora todo es automático en on_predict

    def on_save(self):
        if self.raw_audio is None:
            QMessageBox.warning(self, "Nothing to save", "Graba o carga un audio primero.")
            return
        dlg = QFileDialog(self, "Save WAV", DATA_DIR, "WAV files (*.wav)")
        dlg.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        if dlg.exec():
            path = dlg.selectedFiles()[0]
            try:
                save_wav_16k(path, self.raw_audio, self.raw_sr)
                self.last_saved_path = path
                QMessageBox.information(self, "Saved", f"Saved {path}")
            except Exception as e:
                QMessageBox.critical(self, "Save error", str(e))

    def on_predict(self):
        if self.raw_audio is None:
            QMessageBox.warning(self, "No audio", "Graba o carga un audio primero.")
            return
        try:
            # Umbral fijo y modo robusto opcional
            # Usar umbral calibrado por defecto (inference lo cargará si pasamos None)
            threshold = None
            robust = self.robust_checkbox.isChecked()
            self.last_result = predict_from_file_or_array(self.model_path, self.raw_audio, self.config, threshold_parkinson=threshold, robust=robust)
            result = self.last_result
            if result is None:
                QMessageBox.warning(self, "Predicción fallida", "No se obtuvo resultado.")
                self.btn_export_pdf.setEnabled(False)
                return
            # Graficar waveform y espectrograma del audio procesado internamente por el predictor
            # (opcional: si quieres mostrar el audio procesado, puedes extraerlo del predictor)
            # Construir mensaje llamativo
            color = "#2ecc40" if "SALUDABLE" in result['prediction'] else ("#ffb300" if "MODERADO" in result['prediction'] else ("#e74c3c" if "ALTO" in result['prediction'] else "#888"))
            html = f"""
            <div style='font-size:15pt; font-weight:bold; color:{color}; margin-bottom:8px;'>
                {result['prediction']}
            </div>
            <div style='font-size:11pt; margin-bottom:6px;'><b>Explicación:</b> {result['explanation']}</div>
            <div style='font-size:10.5pt;'>
                <b>Confianza:</b> {result['confidence']:.1%} ({result['confidence_level']})<br>
                <b>Score de riesgo Parkinson:</b> {result['parkinson_risk_score']:.1%}<br>
                <b>Diferencia entre clases:</b> {result['probability_difference']:.1%}<br>
                <b>Probabilidad de estar sano:</b> {result['probabilities']['Healthy']:.1%}<br>
                <b>Probabilidad de Parkinson:</b> {result['probabilities']['Parkinson']:.1%}<br>
            </div>
            <div style='font-size:9pt; color:#888; margin-top:8px;'>IMPORTANTE: Este resultado es solo una herramienta de apoyo y no reemplaza la valoración médica profesional.</div>
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
            from reportlab.pdfgen import canvas
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
            from datetime import datetime

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

            doc = SimpleDocTemplate(path, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []

            # Título
            story.append(Paragraph("<b>Informe de Análisis de Voz — NeuroVoice</b>", styles['Title']))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"<b>Fecha y hora:</b> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", styles['Normal']))
            story.append(Spacer(1, 8))

            # Diagnóstico principal
            color = colors.green if "SALUDABLE" in self.last_result['prediction'] else (colors.orange if "MODERADO" in self.last_result['prediction'] else (colors.red if "ALTO" in self.last_result['prediction'] else colors.grey))
            diag_table = Table([[self.last_result['prediction']]], style=[
                ('BACKGROUND', (0,0), (-1,-1), color),
                ('TEXTCOLOR', (0,0), (-1,-1), colors.white),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTSIZE', (0,0), (-1,-1), 16),
                ('BOTTOMPADDING', (0,0), (-1,-1), 10),
                ('TOPPADDING', (0,0), (-1,-1), 10),
            ])
            story.append(diag_table)
            story.append(Spacer(1, 10))

            # Explicación
            story.append(Paragraph(f"<b>Explicación:</b> {self.last_result['explanation']}", styles['Normal']))
            story.append(Spacer(1, 8))

            # Detalles numéricos
            data = [
                ["Confianza", f"{self.last_result['confidence']:.1%} ({self.last_result['confidence_level']})"],
                ["Score de riesgo Parkinson", f"{self.last_result['parkinson_risk_score']:.1%}"],
                ["Diferencia entre clases", f"{self.last_result['probability_difference']:.1%}"],
                ["Probabilidad de estar sano", f"{self.last_result['probabilities']['Healthy']:.1%}"],
                ["Probabilidad de Parkinson", f"{self.last_result['probabilities']['Parkinson']:.1%}"],
            ]
            t = Table(data, style=[
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('FONTSIZE', (0,0), (-1,-1), 11),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                ('TOPPADDING', (0,0), (-1,-1), 4),
            ])
            story.append(t)
            story.append(Spacer(1, 10))

            # Advertencia
            story.append(Paragraph("<font color='grey' size=9>IMPORTANTE: Este resultado es solo una herramienta de apoyo y no reemplaza la valoración médica profesional.</font>", styles['Normal']))

            doc.build(story)
            QMessageBox.information(self, "PDF generado", f"Informe guardado en:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error al exportar PDF", str(e))

    def on_gradcam(self):
        if self.proc_audio is None:
            QMessageBox.warning(self, "No input", "Preprocess first.")
            return
        try:
            mel_db = mel_spectrogram_db(self.proc_audio, self.proc_sr)
            overlay = gradcam_overlay_placeholder(mel_db)  # HxW heatmap [0,1]
            self._plot_spec(mel_db, overlay=overlay)
        except Exception as e:
            QMessageBox.critical(self, "Grad-CAM error", str(e))

    # -------- Plot helpers --------

    def _plot_wave(self, y, sr, title=""):
        self.ax_wave.clear()
        t = np.arange(len(y)) / sr if len(y) > 0 else np.arange(1) / 1
        self.ax_wave.plot(t, y, linewidth=0.9)
        if len(t) > 0:
            self.ax_wave.set_xlim(0, t[-1])
        self.ax_wave.set_title(title, fontsize=10)
        self.ax_wave.set_xlabel("Time [s]", fontsize=9)
        self.ax_wave.set_ylabel("Amplitude", fontsize=9)
        self.ax_wave.grid(alpha=0.25)
        self.fig_wave.tight_layout()
        self.canvas_wave.draw()

        # tooltip interactivo opcional
        if MPLCURSORS_AVAILABLE:
            mplcursors.cursor(self.ax_wave, hover=True)

    def _plot_spec(self, mel_db, overlay=None):
        self.ax_spec.clear()
        im = self.ax_spec.imshow(mel_db, origin='lower', aspect='auto', cmap='magma')
        self.ax_spec.set_title("Mel-spectrogram (dB)", fontsize=10)
        self.ax_spec.set_xlabel("Frames", fontsize=9)
        self.ax_spec.set_ylabel("Mel bins", fontsize=9)
        if overlay is not None:
            self.ax_spec.imshow(overlay, origin='lower', aspect='auto', alpha=0.45, cmap='jet')
        self.fig_spec.colorbar(im, ax=self.ax_spec, fraction=0.046, pad=0.04)
        self.fig_spec.tight_layout()
        self.canvas_spec.draw()

        if MPLCURSORS_AVAILABLE:
            mplcursors.cursor(self.ax_spec, hover=True)