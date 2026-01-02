"""
============================================================
APLICACIÓN OCR STANDALONE - PARA EJECUTAR EN TU PC
============================================================

INICIO RÁPIDO (si ya tienes todo instalado):
--------------------------------------------
.\\venv_ocr\\Scripts\\activate; cd 2_CODIGO; python ocr_app_standalone.py

============================================================
INSTALACIÓN COMPLETA (primera vez):
--------------------------------------------
1. Asegúrate de tener Python 3.8+ instalado

2. Entorno virtual:

   # Si ya tienes venv_ocr creado, solo actívalo:
   venv_ocr\\Scripts\\activate

   # Si NO lo tienes, créalo primero:
   python -m venv venv_ocr
   # Y luego actívalo con los comandos de arriba


3. Instala las dependencias:
   pip install -r 2_CODIGO/requirements.txt

4. Descarga el modelo entrenado desde Colab:
   - En Colab: files.download('/content/models/ocr_model_printed_final.pth')
   - Guárdalo en: 2_CODIGO/models/

5. Ejecuta:
   cd 2_CODIGO
   python ocr_app_standalone.py

6. Abre el navegador en: http://localhost:7860

============================================================
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import string
import os
import sys
import json
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================

# Rutas
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "ocr_model_printed_final.pth"
TEST_RESULTS_PATH = MODEL_DIR / "test_results_printed.json"

# Crear directorio de modelos si no existe
MODEL_DIR.mkdir(exist_ok=True)

# Vocabulario (debe coincidir con el entrenamiento)
CHARS = (
    string.ascii_letters +
    string.digits +
    string.punctuation +
    ' ' +
    'áéíóúÁÉÍÓÚñÑüÜ¿¡'
)

CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHARS)}
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}
IDX_TO_CHAR[0] = '<blank>'
NUM_CLASSES = len(CHAR_TO_IDX) + 1

print(f"Vocabulario cargado: {NUM_CLASSES} clases")

# ============================================================
# MODELO
# ============================================================

class ImprovedCRNN(nn.Module):
    """CRNN para OCR - debe coincidir exactamente con el entrenamiento"""
    def __init__(self, img_height=64, num_classes=NUM_CLASSES, hidden_size=512, num_lstm_layers=3):
        super(ImprovedCRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, (4, 3), padding=(0, 1)), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )

        self.rnn = nn.LSTM(512, hidden_size, num_lstm_layers, bidirectional=True,
                          batch_first=True, dropout=0.3 if num_lstm_layers > 1 else 0)

        # Attention mechanism (debe estar aunque no se use)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.dropout = nn.Dropout(0.3)
        self.linear1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        conv_out = self.cnn(x)
        conv_out = conv_out.squeeze(2).permute(0, 2, 1)
        rnn_out, _ = self.rnn(conv_out)
        rnn_out = self.dropout(rnn_out)
        output = self.linear1(rnn_out)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear2(output)
        output = output.permute(1, 0, 2)
        return F.log_softmax(output, dim=2)

# ============================================================
# UTILIDADES
# ============================================================

def decode_prediction(indices):
    """CTC decode"""
    chars = []
    prev_idx = -1
    for idx in indices:
        if idx != 0 and idx != prev_idx:
            char = IDX_TO_CHAR.get(idx, '')
            if char and char != '<blank>':
                chars.append(char)
        prev_idx = idx
    return ''.join(chars)

transform = A.Compose([
    A.Resize(64, 256),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ============================================================
# CARGAR MODELO
# ============================================================

def load_model():
    """Carga el modelo desde disco"""

    if not MODEL_PATH.exists():
        print("\n" + "="*70)
        print("❌ ERROR: NO SE ENCUENTRA EL MODELO")
        print("="*70)
        print(f"\nBuscando en: {MODEL_PATH.absolute()}")
        print("\n📥 DESCARGA EL MODELO:")
        print("   1. En Google Colab, ejecuta:")
        print("      from google.colab import files")
        print("      files.download('/content/models/ocr_model_printed_final.pth')")
        print("\n   2. Guarda el archivo en la carpeta 'models/' de este proyecto")
        print("\n   3. Vuelve a ejecutar este script")
        print("="*70 + "\n")
        sys.exit(1)

    print(f"\n📦 Cargando modelo desde: {MODEL_PATH}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Dispositivo: {device}")

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

        model = ImprovedCRNN(
            img_height=64,
            num_classes=NUM_CLASSES,
            hidden_size=512,
            num_lstm_layers=3
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        print("✅ Modelo cargado correctamente")

        # Intentar cargar resultados de tests
        test_metrics = {}
        if TEST_RESULTS_PATH.exists():
            try:
                with open(TEST_RESULTS_PATH, 'r') as f:
                    test_metrics = json.load(f)

                print(f"\n📊 Rendimiento del modelo (Test Set):")
                print(f"   - Test Loss: {test_metrics.get('test_loss', 0):.4f}")
                print(f"   - Test CER: {test_metrics.get('test_cer', 0):.4f} (~{(1-test_metrics.get('test_cer', 0))*100:.1f}% precisión)")
                print(f"   - Test Accuracy: {test_metrics.get('test_acc', 0):.2%}")
                print(f"   - Test WER: {test_metrics.get('test_wer', 0):.4f}")
            except Exception as e:
                print(f"⚠️ No se pudo leer test_results.json: {e}")
        else:
            # Mostrar info del checkpoint si no hay json
            if 'training_info' in checkpoint:
                info = checkpoint['training_info']
                print(f"\n📊 Rendimiento del modelo (Checkpoint):")
                print(f"   - Época: {info.get('best_epoch', 'N/A')}")
                print(f"   - Val CER: {info.get('best_val_cer', 0):.4f}")

        return model, device, checkpoint, test_metrics

    except Exception as e:
        print(f"\n❌ Error al cargar el modelo: {str(e)}")
        print("\nAsegúrate de que:")
        print("  1. El archivo del modelo es correcto")
        print("  2. Tienes PyTorch instalado")
        print("  3. El modelo fue entrenado con la misma arquitectura")
        sys.exit(1)

# ============================================================
# PREDICCIÓN
# ============================================================

def predict_image(image, model, device):
    """Predice texto en imagen"""

    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convertir a RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Aplicar transformaciones
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(device)

    # Predicción
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.exp(output)
        confidence = probs.max(2)[0].mean().item()

        _, pred_indices = output.max(2)
        pred_indices = pred_indices.squeeze(1).cpu().numpy()
        predicted_text = decode_prediction(pred_indices)

    return predicted_text, confidence

# ============================================================
# INTERFAZ GRADIO
# ============================================================

class OCRInterface:
    """Clase para manejar la interfaz"""

    def __init__(self):
        print("\n🚀 Iniciando aplicación OCR...")
        self.model, self.device, self.checkpoint, self.test_metrics = load_model()
        print("✅ Aplicación lista\n")

    def process_single_image(self, image):
        """Procesa una imagen individual"""
        if image is None:
            return "❌ No se cargó ninguna imagen", "0%", None, ""

        try:
            # Predecir
            text, confidence = predict_image(image, self.model, self.device)

            # Crear imagen anotada
            annotated = self._annotate_image(image, text, confidence)

            # Formatear salida
            result = f"📝 **Texto detectado:**\n\n{text}"
            conf_text = f"{confidence:.1%}"

            # Información adicional
            info = f"Longitud: {len(text)} caracteres"
            if confidence > 0.8:
                info += " | ✅ Alta confianza"
            elif confidence > 0.5:
                info += " | ⚠️ Confianza media"
            else:
                info += " | ❌ Baja confianza"

            return result, conf_text, annotated, info

        except Exception as e:
            return f"❌ Error: {str(e)}", "0%", None, ""

    def _annotate_image(self, image, text, confidence):
        """Añade anotaciones a la imagen"""
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        h, w = img.shape[:2]

        # Redimensionar si es muy grande
        if w > 1200:
            scale = 1200 / w
            img = cv2.resize(img, (1200, int(h * scale)))
            h, w = img.shape[:2]

        # Añadir barra inferior con info
        bar_height = 60
        result = np.ones((h + bar_height, w, 3), dtype=np.uint8) * 240
        result[:h, :] = img

        # Texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_display = text if len(text) <= 60 else text[:57] + "..."
        cv2.putText(result, f"Texto: {text_display}", (10, h + 25),
                   font, 0.5, (0, 0, 0), 1)

        # Confianza con color
        color = (0, 150, 0) if confidence > 0.8 else (200, 100, 0) if confidence > 0.5 else (200, 0, 0)
        cv2.putText(result, f"Confianza: {confidence:.1%}", (10, h + 50),
                   font, 0.5, color, 1)

        return result

    def process_batch(self, files):
        """Procesa múltiples imágenes"""
        if not files:
            return "❌ No se cargaron imágenes"

        results = []
        for idx, file in enumerate(files, 1):
            try:
                image = Image.open(file).convert('RGB')
                text, conf = predict_image(np.array(image), self.model, self.device)
                results.append(f"**{idx}. {Path(file.name).name}**\n   Texto: '{text}'\n   Confianza: {conf:.1%}\n")
            except Exception as e:
                results.append(f"**{idx}. {Path(file.name).name}**\n   ❌ Error: {str(e)}\n")

        return "\n".join(results)

    def get_model_info(self):
        """Obtiene información del modelo"""
        info = "### 📊 Información del Modelo\n\n"

        if self.test_metrics:
            # Usar métricas cargadas del JSON
            tm = self.test_metrics
            info += f"- **Test CER:** {tm.get('test_cer', 0):.4f} (~{(1-tm.get('test_cer', 0))*100:.2f}% precisión)\n"
            info += f"- **Test Accuracy:** {tm.get('test_acc', 0):.2%}\n"
            info += f"- **Test WER:** {tm.get('test_wer', 0):.4f}\n"
            info += f"- **Test Loss:** {tm.get('test_loss', 0):.4f}\n"
            if 'timestamp' in tm:
                info += f"- **Fecha entrenamiento:** {tm['timestamp'].split('T')[0]}\n"
            info += "\n"

        # Intentar complementar con info del checkpoint si hace falta
        if 'training_info' in self.checkpoint:
            ti = self.checkpoint['training_info']
            info += f"- **Mejor época:** {ti.get('best_epoch', '29')}\n"
        else:
             info += f"- **Mejor época:** 29 (Final)\n"

        info += f"- **Dispositivo:** {self.device}\n"
        info += f"- **Vocabulario:** {NUM_CLASSES} clases\n"
        info += f"- **Soporte:** Español, inglés, números, acentos (áéíóúñ¿¡)"

        return info

def create_interface(ocr):
    """Crea la interfaz de Gradio"""

    with gr.Blocks(title="OCR - Reconocimiento de Texto") as app:

        gr.Markdown("""
        # 📝 Sistema OCR - Reconocimiento de Texto

        Aplicación profesional de OCR entrenada con Deep Learning (CRNN + CTC)
        """)

        with gr.Tabs():
            # Tab 1: Imagen individual
            with gr.Tab("📷 Imagen Individual"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_img = gr.Image(label="Sube tu imagen", type="numpy", height=350)
                        btn_process = gr.Button("🔍 Reconocer Texto", variant="primary", size="lg")

                        gr.Markdown("""
                        ### 💡 Consejos:
                        - **Fondo claro** (blanco/gris)
                        - **Texto horizontal** o poco inclinado
                        - **Buena resolución** (300+ DPI)
                        - **Alto contraste** texto-fondo

                        ### ✅ Funciona bien con:
                        - Documentos escaneados
                        - Fotos de texto impreso
                        - Títulos y encabezados
                        - Texto con acentos españoles
                        """)

                    with gr.Column(scale=1):
                        output_text = gr.Textbox(label="✍️ Texto Reconocido", lines=6)
                        output_conf = gr.Textbox(label="📊 Confianza", lines=1)
                        output_img = gr.Image(label="🖼️ Resultado", height=350)
                        output_info = gr.Textbox(label="ℹ️ Información", lines=1)

                btn_process.click(
                    fn=ocr.process_single_image,
                    inputs=[input_img],
                    outputs=[output_text, output_conf, output_img, output_info]
                )

            # Tab 2: Múltiples imágenes
            with gr.Tab("📚 Procesamiento por Lotes"):
                gr.Markdown("### Procesa varias imágenes a la vez")

                batch_input = gr.File(label="Selecciona múltiples imágenes", file_count="multiple")
                batch_btn = gr.Button("🔍 Procesar Todas", variant="primary")
                batch_output = gr.Markdown(label="Resultados")

                batch_btn.click(
                    fn=ocr.process_batch,
                    inputs=[batch_input],
                    outputs=[batch_output]
                )

            # Tab 3: Info
            with gr.Tab("ℹ️ Información"):
                gr.Markdown(ocr.get_model_info())

                gr.Markdown("""
                ---
                ### 🎓 Sobre el Proyecto

                **Proyecto:** Trabajo Final - Inteligencia Artificial
                **Asignatura:** 051 - Inteligencia Artificial

                ### 🏗️ Arquitectura Técnica

                - **Modelo:** CRNN (Convolutional Recurrent Neural Network)
                - **Componentes:**
                  - CNN: Extracción de características visuales
                  - LSTM Bidireccional (3 capas): Modelado de secuencias
                  - CTC Loss: Alineación automática texto-imagen

                ### 📚 Dataset de Entrenamiento

                - 100,000 imágenes de entrenamiento sintéticas
                - 10,000 imágenes de validación
                - 5,000 imágenes de test
                - Augmentations realistas (rotación, blur, ruido, perspectiva)
                - Vocabulario extendido con caracteres españoles

                ### ⚙️ Requisitos Técnicos

                - Python 3.8+
                - PyTorch 2.0+
                - CUDA (opcional, para GPU)
                - 4GB RAM mínimo

                ### 🚀 Mejoras Futuras

                - Soporte para texto manuscrito
                - Detección automática de líneas de texto
                - Corrección ortográfica con LM
                - Exportación a múltiples formatos
                """)

        gr.Markdown("""
        ---
        <div style="text-align: center;">
            <p>💻 Desarrollado con PyTorch + Gradio | 🎓 Proyecto Académico - IA</p>
        </div>
        """)

    return app

# ============================================================
# MAIN
# ============================================================

def main():
    """Función principal"""

    print("\n" + "="*70)
    print("🚀 APLICACIÓN OCR - SISTEMA DE RECONOCIMIENTO DE TEXTO")
    print("="*70)

    # Inicializar OCR
    ocr = OCRInterface()

    # Crear interfaz
    app = create_interface(ocr)

    # Lanzar
    print("\n" + "="*70)
    print("✅ APLICACIÓN LISTA")
    print("="*70)
    print("\n📱 Abriendo navegador...")
    print("   Si no se abre automáticamente, accede a: http://localhost:7860")
    print("\n⌨️  Presiona Ctrl+C para detener\n")

    app.launch(
        server_name="0.0.0.0",  # Permite acceso desde red local
        server_port=7860,
        share=False,  # Cambia a True para link público
        show_error=True,
        theme=gr.themes.Soft()  # Tema visual de la aplicación
    )

if __name__ == "__main__":
    main()