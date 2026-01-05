"""
============================================================
APLICACIÓN OCR DUAL - IMPRESO + MANUSCRITO
============================================================

INICIO RÁPIDO:
--------------------------------------------
.\\venv_ocr\\Scripts\\activate
cd 2_CODIGO
python ocr_app_standalone.py

============================================================
INSTALACIÓN (primera vez):
--------------------------------------------
1. Crear entorno virtual:
   python -m venv venv_ocr
   venv_ocr\\Scripts\\activate

2. Instalar dependencias:
   pip install -r requirements.txt

3. Modelos ya incluidos en: 2_CODIGO/models/
   - ocr_model_printed_final.pth (94% precisión)
   - ocr_model_handwriting_final.pth (90% precisión)

4. Ejecutar:
   python ocr_app_standalone.py

5. Abrir navegador: http://localhost:7860

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
from datetime import datetime

# ============================================================
# CONFIGURACIÓN
# ============================================================

# Rutas
MODEL_DIR = Path("models")
PRINTED_MODEL_PATH = MODEL_DIR / "ocr_model_printed_final.pth"
HANDWRITING_MODEL_PATH = MODEL_DIR / "ocr_model_handwriting_final.pth"
PRINTED_RESULTS_PATH = MODEL_DIR / "test_results_printed.json"
HANDWRITING_RESULTS_PATH = MODEL_DIR / "test_results_handwriting.json"

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

print(f"✅ Vocabulario: {NUM_CLASSES} clases")

# ============================================================
# MODELO
# ============================================================

class ImprovedCRNN(nn.Module):
    """
    CRNN para OCR
    Arquitectura: CNN (5 bloques) + LSTM (3 capas bidireccionales) + FC
    """
    def __init__(self, img_height=64, num_classes=NUM_CLASSES, hidden_size=512, num_lstm_layers=3):
        super(ImprovedCRNN, self).__init__()

        # CNN: Extracción de características visuales
        self.cnn = nn.Sequential(
            # Bloque 1: 64 filtros
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x128

            # Bloque 2: 128 filtros
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x64

            # Bloque 3: 256 filtros
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # 16x64

            # Bloque 4: 512 filtros
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # 8x64

            # Bloque 5: 512 filtros (sin pooling)
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Reducción final
            nn.Conv2d(512, 512, (4, 3), padding=(0, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # RNN: Modelado de secuencias
        self.rnn = nn.LSTM(
            512,
            hidden_size,
            num_lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.3 if num_lstm_layers > 1 else 0
        )

        # Attention mechanism (opcional, no usado en inferencia)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Fully Connected: Clasificación
        self.dropout = nn.Dropout(0.3)
        self.linear1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # CNN
        conv_out = self.cnn(x)  # (batch, 512, 1, width)
        conv_out = conv_out.squeeze(2).permute(0, 2, 1)  # (batch, width, 512)

        # RNN
        rnn_out, _ = self.rnn(conv_out)  # (batch, width, hidden*2)
        rnn_out = self.dropout(rnn_out)

        # FC
        output = self.linear1(rnn_out)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear2(output)  # (batch, width, num_classes)

        # Formato CTC: (width, batch, num_classes)
        output = output.permute(1, 0, 2)

        return F.log_softmax(output, dim=2)

# ============================================================
# UTILIDADES
# ============================================================

def decode_prediction(indices):
    """
    CTC Decoding: Elimina blanks y caracteres repetidos
    """
    chars = []
    prev_idx = -1
    for idx in indices:
        if idx != 0 and idx != prev_idx:  # 0 = blank
            char = IDX_TO_CHAR.get(idx, '')
            if char and char != '<blank>':
                chars.append(char)
        prev_idx = idx
    return ''.join(chars)

# Transformaciones para inferencia
transform = A.Compose([
    A.Resize(64, 256),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ============================================================
# CARGAR MODELOS
# ============================================================

def load_model(model_path, model_type="printed"):
    """
    Carga un modelo desde disco

    Args:
        model_path: Ruta al archivo .pth
        model_type: "printed" o "handwriting"

    Returns:
        model, device, checkpoint, test_metrics
    """

    if not model_path.exists():
        print("\n" + "="*70)
        print(f"❌ ERROR: NO SE ENCUENTRA EL MODELO {model_type.upper()}")
        print("="*70)
        print(f"\nBuscando en: {model_path.absolute()}")
        print("\n📥 SOLUCIÓN:")
        print(f"   Asegúrate de que el archivo existe en:")
        print(f"   {model_path}")
        print("="*70 + "\n")
        return None, None, None, None

    print(f"\n📦 Cargando modelo {model_type}...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        model = ImprovedCRNN(
            img_height=64,
            num_classes=NUM_CLASSES,
            hidden_size=512,
            num_lstm_layers=3
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        print(f"✅ Modelo {model_type} cargado")

        # Cargar métricas de test
        test_metrics = {}
        results_path = PRINTED_RESULTS_PATH if model_type == "printed" else HANDWRITING_RESULTS_PATH

        if results_path.exists():
            with open(results_path, 'r') as f:
                test_metrics = json.load(f)

            cer = test_metrics.get('test_cer', 0)
            acc = test_metrics.get('test_acc', 0)
            print(f"   📊 Test CER: {cer:.4f} ({(1-cer)*100:.2f}% precisión)")
            print(f"   📊 Test ACC: {acc:.2%}")

        return model, device, checkpoint, test_metrics

    except Exception as e:
        print(f"\n❌ Error al cargar modelo {model_type}: {str(e)}")
        return None, None, None, None

# ============================================================
# PREDICCIÓN
# ============================================================

def predict_image(image, model, device):
    """
    Predice texto en una imagen

    Args:
        image: numpy array o PIL Image
        model: modelo CRNN
        device: torch device

    Returns:
        predicted_text, confidence
    """

    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convertir a RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Transformaciones
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(device)

    # Predicción
    with torch.no_grad():
        output = model(image_tensor)

        # Calcular confianza
        probs = torch.exp(output)
        confidence = probs.max(2)[0].mean().item()

        # Decodificar
        _, pred_indices = output.max(2)
        pred_indices = pred_indices.squeeze(1).cpu().numpy()
        predicted_text = decode_prediction(pred_indices)

    return predicted_text, confidence

# ============================================================
# INTERFAZ GRADIO
# ============================================================

class DualOCRInterface:
    """Interfaz para ambos modelos (impreso + manuscrito)"""

    def __init__(self):
        # Cargar modelo impreso
        self.model_printed, self.device, self.checkpoint_printed, self.metrics_printed = \
            load_model(PRINTED_MODEL_PATH, "printed")

        # Cargar modelo manuscrito
        self.model_handwriting, _, self.checkpoint_handwriting, self.metrics_handwriting = \
            load_model(HANDWRITING_MODEL_PATH, "handwriting")

        # Verificar que al menos un modelo cargó
        if self.model_printed is None and self.model_handwriting is None:
            print("\nERROR: No se pudo cargar ningún modelo")
            sys.exit(1)

    def process_single_image(self, image, model_type):
        """
        Procesa una imagen con el modelo seleccionado

        Args:
            image: imagen a procesar
            model_type: "Texto Impreso" o "Texto Manuscrito"

        Returns:
            result_text, confidence, annotated_image, info
        """

        if image is None:
            return "❌ No se cargó ninguna imagen", "0%", None, ""

        # Seleccionar modelo
        if model_type == "Texto Impreso":
            model = self.model_printed
            if model is None:
                return "❌ Modelo impreso no disponible", "0%", None, ""
        else:  # Texto Manuscrito
            model = self.model_handwriting
            if model is None:
                return "❌ Modelo manuscrito no disponible", "0%", None, ""

        try:
            # Predecir
            text, confidence = predict_image(image, model, self.device)

            # Crear imagen anotada
            annotated = self._annotate_image(image, text, confidence, model_type)

            # Formatear salida
            emoji = "🖨️" if model_type == "Texto Impreso" else "✍️"
            result = f"📝 **Texto detectado ({model_type}):**\n\n{text}"
            conf_text = f"{confidence:.1%}"

            # Información adicional
            info = f"Longitud: {len(text)} caracteres | Modelo: {model_type}"
            if confidence > 0.8:
                info += " | ✅ Alta confianza"
            elif confidence > 0.5:
                info += " | ⚠️ Confianza media"
            else:
                info += " | ❌ Baja confianza"

            return result, conf_text, annotated, info

        except Exception as e:
            return f"❌ Error: {str(e)}", "0%", None, ""

    def _annotate_image(self, image, text, confidence, model_type):
        """Añade anotaciones visuales a la imagen"""

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

        # Barra inferior con información
        bar_height = 80
        result = np.ones((h + bar_height, w, 3), dtype=np.uint8) * 240
        result[:h, :] = img

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Tipo de modelo
        emoji = "🖨️" if model_type == "Texto Impreso" else "✍️"
        cv2.putText(result, f"{emoji} {model_type}", (10, h + 20),
                   font, 0.6, (50, 50, 50), 2)

        # Texto detectado
        text_display = text if len(text) <= 55 else text[:52] + "..."
        cv2.putText(result, f"Texto: {text_display}", (10, h + 45),
                   font, 0.5, (0, 0, 0), 1)

        # Confianza con color
        color = (0, 150, 0) if confidence > 0.8 else (200, 100, 0) if confidence > 0.5 else (200, 0, 0)
        cv2.putText(result, f"Confianza: {confidence:.1%}", (10, h + 70),
                   font, 0.5, color, 2)

        return result

    def process_batch(self, files, model_type, progress=gr.Progress()):
        """Procesa múltiples imágenes con barra de progreso"""

        if not files:
            return "❌ No se cargaron imágenes"

        # Seleccionar modelo
        model = self.model_printed if model_type == "Texto Impreso" else self.model_handwriting

        if model is None:
            return f"❌ Modelo {model_type} no disponible"

        results = []
        emoji = "🖨️" if model_type == "Texto Impreso" else "✍️"
        total = len(files)

        results.append(f"## {emoji} Resultados: {model_type}\n\n")
        results.append(f"**Total procesadas:** {total} imágenes\n\n")

        for idx, file in enumerate(progress.tqdm(files, desc="🔍 Procesando imágenes...")):
            try:
                image = Image.open(file).convert('RGB')
                text, conf = predict_image(np.array(image), model, self.device)

                conf_icon = "✅" if conf > 0.8 else "⚠️" if conf > 0.5 else "❌"
                results.append(
                    f"**{idx+1}. {Path(file.name).name}** {conf_icon}\n"
                    f"   📝 Texto: `{text}`\n"
                    f"   📊 Confianza: {conf:.1%}\n"
                )
            except Exception as e:
                results.append(
                    f"**{idx+1}. {Path(file.name).name}** ❌\n"
                    f"   Error: {str(e)}\n"
                )

        return "\n".join(results)

    def get_model_info(self, model_type):
        """Obtiene información detallada de un modelo"""

        if model_type == "Texto Impreso":
            metrics = self.metrics_printed
            checkpoint = self.checkpoint_printed
            dataset = "100,000 imágenes sintéticas"
            strategy = "Generación sintética con variaciones tipográficas"
        else:
            metrics = self.metrics_handwriting
            checkpoint = self.checkpoint_handwriting
            dataset = "96,456 palabras manuscritas (IAM Database)"
            strategy = "Transfer Learning desde modelo impreso"

        if not metrics:
            return f"### {model_type}\n\nModelo no disponible"

        info = f"### {model_type}\n\n"
        info += "#### Métricas de Evaluación (Test Set)\n\n"

        cer = metrics.get('test_cer', 0)
        acc = metrics.get('test_acc', 0)
        wer = metrics.get('test_wer', 0)
        loss = metrics.get('test_loss', 0)

        info += f"- **Test CER:** {cer:.4f} → **{(1-cer)*100:.2f}% precisión**\n"
        info += f"- **Test Accuracy:** {acc:.2%} (palabras exactas)\n"
        info += f"- **Test WER:** {wer:.4f}\n"
        info += f"- **Test Loss:** {loss:.4f}\n\n"

        info += "#### Dataset de Entrenamiento\n\n"
        info += f"- {dataset}\n"
        info += f"- Estrategia: {strategy}\n"
        info += f"- Augmentations: Rotación, blur, ruido, perspectiva\n\n"

        if 'total_samples' in metrics:
            info += f"- **Muestras test:** {metrics['total_samples']:,}\n"
            info += f"- **Correctas:** {metrics.get('correct', 0):,}\n\n"

        info += "#### Arquitectura\n\n"
        info += "- **Modelo:** CRNN\n"
        info += "- **CNN:** 5 bloques convolucionales\n"
        info += "- **RNN:** 3 capas LSTM bidireccionales\n"
        info += "- **CTC Loss:** Alineación automática\n"
        info += "- **Parámetros:** ~22M entrenables\n"

        return info

    def get_comparison(self):
        """Comparativa entre ambos modelos"""

        info = "### Comparativa de Modelos\n\n"
        info += "| Característica | Texto Impreso | Texto Manuscrito |\n"
        info += "|----------------|---------------|------------------|\n"

        if self.metrics_printed and self.metrics_handwriting:
            p_cer = self.metrics_printed.get('test_cer', 0)
            h_cer = self.metrics_handwriting.get('test_cer', 0)
            p_acc = self.metrics_printed.get('test_acc', 0)
            h_acc = self.metrics_handwriting.get('test_acc', 0)

            info += f"| **Precisión** | {(1-p_cer)*100:.2f}% | {(1-h_cer)*100:.2f}% |\n"
            info += f"| **Accuracy** | {p_acc:.2%} | {h_acc:.2%} |\n"
            info += f"| **CER** | {p_cer:.4f} | {h_cer:.4f} |\n"

        info += "| **Dataset** | 100k sintéticas | 96k IAM reales |\n"
        info += "| **Estrategia** | Desde cero | Transfer Learning |\n"
        info += "| **Uso ideal** | Documentos, libros | Notas, formularios |\n\n"

        info += "### Recomendaciones de Uso\n\n"
        info += "**Texto Impreso:** Documentos escaneados, libros, revistas, carteles, PDFs\n\n"
        info += "**Texto Manuscrito:** Notas a mano, formularios, apuntes, cartas, anotaciones\n"

        return info

def create_interface(ocr):
    """Crea la interfaz de Gradio con mejoras UI/UX"""

    # ========================================
    # TEMA - Solo ajustes sutiles
    # ========================================
    custom_theme = gr.themes.Soft(
        primary_hue="blue",
        font=("Inter", "system-ui", "sans-serif"),
    )

    # ========================================
    # CSS PERSONALIZADO (mínimo)
    # ========================================
    custom_css = """
    /* Badges de estado */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 500;
    }
    .badge-printed { background: #387fc0; color: white; }
    .badge-handwriting { background: #10b981; color: white; }
    .badge-device { background: #6b7280; color: white; }

    /* Footer */
    .footer-text {
        text-align: center;
        color: #6b7280;
        font-size: 0.9em;
        padding: 12px;
        margin-top: 20px;
    }
    """

    with gr.Blocks(title="OCR Dual - Impreso + Manuscrito") as app:

        gr.Markdown("""
        # 📝 Sistema OCR Dual

        **Proyecto Final de Inteligencia Artificial** — Reconocimiento óptico de caracteres
        """)

        with gr.Tabs():

            # ========================================
            # TAB 1: Imagen Individual
            # ========================================
            with gr.Tab("📷 Reconocimiento"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_img = gr.Image(
                            label="Sube tu imagen",
                            type="numpy",
                            height=320
                        )

                        model_selector = gr.Radio(
                            choices=["Texto Impreso", "Texto Manuscrito"],
                            value="Texto Impreso",
                            label="Tipo de texto",
                            info="Selecciona según el contenido de tu imagen"
                        )

                        btn_process = gr.Button(
                            "Reconocer Texto",
                            variant="primary",
                            size="lg"
                        )

                        with gr.Accordion("Consejos para mejores resultados", open=False):
                            gr.Markdown("""
                            **Requisitos de imagen:**
                            - Buena iluminación uniforme
                            - Texto horizontal (máx ±5° inclinación)
                            - Alto contraste texto-fondo
                            - Resolución mínima 300 DPI

                            **Texto Impreso:** Documentos, libros, carteles
                            **Manuscrito:** Notas, formularios, apuntes
                            """)

                    with gr.Column(scale=1):
                        output_text = gr.Textbox(
                            label="Texto Reconocido",
                            lines=6,
                            placeholder="El texto reconocido aparecerá aquí..."
                        )

                        output_conf = gr.Textbox(
                            label="Confianza",
                            lines=1
                        )

                        output_img = gr.Image(
                            label="Resultado Anotado",
                            height=300
                        )

                        output_info = gr.Textbox(
                            label="Información",
                            lines=1
                        )

                gr.Markdown("*Hay imágenes de prueba disponibles en la carpeta `5_EJEMPLOS/`*")

                btn_process.click(
                    fn=ocr.process_single_image,
                    inputs=[input_img, model_selector],
                    outputs=[output_text, output_conf, output_img, output_info]
                )

            # ========================================
            # TAB 2: Procesamiento por Lotes
            # ========================================
            with gr.Tab("📚 Lotes"):
                gr.Markdown("""
                ### Procesa múltiples imágenes

                Ideal para digitalizar documentos o colecciones de imágenes.
                Puedes usar las imágenes de prueba en `5_EJEMPLOS/`.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        batch_model = gr.Radio(
                            choices=["Texto Impreso", "Texto Manuscrito"],
                            value="Texto Impreso",
                            label="Modelo a utilizar"
                        )

                        batch_input = gr.File(
                            label="Selecciona imágenes",
                            file_count="multiple",
                            file_types=["image"]
                        )

                        batch_btn = gr.Button(
                            "Procesar Todas",
                            variant="primary",
                            size="lg"
                        )

                    with gr.Column(scale=1):
                        batch_output = gr.Markdown(
                            value="*Los resultados aparecerán aquí tras el procesamiento*"
                        )

                batch_btn.click(
                    fn=ocr.process_batch,
                    inputs=[batch_input, batch_model],
                    outputs=[batch_output]
                )

            # ========================================
            # TAB 3: Información Técnica (Consolidada)
            # ========================================
            with gr.Tab("📊 Modelos"):
                gr.Markdown("## Información de los Modelos")

                with gr.Row():
                    with gr.Column():
                        with gr.Accordion("Modelo Texto Impreso", open=True):
                            gr.Markdown(ocr.get_model_info("Texto Impreso"))

                    with gr.Column():
                        with gr.Accordion("Modelo Texto Manuscrito", open=True):
                            gr.Markdown(ocr.get_model_info("Texto Manuscrito"))

                with gr.Accordion("Comparativa de Modelos", open=False):
                    gr.Markdown(ocr.get_comparison())

            with gr.Tab("📖 Proyecto"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
                        ## Proyecto Final - IA

                        **Asignatura:** Inteligencia Artificial

                        **Objetivo:** Sistema OCR con Deep Learning
                        (sin librerías OCR externas como Tesseract)

                        ---

                        ### Arquitectura CRNN

                        | Componente | Función | Detalles |
                        |------------|---------|----------|
                        | **CNN** | Extracción visual | 5 bloques conv. |
                        | **RNN** | Secuencias | 3 LSTM bidireccionales |
                        | **CTC** | Alineación | Sin segmentación manual |

                        **Parámetros:** ~22M entrenables
                        """)

                    with gr.Column():
                        gr.Markdown("""
                        ### Transfer Learning

                        1. Entrenamiento con texto impreso sintético (100k imágenes)
                        2. Fine-tuning con manuscrito real (IAM Database, 96k palabras)

                        **Beneficios:**
                        - Convergencia 40% más rápida
                        - Mejor generalización

                        ---

                        ### Referencias

                        - IAM Database: Marti & Bunke (2002)
                        - CTC Loss: Graves et al. (2006)
                        - CRNN: Shi et al. (2015)
                        - Transfer Learning: Yosinski et al. (2014)
                        """)

                with gr.Row():
                    with gr.Column():
                        with gr.Accordion("Requisitos Técnicos", open=False):
                            gr.Markdown("""
                            **Software:** Python 3.8+, PyTorch 2.0+, OpenCV, Gradio

                            **Hardware:** CPU 4+ cores, 8GB RAM, GPU opcional (10x más rápido)

                            **Velocidad:** ~0.5s/imagen (CPU) | ~0.05s/imagen (GPU)
                            """)

                    with gr.Column():
                        with gr.Accordion("Posibles Mejoras", open=False):
                            gr.Markdown("""
                            - Attention Mechanism
                            - Beam Search Decoding
                            - Language Model
                            - Detección de Líneas
                            - Modelo Ensemble
                            """)

        gr.Markdown("""
        ---
        <div style="text-align: center; color: #666; font-size: 0.9em;">
            Desarrollado con PyTorch + Gradio | Trabajo Final IA | Enero 2026
        </div>
        """)

    return app, custom_theme, custom_css

# ============================================================
# MAIN
# ============================================================

def main():
    """Función principal"""

    print("\n" + "="*60)
    print("  OCR DUAL - Texto Impreso + Manuscrito")
    print("="*60)

    # Inicializar interfaz
    ocr = DualOCRInterface()

    # Crear app
    app, custom_theme, custom_css = create_interface(ocr)

    print("\n  URL: http://localhost:7860")
    print("  Ctrl+C para detener\n")

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # True para link público
        show_error=True,
        theme=custom_theme,
        css=custom_css
    )

if __name__ == "__main__":
    main()
