# Sistema OCR - Transfer Learning

**Asignatura:** Inteligencia Artificial
**Autor:** Álvaro Lostal Sanz
**Fecha:** Enero 2026

## 📊 Rendimiento

### Modelo Texto Impreso

- **Precisión:** 93.97% (CER 0.0603)
- **Dataset:** 100k imágenes sintéticas
- **Tiempo:** 8h entrenamiento

### Modelo Texto Manuscrito

- **Precisión:** 89.95% (CER 0.1005)
- **Dataset:** 96k palabras IAM
- **Estrategia:** Transfer Learning
- **Tiempo:** 6.5h entrenamiento

### Validación Transfer Learning

- Con Transfer: 89.95%
- Desde cero: 82.51%
- **Mejora: +7.44 puntos**

## 🚀 Ejecutar

**Instalación:**

    pip install -r 2_CODIGO/requirements.txt
    cd 2_CODIGO
    python ocr_app_standalone.py

**Abrir:** http://localhost:7860

## 📂 Estructura

    OCR_AlvaroLostal/
    ├── 1_TRABAJO_ESCRITO/
    │   └── TFI_OCR_AlvaroLostal.pdf
    ├── 2_CODIGO/
    │   ├── models/                    (2 modelos .pth + métricas)
    │   ├── training/                  (2 notebooks Colab)
    │   ├── ocr_app_standalone.py
    │   └── requirements.txt
    ├── 3_REGISTRO_RESULTADOS/
    │   └── bitacora.md
    ├── 4_VIDEO/
    │   ├── presentacion_ocr.mp4
    │   └── COMO_EJECUTAR.txt
    ├── 5_EJEMPLOS/
    │   ├── printed/                   (10 imágenes)
    │   └── handwriting/               (10 imágenes)
    └── README.md

## 🎯 Tecnologías

- **Framework:** PyTorch 2.0+
- **Arquitectura:** CRNN (5 CNN + 3 BiLSTM)
- **Loss:** CTC
- **Interfaz:** Gradio
- **Training:** Google Colab GPU T4

## ✅ Cumplimiento

- ✅ Texto impreso: 93.97% (≥85%)
- ✅ Texto manuscrito: 89.95% (≥85%)
- ✅ Sin librerías OCR externas
- ✅ Formatos: JPG, PNG, BMP, TIFF

## 📚 Documentación

- **Trabajo completo:** 1_TRABAJO_ESCRITO/TFI_OCR_AlvaroLostal.pdf
- **Vídeo demo:** 4_VIDEO/presentacion_ocr.mp4
- **Registro técnico:** 3_REGISTRO_RESULTADOS/bitacora.md
- **Ejemplos:** 5_EJEMPLOS/

## 🔬 Conceptos Clave

- Transfer Learning (impreso → manuscrito)
- Fine-tuning progresivo (freeze/unfreeze)
- Data augmentation agresivo
- CTC Loss (alineación automática)
- BiLSTM (contexto bidireccional)
