# Sistema OCR - Deep Learning

Asignatura 051 - Inteligencia Artificial

## Rendimiento

- Val CER: 0.0609 (94% precisión texto impreso)
- Dataset: 100k imágenes sintéticas
- Arquitectura: CRNN + CTC

## Ejecutar

```bash
pip install -r 2_CODIGO/requirements.txt
cd 2_CODIGO
python ocr_app_standalone.py
```

## Documentación

- Trabajo: `1_TRABAJO_ESCRITO/TFI_OCR_AlvaroLostal.pdf`
- Vídeo: `4_VIDEO/presentacion_ocr.mp4`
- Ejemplos: `5_EJEMPLOS/`

## Cumplimiento

✅ Texto impreso: 94% precisión
⚠️ Manuscrito: No entrenado (requiere dataset IAM)
