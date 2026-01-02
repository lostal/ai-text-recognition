# Registro de Decisiones y Resultados

## 20-12-2025 | Inicio del Proyecto

**Decisión:** Arquitectura CRNN + CTC
**Razón:** Estado del arte para OCR secuencial, no requiere segmentación manual
**Resultado:** Arquitectura implementada

## 21-12-2025 | Dataset

**Decisión:** Dataset sintético (100k imágenes)
**Razón:** Control total, escalable, no requiere etiquetado manual
**Resultado:** Generadas 100k train, 10k val, 5k test

## 22-12-2025 | Entrenamiento

**Decisión:** 41 épocas, batch 128, OneCycleLR
**Razón:** Balance entre tiempo y rendimiento
**Resultado:** Val CER 0.0609 (94% precisión)
**Tiempo:** 10 horas en T4 GPU

## 02-01-2026 | Manuscrito

**Decisión:** NO entrenar para manuscrito
**Razón:** Requiere dataset IAM + 20-40h adicionales
**Resultado:** Priorizado impreso al 94%

## 03-01-2026 | Aplicación Final

**Decisión:** Gradio para interfaz
**Razón:** Simple, interactivo, profesional
**Resultado:** Aplicación funcional
