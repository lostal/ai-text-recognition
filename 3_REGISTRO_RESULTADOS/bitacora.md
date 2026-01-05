# Registro de Decisiones y Resultados

## 📅 20-12-2024 | Arquitectura Base

**Decisión:** CRNN + CTC Loss
**Razón:** Estado del arte OCR, no requiere segmentación manual
**Resultado:** Arquitectura implementada en PyTorch

## 📅 21-12-2024 | Dataset Impreso

**Decisión:** 100k imágenes sintéticas dinámicas
**Razón:** Control total, escalable, variedad infinita
**Configuración:** 50+ fuentes, augmentations (rotación, blur, noise)
**Resultado:** Dataset generado

## 📅 22-12-2024 | Entrenamiento Impreso

**Prueba:** 50 épocas, batch 128, AdamW + OneCycleLR
**Hardware:** Google Colab GPU T4
**Resultados:**

- Val CER: 0.0609 (93.97% precisión)
- Test CER: 0.0603
- Val Accuracy: 80.78%
- Tiempo: 8 horas

**Conclusión:** Modelo base robusto

## 📅 02-01-2025 | Estrategia Manuscrito

**Decisión:** Transfer Learning desde modelo impreso
**Razón:** Dataset IAM (96k) insuficiente para init aleatorio
**Estrategia:**

- Épocas 1-15: CNN congelado, solo entrenar LSTM
- Épocas 16-35: Fine-tuning completo, LR reducido

## 📅 03-01-2025 | Entrenamiento Manuscrito - Fase 1

**Prueba:** CNN congelado (épocas 1-15)
**Resultados:**

- Época 1: Val CER 0.9891 (1% precisión)
- Época 15: Val CER 0.1323 (86.77% precisión)
- Mejora: 85.68 puntos porcentuales
- Tiempo/época: ~7 min

**Conclusión:** Transfer Learning efectivo

## 📅 04-01-2025 | Entrenamiento Manuscrito - Fase 2

**Prueba:** CNN descongelado (épocas 16-35)
**Resultados:**

- Época 35: Val CER 0.1005 (89.95% precisión)
- Test CER: 0.1005
- Test Accuracy: 74.24%
- Mejora adicional: +3.18 puntos
- Tiempo/época: ~11 min

**Conclusión:** Objetivo 85-90% superado

## 📅 04-01-2025 22:00 | Experimento Validación (Kaggle)

**Objetivo:** Validar necesidad de Transfer Learning
**Configuración:** CRNN idéntica, pesos aleatorios (sin transfer)
**Resultados:**

- Época 5: Val CER 0.1749 (82.51%) ✅ Mejor
- Época 30: Val CER 0.3088 (69.12%) ❌ Overfitting

**Comparativa:**

- Con Transfer Learning: 89.95%
- Desde cero: 82.51%
- **Diferencia: +7.44 puntos porcentuales**

**Conclusión:** Transfer Learning esencial, validado empíricamente

## 📅 05-01-2025 | Evaluación Final

**Test Set Results:**

Modelo Impreso:

- CER: 0.0603 (93.97%)
- Accuracy: 80.78%
- 5,000 muestras

Modelo Manuscrito:

- CER: 0.1005 (89.95%)
- Accuracy: 74.24%
- 9,646 muestras

**Conclusión:** Ambos superan objetivo 85-90%

## 📅 05-01-2025 | Aplicación Gradio

**Decisión:** Interfaz web interactiva
**Funcionalidades:** Dual models, batch processing, confianza
**Resultado:** Aplicación funcional

## 📅 06-01-2025 | Validación Imágenes Reales

**Prueba:** 30 palabras manuscritas (3 autores)
**Resultados:**

- 3 perfectas (100%): Jose, Barcelona, madrid
- 4 muy buenas (80-92%): Informacion, Universidad, Profesor
- 3 reconocibles (60-79%): Espana, Diciembre
- Promedio: ~82%

**Observaciones:**

- ✅ Palabras cortas: Excelente
- ⚠️ Palabras largas: Buena con errores
- ⚠️ Caracteres ñ: Confusión (dataset inglés)

**Conclusión:** Generalización real exitosa (82% real vs 90% test)
