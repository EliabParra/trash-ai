# Model Card: TrashNet MobileNetV2 Classifier

## Resumen del Modelo

Este modelo es un clasificador de imágenes de residuos sólidos basado en **MobileNetV2** (Transfer Learning), optimizado para ejecutarse en entornos con recursos limitados (Edge AI) y diseñado para ser robusto frente a ruido visual (concepto **FocalX**).

**Tipo de Modelo**: Red Neuronal Convolucional (CNN) con Transfer Learning.
**Framework**: TensorFlow / Keras.
**Input**: Imágenes RGB de 224x224 píxeles.
**Output**: Clasificación Softmax de 6 clases.

---

## Arquitectura Técnica

### 1. Base Model (Feature Extractor)

- **Base**: `MobileNetV2` (pre-entrenada en ImageNet).
- **Pesos**: Transferencia de conocimiento inicial, seguido de Fine-Tuning.
- **Ventaja**: Arquitectura ligera que utiliza _Depthwise Separable Convolutions_ para reducir drásticamente el costo computacional sin sacrificar precisión.

### 2. Capas Personalizadas (Top Head)

Se ha diseñado una cabecera de clasificación específica para evitar el _overfitting_ en el pequeño dataset TrashNet.

| Capa                       | Configuración       | Propósito                                                                                                                                |
| :------------------------- | :------------------ | :--------------------------------------------------------------------------------------------------------------------------------------- |
| **Input**                  | `(224, 224, 3)`     | Entrada estándar de MobileNetV2.                                                                                                         |
| **GaussianNoise**          | `stddev=0.01`       | **FocalX Robustness**: Simula ruido de sensor/cámara para forzar a la red a aprender características estructurales y no píxeles exactos. |
| **MobileNetV2**            | `include_top=False` | Extracción de características de alto nivel.                                                                                             |
| **GlobalAveragePooling2D** | -                   | Reduce los mapas de características espaciales a un vector global, minimizando parámetros.                                               |
| **Dense**                  | `128 units, ReLU`   | Procesamiento intermedio.                                                                                                                |
| **Dropout**                | `rate=0.2`          | Regularización para prevenir memorización (overfitting).                                                                                 |
| **Output (Dense)**         | `6 units, Softmax`  | Probabilidades finales por clase.                                                                                                        |

---

## Pipeline de Entrenamiento (Two-Stage)

El entrenamiento se realiza en dos fases automáticas para maximizar la estabilidad y precisión:

### Fase 1: Calentamiento (Head Training)

- **Objetivo**: Entrenar solo las capas nuevas (Dense) mientras MobileNetV2 se mantiene congelado.
- **Optimizador**: Adam (`lr=1e-3`).
- **Duración**: Hasta convergencia temprana (aprox. 10-15 épocas).

### Fase 2: Fine-Tuning

- **Objetivo**: Descongelar MobileNetV2 y ajustar _todos_ los pesos para adaptarse específicamente a las texturas de basura.
- **Optimizador**: Adam con **Low Learning Rate** (`lr=1e-5`).
- **Data Augmentation**:
    - Rotación: ±30°
    - Zoom: ±20%
    - Shift (H/V): ±20%
    - Horizontal Flip
    - _Nota: Aplicado solo al set de entrenamiento._

---

## Métricas y Desempeño

**Evaluación Final (Test Set)**:

- **Accuracy**: ~75%
- **F1-Score Promedio**: 0.76

### Análisis por Clase

- **Alto Desempeño**: Papel, Cartón.
- **Desafíos**: Confusión entre Vidrio, Metal y Plástico debido a similitudes visuales (transparencia/reflejos) y tamaño limitado del dataset.

## Licencia y Uso

Modelo diseñado para fines académicos y experimentales. Dataset TrashNet propiedad de Gary Thung.
