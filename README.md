# Sistema de Clasificación de Residuos TrashNet

Este proyecto implementa un Modelo Mínimo Viable (MVM) para la clasificación de residuos utilizando una Red Neuronal Convolucional (CNN) y el dataset TrashNet.

## Estructura del Proyecto

```
trash-ai/
├── data/
│   └── raw/dataset-resized/  # Dataset (Glass, Paper, Cardboard, Plastic, Metal, Trash)
├── models/                   # Modelos entrenados (.keras)
├── src/
│   ├── data_loader.py        # Carga y preprocesamiento de datos
│   ├── model.py              # Arquitectura CNN (FocalX)
│   ├── train.py              # Entrenamiento del modelo
│   └── evaluate.py           # Evaluación y métricas
├── requirements.txt          # Dependencias
└── README.md
```

## Requisitos

- Python 3.8+
- Dependencias listadas en `requirements.txt`

## Instalación

1.  Clonar el repositorio o descargar el código.
2.  Instalar dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

### 1. Preparación de Datos

El dataset **TrashNet** debe estar descargado y descomprimido en `data/raw/dataset-resized`.

### 2. Entrenamiento

Para entrenar el modelo (aprox. 70 épocas):

```bash
python src/train.py
```

El mejor modelo se guardará automáticamente en `models/trashnet_cnn_v1.keras`.

### 3. Evaluación

Para evaluar el modelo y generar la matriz de confusión (`confusion_matrix.png`):

```bash
python src/evaluate.py
```

## Historial de Implementación (Log de Cambios)

De acuerdo al plan de ejecución, se han realizado las siguientes implementaciones:

- **Fase 1**: `feat: implement data loading and preprocessing for TrashNet`
    - Implementado en `src/data_loader.py`: Redimensionamiento 64x64, Normalización, Split 75/25.

- **Fase 2**: `feat: define CNN architecture with Conv2D and Dropout layers`
    - Implementado en `src/model.py`: 3 Bloques Conv, FocalX GaussianNoise, Dropout.

- **Fase 3**: `fix: optimize training pipeline with Adam and EarlyStopping`
    - Implementado en `src/train.py`: Adam, Categorical Cross-Entropy, Callbacks.

- **Fase 4**: `docs: update evaluation metrics and confusion matrix results`
    - Implementado en `src/evaluate.py`: Reporte de clasificación y matriz de confusión.

## Referencias

- TrashNet Dataset: Gary Thung
- FocalX AI: Robustez y seguridad adversarial
