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

## Referencias

- TrashNet Dataset: Gary Thung
- FocalX AI: Robustez y seguridad adversarial
