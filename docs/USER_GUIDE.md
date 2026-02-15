# Guía de Usuario: TrashAI System

Esta guía explica cómo ejecutar, entrenar y evaluar el sistema de clasificación de residuos utilizando **Docker**. No se requiere instalación local de Python.

## Requisitos Previos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y ejecutándose.
- Git (para clonar el repositorio).

## Estructura del Proyecto

```
trash-ai/
├── data/               # Dataset (imágenes raw)
├── docs/               # Documentación
├── models/             # Archivos .keras guardados
├── src/                # Código fuente (train.py, model.py)
├── docker-compose.yml  # Orquestador de servicios
└── Dockerfile          # Definición del entorno
```

---

## Comandos Principales

### 1. Construir el Entorno

Si es la primera vez que ejecutas el proyecto o si has cambiado dependencias:

```bash
docker-compose build
```

### 2. Entrenar el Modelo

Este comando iniciará el proceso de entrenamiento automático en dos fases (Calentamiento + Fine-Tuning):

```bash
docker-compose up train
```

- **Salida**: El mejor modelo se guardará automáticamente en `models/trashnet_cnn_v1.keras`.
- **Logs**: Podrás ver el progreso (accuracy/loss) en la terminal.

### 3. Evaluar el Modelo

Para verificar qué tan bien funciona el modelo con datos que nunca ha visto:

```bash
docker-compose up evaluate
```

- **Salida**:
    - Generará un reporte de métricas en la terminal.
    - Creará la **Matriz de Confusión** en `models/confusion_matrix.png`.

---

## Solución de Problemas Comunes

**Error: `GPU will not be used`**

- Es normal si no tienes configurado Docker con soporte NVIDIA. El modelo entrenará usando CPU (un poco más lento, pero funcional).

**Baja Precisión (~50%)**

- Asegúrate de que el entrenamiento haya completado la **Fase 2 (Fine-Tuning)**. Si se detuvo antes, vuelve a ejecutar `docker-compose up train`.
