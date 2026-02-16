# TrashAI — Guía de Despliegue con Docker

## Requisitos Previos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y corriendo
- Git

## Pasos para Ejecutar

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd trash-ai
```

### 2. Verificar que el modelo exista

El archivo `models/trashnet_cnn_v1.keras` es el modelo entrenado. Debe existir antes de iniciar el servidor web.

Si necesitas re-entrenar el modelo desde cero:

```bash
docker-compose up train
```

### 3. Iniciar la aplicación web

```bash
docker-compose up web
```

La primera ejecución descargará las dependencias (~620MB por TensorFlow). Ejecuciones posteriores serán instantáneas.

Cuando veas:

```
✅ Model loaded and ready!
🚀 TrashAI running at http://localhost:3000
```

Abre **http://localhost:3000** en tu navegador.

### 4. Usar la aplicación

1. Arrastra o selecciona una imagen de un residuo (JPG, PNG, WEBP)
2. Presiona **"Clasificar Residuo"**
3. Verás la clasificación con gráficos de probabilidad

## Otros Comandos

| Comando                      | Descripción                               |
| ---------------------------- | ----------------------------------------- |
| `docker-compose up train`    | Entrenar el modelo desde cero             |
| `docker-compose up evaluate` | Evaluar el modelo con datos de validación |
| `docker-compose up web`      | Iniciar el servidor web en puerto 3000    |
| `docker-compose down`        | Detener todos los servicios               |

## Estructura del Proyecto

```
trash-ai/
├── data/                  # Dataset TrashNet (generado al entrenar)
├── models/
│   └── trashnet_cnn_v1.keras  # Modelo entrenado (MobileNetV2)
├── src/
│   ├── model.py           # Arquitectura del modelo
│   ├── train.py           # Script de entrenamiento
│   ├── evaluate.py        # Script de evaluación
│   ├── data_loader.py     # Carga de datos
│   └── server.py          # Servidor web (Flask + TensorFlow)
├── web/
│   ├── Dockerfile         # Imagen Docker para el servidor web
│   └── public/
│       ├── index.html     # Frontend
│       ├── css/style.css  # Estilos (dark mode)
│       └── js/app.js      # Lógica del cliente + Chart.js
├── docs/
│   ├── MODEL_CARD.md      # Documentación del modelo
│   └── USER_GUIDE.md      # Guía de usuario
├── Dockerfile             # Imagen base (entrenamiento/evaluación)
├── docker-compose.yml     # Orquestación de servicios
└── requirements.txt       # Dependencias Python
```

## Categorías de Clasificación

| Categoría      | Emoji |
| -------------- | ----- |
| Cartón         | 📦    |
| Vidrio         | 🍶    |
| Metal          | 🥫    |
| Papel          | 📄    |
| Plástico       | 🧴    |
| Basura General | 🗑️    |

## Solución de Problemas

**El modelo no se encuentra:**
Asegúrate de que `models/trashnet_cnn_v1.keras` existe. Si no, ejecuta `docker-compose up train`.

**Puerto 3000 en uso:**
Cambia el puerto en `docker-compose.yml`:

```yaml
ports:
    - "8080:3000" # Accede en http://localhost:8080
```

**La build tarda mucho:**
La primera build descarga TensorFlow (~620MB). Builds posteriores usan caché de Docker y son rápidas.
