# Use official TensorFlow image as base (includes Python & TF pre-installed)
FROM tensorflow/tensorflow:latest

# Set working directory
WORKDIR /app

# Install additional dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY data/ ./data/
# Note: models directory is not copied to avoid overwriting; it should be a volume

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden in docker-compose)
CMD ["python", "src/train.py"]
