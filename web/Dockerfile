FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies only
RUN pip install --no-cache-dir tensorflow numpy pillow flask

# Copy application code
COPY src/ ./src/
COPY web/public/ ./web/public/

EXPOSE 3000

CMD ["python", "src/server.py"]
