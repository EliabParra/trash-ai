FROM python:3.11-slim

# Install Node.js
RUN apt-get update && \
    apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (TF for inference)
COPY requirements.txt .
RUN pip install --no-cache-dir tensorflow numpy pillow

# Install Node dependencies
COPY web/package.json web/package-lock.json* ./web/
RUN cd web && npm install --production

# Copy application code
COPY src/ ./src/
COPY web/ ./web/

# Expose port
EXPOSE 3000

# Start the Express server
CMD ["node", "web/server.js"]
