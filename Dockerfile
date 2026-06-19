# DoubleChat — container image for Google Cloud Run.
FROM python:3.12-slim

# Cleaner, faster Python in containers.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first so this layer is cached when only code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source.
COPY . .

# Cloud Run provides the port to listen on via $PORT (defaults to 8080).
ENV PORT=8080
EXPOSE 8080

# Launch Streamlit bound to all interfaces and the Cloud Run port.
# Shell form so $PORT is expanded at runtime.
CMD streamlit run app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
