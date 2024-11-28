# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    gcc \
    g++ \
    cmake \
    make \
    libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages in stages to better handle dependencies
COPY requirements.txt .

# First install core dependencies
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    elasticsearch \
    boto3 \
    python-dotenv \
    apscheduler \
    python-multipart \
    fastapi \
    uvicorn

# Install Prophet with system configuration
ENV STAN_BACKEND="CMDSTANPY"
RUN pip install --no-cache-dir prophet cmdstanpy

# Create necessary directories
RUN mkdir -p /app/logs /app/data /tmp/models

# Copy application code
COPY ./app ./app
COPY .env .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]