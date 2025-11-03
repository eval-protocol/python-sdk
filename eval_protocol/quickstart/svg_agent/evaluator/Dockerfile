FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for SVG rendering
RUN apt-get update && apt-get install -y \
    chromium \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy evaluation code
COPY . .

# Set environment for better logging
ENV PYTHONUNBUFFERED=1
