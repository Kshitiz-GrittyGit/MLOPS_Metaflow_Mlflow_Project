# Use a smaller Python base image
FROM python:3.12-slim

# Avoid prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# Install required system packages (minimized)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only necessary folders
COPY inference ./inference
COPY model ./model
COPY Requirements.txt .

# Install Python dependencies efficiently
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r Requirements.txt

# Expose FastAPI port
EXPOSE 7000

# Run FastAPI server
CMD ["uvicorn", "inference.serve:app", "--host", "0.0.0.0", "--port", "7000"]

