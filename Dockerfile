# Use build argument to determine base image
ARG USE_GPU=false

# Set base image based on USE_GPU argument
FROM python:3.11-slim

# If GPU is enabled, switch to CUDA base
RUN if [ "$USE_GPU" = "true" ] ; then \
    echo "Using GPU base image" && \
    apt-get update && \
    apt-get install -y nvidia-cuda-toolkit ; \
    fi

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Set the entrypoint
ENTRYPOINT ["streamlit", "run", "ZERO_YOLO_app.py", "--server.address", "0.0.0.0"] 