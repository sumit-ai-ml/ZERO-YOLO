#!/bin/bash

echo "Installing MED-YOLO..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first:"
    echo "Visit: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Create necessary directories
mkdir -p data/input data/output

# Pull the Docker image
echo "Downloading MED-YOLO..."
docker pull sumit-ai-ml/med-yolo:latest

# Create a simple run script
cat > run-med-yolo.sh << 'EOL'
#!/bin/bash

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected. Starting GPU version..."
    docker compose --profile gpu up
    echo "Open http://localhost:8502 in your web browser"
else
    echo "No GPU detected. Starting CPU version..."
    docker compose --profile cpu up
    echo "Open http://localhost:8501 in your web browser"
fi
EOL

# Make the run script executable
chmod +x run-med-yolo.sh

echo "Installation complete!"
echo "To run MED-YOLO, use: ./run-med-yolo.sh"
echo "For more information, visit: https://github.com/sumit-ai-ml/MED-YOLO" 