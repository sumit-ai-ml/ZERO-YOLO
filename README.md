# ZERO-YOLO: A No-Code GUI for Training Custom YOLO Segmentation Models 🏥

## What is ZERO-YOLO? 🤔
ZERO-YOLO is a user-friendly tool that enables enthusiasts, engineers, and researchers to train YOLO models on their own private segmentation dataset. It streamlines the entire workflow—from data preprocessing and preparation to model training and result analysis.

## Tutorial 
[![Watch the video](https://img.youtube.com/vi/L1_IRtNpP4Y/0.jpg)](https://www.youtube.com/watch?v=L1_IRtNpP4Y)


## Features ✨
- Easy-to-use web interface
- No technical knowledge required
- Works on both regular computers and computers with or without GPUs. 
- Shows results in real-time

## Before You Start 📋
You'll need:
1. A computer running Windows, Mac, or Linux
2. Docker installed on your computer (we'll help you install it)
3. (Optional) A GPU (NVIDIA) for faster processing

## Step-by-Step Installation Guide 🚀

### Step 1: Install Docker
1. Go to [Docker's website](https://www.docker.com/products/docker-desktop)
2. Click "Download Docker Desktop"
3. Run the installer
4. Follow the installation wizard
5. Restart your computer when asked

### Step 2: Download ZERO-YOLO
1. Click the green "Code" button at the top of this page
2. Click "Download ZIP"
3. Extract the ZIP file to a location you can easily find (like your Desktop)

### Step 3: Open Terminal/Command Prompt
- **Windows**: 
  - Press `Windows + R`
  - Type `cmd` and press Enter
- **Mac**: 
  - Press `Command + Space`
  - Type `Terminal` and press Enter
- **Linux**: 
  - Press `Ctrl + Alt + T`

### Step 4: Navigate to ZERO-YOLO
In the terminal, type:
```bash
# If you saved it to Desktop (Windows)
cd Desktop/ZERO-YOLO

# If you saved it to Desktop (Mac/Linux)
cd ~/Desktop/ZERO-YOLO
```

### Step 5: Run ZERO-YOLO
Choose one of these options based on your computer:

#### Option A: If you have a graphics card (NVIDIA)
```bash
docker compose --profile gpu up --build
```
Then open: http://localhost:8501 in your web browser

#### Option B: If you don't have a graphics card
```bash
docker compose --profile cpu up --build
```
Then open: http://localhost:8501 in your web browser

## How to Use ZERO-YOLO 🎯

### Step 1: Prepare Your Images
1. Create a folder named `data` in the ZERO-YOLO folder
2. Create two folders inside `data`:
   a. _'image'_: contains all images.
   b. _'mask'_: contains all segmentation masks.
4. Put your images in the `input` folder
   - Supported formats:  TIFF, PNG, JPG

### Step 2: Use the Web Interface
1. Open your web browser
2. Go to:
   - http://localhost:8501 (For web app)


## Common Problems and Solutions 🔧

### Problem: "Docker not found"
Solution: Make sure Docker is installed and running. Try restarting your computer. 

### Prroblem: 

### Problem: "Port already in use"
Solution: 
1. Close any other applications that might be using the ports
2. Or try these commands:
```bash
docker compose down
docker compose --profile cpu up --build  # for CPU version
# OR
docker compose --profile gpu up --build  # for GPU version
```

## Play with sample dataset. 
With this repo you have sample images in `data.zip` unzip it and start playing with it. For labels we have provided an excel file `label_names.xlsx`.  

## Need Help? 🤝
If you run into any problems:
1. Check the "Common Problems" section above
2. Look for error messages in the terminal
3. Create an issue on our GitHub page
4. Contact us for support

## Tips and Tricks 💡
- Keep your images organized in the `data/input` folder
- Use clear names for your image files
- Save your results regularly
- If the program is slow, try using the GPU version if available

## Want to Learn More? 📚
- Visit our [GitHub page](https://github.com/sumit-ai-ml/ZEROYOLO) for more information
- Check out our [documentation](https://github.com/sumit-ai-ml/ZEROYOLO/wiki) for advanced features
- Join our community for updates and support

## Thank You! 🙏
Thank you for using ZERO-YOLO! We hope it helps you in your ical image analysis work.

--------


