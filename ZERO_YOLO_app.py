import streamlit as st
import os
import random
import shutil
from sklearn.model_selection import train_test_split
import m_yolo
from m_yolo.nii_to_2d import process_dataset
from m_yolo.tifffile_text import convert_masks_to_labels
import time
import pandas as pd
from pathlib import Path

import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"



def load_torch():
    """Load PyTorch and YOLO only when needed"""
    try:
        import torch
        from ultralytics import YOLO
        return True, None
    except Exception as e:
        return False, str(e)

st.set_page_config(page_title="ZERO-YOLO", layout="centered")
st.title("ZERO-YOLO: 2D Medical Image Segmentation")

# Add overall app description
st.markdown("""
### Welcome to ZERO-YOLO! 🎯
This application helps you process medical images and train a YOLO model to identify structures in your images. 
Follow the steps below to process your data and train your model.
""")

# Initialize session state variables if they don't exist
if 'img_folder' not in st.session_state:
    st.session_state.img_folder = "images/"
if 'label_folder' not in st.session_state:
    st.session_state.label_folder = "masks/"
if 'output_dir' not in st.session_state:
    st.session_state.output_dir = "Dataset/"

# make step 0: take input image and mask folder, if the file extension is not same in img_folder and label_folder, then make it same,   also make sure the names are same

from PIL import Image

from PIL import Image
import os
import streamlit as st

from PIL import Image
import os
import streamlit as st

st.header("Step 0: Prepare Image & Mask Formats")
st.markdown("""
Upload your images and masks.  
All files (.png, .jpg, .jpeg, .tif, .tiff) will be converted to `.tiff`.  
""")

with st.form("format_form"):
    img_folder_0 = st.text_input("Path to Images Folder (Step 0)", value=st.session_state.img_folder)
    mask_folder_0 = st.text_input("Path to Masks Folder (Step 0)", value=st.session_state.label_folder)
    submitted_0 = st.form_submit_button("Convert All to .tiff")

    allowed_exts = [".png", ".jpg", ".jpeg", ".tiff", ".tif"]

    def convert_folder_to_tiff(folder):
        converted = 0
        skipped = 0
        for f in os.listdir(folder):
            ext = os.path.splitext(f)[1].lower()
            if ext in allowed_exts:
                if ext == ".tiff":
                    skipped += 1
                    continue
                src_path = os.path.join(folder, f)
                dst_path = os.path.join(folder, os.path.splitext(f)[0] + ".tiff")
                try:
                    with Image.open(src_path) as im:
                        im.save(dst_path)
                    os.remove(src_path)
                    converted += 1
                    st.write(f"Converted {f} → {os.path.basename(dst_path)}")
                except Exception as e:
                    st.warning(f"Failed to convert {f}: {e}")
            else:
                skipped += 1
        return converted, skipped

    if submitted_0:
        converted_img, skipped_img = convert_folder_to_tiff(img_folder_0)
        converted_mask, skipped_mask = convert_folder_to_tiff(mask_folder_0)
        st.success(f"Images: Converted {converted_img}, Skipped {skipped_img}. "
                   f"Masks: Converted {converted_mask}, Skipped {skipped_mask}.")
        st.info("Now proceed to Step 1.")



######################################

# Step 1: Data Splitting
st.header("Step 1: For 2D Data")
st.markdown("""
### 📁 Data Organization
This step helps organize your medical images and their corresponding masks (outlines) into three groups:
- **Training set** (70%): Used to teach the model
- **Validation set** (15%): Used to check how well the model is learning
- **Test set** (15%): Used to evaluate the final model performance

Simply provide:
1. Where your images are stored
2. Where your masks are stored
3. Where you want the organized data to be saved

The app will automatically split and organize everything for you! 🎯
""")

with st.form("split_form"):
    img_folder = st.text_input("Path to Images Folder", value=st.session_state.img_folder)
    label_folder = st.text_input("Path to Masks Folder", value=st.session_state.label_folder)
    output_dir = st.text_input("Output Directory", value=st.session_state.output_dir)
    
    

    col1, col2, col3 = st.columns(3)
    with col1:
        train_split = st.number_input("Train Ratio", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    with col2:
        val_split = st.number_input("Validation Ratio", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
    with col3:
        test_split = st.number_input("Test Ratio", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
    
    submitted = st.form_submit_button("Split Data")

    # check if file extension is same in img_folder and label_folder
    

    if submitted:
        
        def split_data(img_folder, label_folder, output_dir, train_split, test_split, val_split):
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'train', 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'train', 'masks'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'test', 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'test', 'masks'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'val', 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'val', 'masks'), exist_ok=True)
            img_files = os.listdir(img_folder)
            random_state = random.randint(0, 1000)
            train_files, test_val_files = train_test_split(img_files, test_size=test_split + val_split, random_state=random_state)
            test_files, val_files = train_test_split(test_val_files, test_size=val_split / (test_split + val_split), random_state=random_state)
            for filename in train_files:
                
                

                src_img_path = os.path.join(img_folder, filename)
                dst_img_path = os.path.join(output_dir, 'train', 'images', filename)
                shutil.copyfile(src_img_path, dst_img_path)
                #label_filename = filename.replace('.jpg', '.txt')
                label_filename = filename
                src_label_path = os.path.join(label_folder, label_filename)
                dst_label_path = os.path.join(output_dir, 'train', 'masks', label_filename)
                shutil.copyfile(src_label_path, dst_label_path)
            for filename in test_files:
                src_img_path = os.path.join(img_folder, filename)
                dst_img_path = os.path.join(output_dir, 'test', 'images', filename)
                shutil.copyfile(src_img_path, dst_img_path)
                #label_filename = filename.replace('.jpg', '.txt')
                label_filename = filename
                src_label_path = os.path.join(label_folder, label_filename)
                dst_label_path = os.path.join(output_dir, 'test', 'masks', label_filename)
                shutil.copyfile(src_label_path, dst_label_path)
            for filename in val_files:
                src_img_path = os.path.join(img_folder, filename)
                dst_img_path = os.path.join(output_dir, 'val', 'images', filename)
                shutil.copyfile(src_img_path, dst_img_path)
                #label_filename = filename.replace('.jpg', '.txt')
                label_filename = filename
                src_label_path = os.path.join(label_folder, label_filename)
                dst_label_path = os.path.join(output_dir, 'val', 'masks', label_filename)
                shutil.copyfile(src_label_path, dst_label_path)
        try:
            split_data(img_folder, label_folder, output_dir, train_split, test_split, val_split)
            st.success(f"Data split successfully! Check the output directory: {output_dir}")
        except Exception as e:
            st.error(f"Error: {e}")



# Step 3: Convert to YOLO Format
st.header("Step 2: Convert to YOLO Format")
st.markdown("""
### 🔄 Format Conversion
This step converts your medical image masks into a format that the YOLO model can understand.
Think of it as translating your medical annotations into a language the computer can read.

You need to specify:
1. Where your masks are stored for each set (train/validation/test)
2. Where you want the converted files to be saved
3. The class IDs (numbers that represent different types of medical structures)

The app will handle the conversion automatically! 🎯
""")

with st.form("yolo_form"):
    st.subheader("Train Data")
    train_mask_dir = st.text_input("Train Masks Directory", value="/path/to/train/masks")
    train_label_dir = st.text_input("Train Labels Output Directory", value="/path/to/train/labels")

    st.subheader("Validation Data")
    val_mask_dir = st.text_input("Validation Masks Directory", value="/path/to/val/masks")
    val_label_dir = st.text_input("Validation Labels Output Directory", value="/path/to/val/labels")

    st.subheader("Test Data")
    test_mask_dir = st.text_input("Test Masks Directory", value="/path/to/test/masks")
    test_label_dir = st.text_input("Test Labels Output Directory", value="/path/to/test/labels")

    # Class mapping configuration
    st.subheader("Class Mapping")
    start_class = st.number_input("Start Class ID", min_value=1, value=1)
    end_class = st.number_input("End Class ID", min_value=1, value=199999)
    
    yolo_submitted = st.form_submit_button("Convert to YOLO Format")

    if yolo_submitted:
        try:
            # Create progress bars
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Create class mapping
            class_mapping = {i: i - 1 for i in range(start_class, end_class + 1)}

            # Process train data
            status_text.text("Converting train masks to YOLO format...")
            os.makedirs(train_label_dir, exist_ok=True)
            convert_masks_to_labels(
                input_mask_dir=train_mask_dir,
                output_label_dir=train_label_dir,
                class_map=class_mapping
            )
            progress_bar.progress(33)
            st.success("Train data converted successfully!")

            # Process validation data
            status_text.text("Converting validation masks to YOLO format...")
            os.makedirs(val_label_dir, exist_ok=True)
            convert_masks_to_labels(
                input_mask_dir=val_mask_dir,
                output_label_dir=val_label_dir,
                class_map=class_mapping
            )
            progress_bar.progress(66)
            st.success("Validation data converted successfully!")

            # Process test data
            status_text.text("Converting test masks to YOLO format...")
            os.makedirs(test_label_dir, exist_ok=True)
            convert_masks_to_labels(
                input_mask_dir=test_mask_dir,
                output_label_dir=test_label_dir,
                class_map=class_mapping
            )
            progress_bar.progress(100)
            st.success("Test data converted successfully!")

            status_text.text("All conversions completed!")
            st.success("All masks converted to YOLO format successfully!")
        except Exception as e:
            st.error(f"Error during conversion: {e}")
            progress_bar.empty()
            status_text.empty()

# Step 4: Create YAML File
st.header("Step 3: Create YAML File")
st.markdown("""
### ⚙️ Configuration Setup
This step creates a configuration file that tells the YOLO model:
1. Where to find all the training data
2. What each class (medical structure) is called
3. How to modify images during training to make the model more robust

#### Required Excel File Format
Upload an Excel file with two columns:
- `label_names`: Names of the medical structures
- `multiplied_labels`: Corresponding class IDs

Example format:
| label_names | multiplied_labels |
|-------------|------------------|
| femoral lateral | 1 |
| Femoral Medial | 2 |
| Patellar | 3 |
| Tibia | 4 |
| Tibia Lateral | 5 |
| Tibia Medial | 6 |

You can:
- Upload your Excel file with the names of your medical structures
- Adjust various parameters that help the model learn better
- Save these settings for future use

The app will create a YAML file with all your settings! 🎯
""")

with st.form("yaml_form"):
    st.subheader("Dataset Paths")
    dataset_dir = st.text_input("Dataset Directory", value=os.getcwd(), help="Enter the full path to your dataset directory")
    
    st.subheader("Label Names")
    excel_file = st.file_uploader("Upload Excel File with Label Names", type=['xlsx'])
    
    st.subheader("Hyperparameters")
    col1, col2 = st.columns(2)
    with col1:
        degrees = st.number_input("Image Rotation (+/- deg)", value=0.5)
        translate = st.number_input("Image Translation (+/- fraction)", value=0.1)
        scale = st.number_input("Image Scale (+/- gain)", value=0.2)
        shear = st.number_input("Image Shear (+/- deg)", value=0.2)
    with col2:
        perspective = st.number_input("Image Perspective (+/- fraction)", value=0.1)
        flipud = st.number_input("Image Flip Up-Down (probability)", value=0.7)
        fliplr = st.number_input("Image Flip Left-Right (probability)", value=0.5)
        mosaic = st.number_input("Image Mosaic (probability)", value=0.8)
        mixup = st.number_input("Image Mixup (probability)", value=0.1)
    
    yaml_submitted = st.form_submit_button("Create YAML File")

    if yaml_submitted:
        try:
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Creating YAML file...")
            
            # Ensure dataset_dir exists
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir, exist_ok=True)
                st.info(f"Created dataset directory: {dataset_dir}")
            
            # Create necessary directories
            train_path = os.path.join(dataset_dir, "train", "images")
            val_path = os.path.join(dataset_dir, "val", "images")
            test_path = os.path.join(dataset_dir, "test", "images")
            
            # Create directories if they don't exist
            os.makedirs(train_path, exist_ok=True)
            os.makedirs(val_path, exist_ok=True)
            os.makedirs(test_path, exist_ok=True)
            
            # Get absolute paths
            train_path = os.path.abspath(train_path)
            val_path = os.path.abspath(val_path)
            test_path = os.path.abspath(test_path)
            
            # Generate YAML content
            yaml_content = f'''train: {train_path}
val: {val_path}
test: {test_path}

names:
'''
            
            # Add label names from Excel file
            if excel_file is not None:
                data = pd.read_excel(excel_file)
                for i, name in enumerate(data['label_names']):
                    yaml_content += f"    {i}: '{name.split('.')[0]}'\n"
            else:
                st.warning("No Excel file uploaded. Using default class names.")
                yaml_content += "    0: 'class_0'\n"
            
            # Add hyperparameters
            yaml_content += f'''
# Hyperparameters
degrees: {degrees}  # image rotation (+/- deg)
translate: {translate}  # image translation (+/- fraction)
scale: {scale}  # image scale (+/- gain)
shear: {shear}  # image shear (+/- deg)
perspective: {perspective}  # image perspective (+/- fraction)
flipud: {flipud}  # image flip up-down (probability)
fliplr: {fliplr}  # image flip left-right (probability)
mosaic: {mosaic}  # image mosaic (probability)
mixup: {mixup}  # image mixup (probability)
'''
            
            # Save YAML file
            yaml_path = os.path.join(dataset_dir, 'data.yaml')
            with open(yaml_path, 'w') as file:
                file.write(yaml_content)
            
            progress_bar.progress(100)
            status_text.text("YAML file created successfully!")
            st.success(f"YAML file created at: {yaml_path}")
            
            # Display the YAML content
            st.code(yaml_content, language='yaml')
            
        except Exception as e:
            st.error(f"Error creating YAML file: {e}")
            progress_bar.empty()
            status_text.empty()

# Step 4: YOLO Training
st.header("Step 4: YOLO Training")
st.markdown("""
### 🎓 Model Training
This is where the actual learning happens! The model will:
1. Look at your medical images
2. Learn to identify different structures
3. Improve its accuracy over time

You can choose:
- Which model to start with (like choosing a pre-trained doctor)
- How many images to process at once
- How long to train
- What size images to use
- Whether to use CPU or GPU for faster training

The app will show you the training progress and results! 🎯
""")

with st.form("training_form"):
    st.subheader("Model Configuration")
    
    # Model path input
    model_path = st.text_input(
        "YOLO Model Path",
        value="yolov8n-seg.pt",
        help="Path to your YOLO model (.pt file). Use 'yolov8n-seg.pt' for latest segmentation model."
    )
    
    # Data configuration
    st.subheader("Data Configuration")
    yaml_path = st.text_input(
        "Path to data.yaml",
        value=os.path.join(os.getcwd(), 'data.yaml'),
        help="Path to your data.yaml file"
    )
    
    st.subheader("Training Parameters")
    col1, col2 = st.columns(2)
    with col1:
        batch_size = st.number_input("Batch Size", min_value=1, value=10)
        epochs = st.number_input("Epochs", min_value=1, value=500)
        image_size = st.number_input("Image Size", min_value=32, value=256, step=32)
    with col2:
        freeze_layers = st.number_input("Freeze Layers", min_value=0, value=0)
        device = st.selectbox(
            "Device",
            ["auto", "cpu", "cuda"],
            help="Select device for training (auto will use CUDA if available)"
        )
    
    train_submitted = st.form_submit_button("Start Training")

    if train_submitted:
        try:
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Initializing PyTorch and YOLO...")
            
            # Load PyTorch and YOLO
            success, error = load_torch()
            if not success:
                raise RuntimeError(f"Failed to initialize PyTorch: {error}")
            
            import torch
            from ultralytics import YOLO
            
            # Validate paths
            if not os.path.exists(yaml_path):
                raise FileNotFoundError(f"data.yaml not found at: {yaml_path}")
            
            if not os.path.exists(model_path) and model_path != "yolov8n-seg.pt":
                raise FileNotFoundError(f"Model not found at: {model_path}")
            
            # Set up device
            if device == "auto":
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            status_text.text("Loading model...")
            progress_bar.progress(10)
            
            # Load model
            model = YOLO(model_path).to(device)
            
            progress_bar.progress(20)
            status_text.text("Model loaded successfully. Starting training...")
            
            # Start training
            results = model.train(
                batch=batch_size,
                device=device,
                data=yaml_path,
                epochs=epochs,
                imgsz=image_size,
                freeze=freeze_layers
            )
            
            progress_bar.progress(100)
            status_text.text("Training completed successfully!")
            st.success("Training completed! Check the 'runs' directory for results.")
            
            # Display training results
            st.subheader("Training Results")
            st.write(f"Best mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")
            st.write(f"Best mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")
            
        except Exception as e:
            st.error(f"Error during training: {e}")
            progress_bar.empty()
            status_text.empty()

# Step 5: Generate Masks
st.header("Step 5: Generate Masks")
st.markdown("""
### 🎨 Mask Generation
After training, this step lets you use your trained model to:
1. Look at new medical images
2. Automatically identify and outline the structures it learned about
3. Save these outlines (masks) for future use

You can:
- Process both training and validation images
- See how well the model works on different images
- Save the generated masks for further analysis

The app will show you the progress and save the results! 🎯
""")

with st.form("mask_generation_form"):
    st.subheader("Model and Data Configuration")
    
    # Model path input
    model_path = st.text_input(
        "Path to Trained Model",
        value="/path/to/your/model.pt",
        help="Path to your trained YOLO model weights (.pt file)"
    )
    
    # Directory configuration
    st.subheader("Directory Configuration")
    col1, col2 = st.columns(2)
    with col1:
        train_image_dir = st.text_input(
            "Train Images Directory",
            value="dataset/train/images",
            help="Directory containing training images"
        )
        train_output_dir = st.text_input(
            "Train Output Directory",
            value="dataset/train/predicted_masks",
            help="Directory to save generated masks for training images"
        )
    with col2:
        val_image_dir = st.text_input(
            "Validation Images Directory",
            value="dataset/test/images",
            help="Directory containing validation/test images"
        )
        val_output_dir = st.text_input(
            "Validation Output Directory",
            value="dataset/test/predicted_masks",
            help="Directory to save generated masks for validation/test images"
        )
    
    generate_submitted = st.form_submit_button("Generate Masks")

    if generate_submitted:
        try:
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Initializing mask generation...")
            
            # Load PyTorch and YOLO
            success, error = load_torch()
            if not success:
                raise RuntimeError(f"Failed to initialize PyTorch: {error}")
            
            # Validate paths
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at: {model_path}")
            
            # Create output directories
            os.makedirs(train_output_dir, exist_ok=True)
            os.makedirs(val_output_dir, exist_ok=True)
            
            status_text.text("Starting mask generation...")
            progress_bar.progress(20)
            
            # Import and run mask generation
            from m_yolo.predict_yolo import generate_masks
            
            # Generate masks
            generate_masks(
                model_path=model_path,
                train_image_dir=train_image_dir,
                val_image_dir=val_image_dir,
                train_output_dir=train_output_dir,
                val_output_dir=val_output_dir
            )
            
            progress_bar.progress(100)
            status_text.text("Mask generation completed successfully!")
            st.success("Masks generated successfully!")
            
            # Display completion message with paths
            st.info(f"""
            Masks have been generated and saved to:
            - Training masks: {train_output_dir}
            - Validation/Test masks: {val_output_dir}
            """)


            # Create progress bar
        except Exception as e:
            st.error(f"Error during mask generation: {e}")
            progress_bar.empty()
            status_text.empty()

# Step 6: Data Visualization
st.header("Step 6: Data Visualization")
st.markdown("""
### 👀 Results Visualization
This is your demonstration step! You can:
1. Pick any medical image
2. See what structures the model identifies
3. View:
   - The original image
   - The predicted outlines
   - An overlay of both
4. See how confident the model is about its predictions

This helps you:
- Check if the model is working correctly
- Understand what the model sees
- Identify any areas for improvement

The app will show you clear visualizations of the results! 🎯
""")

with st.form("visualization_form"):
    st.subheader("Model and Image Selection")
    
    # Model path input
    viz_model_path = st.text_input(
        "Path to Trained Model",
        value="/home/sumit-pandey/Documents/MED-YOLO/knee/runs/segment/train4/weights/best.pt",
        help="Path to your trained YOLO model weights (.pt file)"
    )
    
    # Image selection with absolute path
    default_image_dir = os.path.abspath(os.path.join(os.getcwd(), "knee/Dataset_2d/train/images"))
    image_dir = st.text_input(
        "Images Directory",
        value=default_image_dir,
        help="Full path to directory containing images to visualize"
    )
    
    # Get list of images in directory
    try:
        # Convert to absolute path and normalize
        image_dir = os.path.abspath(os.path.expanduser(image_dir))
        st.info(f"Looking for images in: {image_dir}")
        
        if os.path.isdir(image_dir):
            image_files = [f for f in os.listdir(image_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
            if image_files:
                selected_image = st.selectbox(
                    "Select an image",
                    image_files,
                    help="Choose an image to visualize"
                )
                st.success(f"Found {len(image_files)} images in directory")
            else:
                st.warning(f"No images found in directory: {image_dir}")
                selected_image = None
        else:
            st.error(f"Not a valid directory: {image_dir}")
            st.info("Please enter a valid directory path")
            selected_image = None
    except Exception as e:
        st.error(f"Error accessing directory: {str(e)}")
        st.info("Please check if the directory exists and you have proper permissions")
        selected_image = None
    
    visualize_submitted = st.form_submit_button("Visualize Prediction")

    if visualize_submitted and selected_image:
        try:
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Loading model and processing image...")
            
            # Load PyTorch and YOLO
            success, error = load_torch()
            if not success:
                raise RuntimeError(f"Failed to initialize PyTorch: {error}")
            
            import torch
            from ultralytics import YOLO
            import cv2
            import numpy as np
            from PIL import Image
            import tifffile  # Added for TIFF support
            
            # Validate model path
            if not os.path.exists(viz_model_path):
                raise FileNotFoundError(f"Model not found at: {viz_model_path}")
            
            # Load model
            model = YOLO(viz_model_path)
            progress_bar.progress(30)
            
            # Load and process image
            image_path = os.path.join(image_dir, selected_image)
            
            # Handle different image formats
            if selected_image.lower().endswith(('.tiff', '.tif')):
                img = tifffile.imread(image_path)
                # Normalize TIFF image to 0-255 range
                if img.dtype != np.uint8:
                    img = ((img - img.min()) * (255.0 / (img.max() - img.min()))).astype(np.uint8)
                if len(img.shape) == 2:  # If grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img = cv2.imread(image_path)
                
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Ensure image is in uint8 format
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            
            # Get prediction
            results = model(img)
            progress_bar.progress(60)
            
            # Process results
            if len(results) > 0:
                result = results[0]
                
                # Get original image and ensure it's in the correct format
                if len(img.shape) == 3 and img.shape[2] == 3:
                    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    original_img = img
                
                # Normalize original image to 0-255 range if needed
                if original_img.dtype != np.uint8:
                    original_img = ((original_img - original_img.min()) * (255.0 / (original_img.max() - original_img.min()))).astype(np.uint8)
                
                # Get mask
                if hasattr(result, 'masks') and result.masks is not None:
                    # Get the original image dimensions
                    h, w = original_img.shape[:2]
                    
                    # Create a blank mask with original image dimensions
                    mask = np.zeros((h, w), dtype=np.uint8)
                    
                    # Function to generate distinct colors
                    def generate_distinct_colors(n_colors):
                        colors = []
                        # Use HSV color space for better color distribution
                        for i in range(n_colors):
                            # Generate hue (0-1) with good spacing
                            hue = i / n_colors
                            # Use high saturation and value for visibility
                            saturation = 0.8
                            value = 0.9
                            # Convert HSV to BGR
                            hsv = np.uint8([[[hue * 180, saturation * 255, value * 255]]])
                            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
                            colors.append(bgr.tolist())
                        return colors
                    
                    # Get number of masks and generate colors
                    n_masks = len(result.masks.data)
                    mask_colors = generate_distinct_colors(min(n_masks, 1000))  # Limit to 1000 colors
                    
                    # Create a colored mask for visualization
                    mask_rgb = np.zeros_like(original_img)
                    
                    # Process each mask
                    for idx, m in enumerate(result.masks.data):
                        if idx >= 1000:  # Skip if more than 1000 masks
                            st.warning(f"Only showing first 1000 masks. Total masks: {n_masks}")
                            break
                                
                        # Resize mask to match original image dimensions
                        m = m.cpu().numpy()
                        if m.shape != (h, w):
                            m = cv2.resize(m.astype(float), (w, h), interpolation=cv2.INTER_NEAREST)
                        # Ensure mask is binary (0 or 1)
                        m = (m > 0.5).astype(np.uint8)
                        
                        # Apply color based on index
                        color = mask_colors[idx]
                        mask_rgb[m > 0] = color
                        
                        # Update combined mask
                        mask = np.logical_or(mask, m).astype(np.uint8)
                    
                    # Create overlay with better transparency
                    overlay = original_img.copy()
                    # Ensure all arrays have the same shape and type
                    if len(overlay.shape) == 2:  # If grayscale
                        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
                    
                    # Create overlay with proper normalization and transparency
                    overlay = overlay.astype(float) / 255.0
                    mask_rgb = mask_rgb.astype(float) / 255.0
                    # Use 0.7 for original image and 0.3 for mask to make mask more visible
                    overlay[mask > 0] = overlay[mask > 0] * 0.7 + mask_rgb[mask > 0] * 0.3
                    overlay = (overlay * 255.0).astype(np.uint8)
                    
                    # Display images with updated captions
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.image(original_img, caption="Original Image", use_column_width=True)
                    with col2:
                        st.image(mask_rgb, caption=f"Predicted Masks ({n_masks} unique colors)", use_column_width=True)
                    with col3:
                        st.image(overlay, caption="Overlay (30% Color-coded Mask)", use_column_width=True)
                    
                    # Display mask information
                    st.info(f"""
                    ### Mask Information:
                    - Total number of masks: {n_masks}
                    - Each mask has a unique color
                    - Colors are distributed across the spectrum for maximum distinction
                    - Maximum display limit: 1000 masks
                    """)
                    
                    # Display prediction metrics if available
                    if hasattr(result, 'boxes'):
                        st.subheader("Detection Metrics")
                        metrics = {
                            "Number of detections": len(result.boxes),
                            "Confidence scores": [f"{score:.2f}" for score in result.boxes.conf.cpu().numpy()]
                        }
                        st.json(metrics)
                else:
                    st.warning("No masks detected in the image")
            else:
                st.warning("No predictions made for the image")
            
            progress_bar.progress(100)
            status_text.text("Visualization completed!")
            
        except Exception as e:
            st.error(f"Error during visualization: {e}")
            progress_bar.empty()
            status_text.empty()

# Step 7: Evaluation Metrics
st.header("Step 7: Evaluation Metrics")
st.markdown("""
### 📊 Performance Evaluation
This step helps you evaluate how well your model is performing by comparing:
- Predicted masks (what the model generated)
- Original masks (ground truth)

The app will calculate and show you:
- **Dice Score**: How well the predicted masks match the original masks (0-1, higher is better)
- **Sensitivity**: How well the model detects the structures (0-1, higher is better)
- **Specificity**: How well the model avoids false positives (0-1, higher is better)
- **IoU (Intersection over Union)**: Overall overlap between predicted and original masks (0-1, higher is better)

Simply provide the paths to your predicted and original masks, and the app will do the calculations! 🎯
""")

with st.form("evaluation_form"):
    st.subheader("Directory Configuration")
    
    # Input directories
    col1, col2 = st.columns(2)
    with col1:
        predicted_masks_dir = st.text_input(
            "Predicted Masks Directory",
            value="dataset/predicted_masks",
            help="Directory containing masks generated by the model"
        )
    with col2:
        original_masks_dir = st.text_input(
            "Original Masks Directory",
            value="dataset/original_masks",
            help="Directory containing the ground truth masks"
        )
    
    evaluate_submitted = st.form_submit_button("Calculate Metrics")

    if evaluate_submitted:
        try:
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Calculating evaluation metrics...")
            
            import numpy as np
            from pathlib import Path
            import cv2
            from tqdm import tqdm
            
            def calculate_metrics(pred_mask, true_mask):
                """Calculate evaluation metrics for a single mask pair"""
                # Ensure masks are binary
                pred_mask = (pred_mask > 0).astype(np.uint8)
                true_mask = (true_mask > 0).astype(np.uint8)
                
                # Calculate intersection and union
                intersection = np.logical_and(pred_mask, true_mask).sum()
                union = np.logical_or(pred_mask, true_mask).sum()
                
                # Calculate metrics
                dice = (2. * intersection) / (pred_mask.sum() + true_mask.sum() + 1e-6)
                iou = intersection / (union + 1e-6)
                
                # Calculate sensitivity (recall) and specificity
                tp = intersection
                fp = pred_mask.sum() - intersection
                fn = true_mask.sum() - intersection
                tn = (pred_mask.shape[0] * pred_mask.shape[1]) - (tp + fp + fn)
                
                sensitivity = tp / (tp + fn + 1e-6)
                specificity = tn / (tn + fp + 1e-6)
                
                return {
                    'dice': dice,
                    'iou': iou,
                    'sensitivity': sensitivity,
                    'specificity': specificity
                }
            
            # Get list of mask files
            pred_path = Path(predicted_masks_dir)
            orig_path = Path(original_masks_dir)
            
            pred_files = sorted([f for f in pred_path.glob('*') if f.suffix.lower() in ['.png', '.jpg', '.tif', '.tiff']])
            orig_files = sorted([f for f in orig_path.glob('*') if f.suffix.lower() in ['.png', '.jpg', '.tif', '.tiff']])
            
            if not pred_files or not orig_files:
                raise ValueError("No mask files found in one or both directories")
            
            # Initialize metrics storage
            all_metrics = []
            
            # Process each pair of masks
            for pred_file, orig_file in tqdm(zip(pred_files, orig_files), total=len(pred_files)):
                # Read masks
                pred_mask = cv2.imread(str(pred_file), cv2.IMREAD_GRAYSCALE)
                orig_mask = cv2.imread(str(orig_file), cv2.IMREAD_GRAYSCALE)
                
                if pred_mask is None or orig_mask is None:
                    st.warning(f"Could not read masks for {pred_file.name}")
                    continue
                
                # Ensure same size
                if pred_mask.shape != orig_mask.shape:
                    pred_mask = cv2.resize(pred_mask, (orig_mask.shape[1], orig_mask.shape[0]))
                
                # Calculate metrics
                metrics = calculate_metrics(pred_mask, orig_mask)
                all_metrics.append(metrics)
                
                progress_bar.progress(len(all_metrics) / len(pred_files))
            
            # Calculate average metrics
            avg_metrics = {
                'Metric': ['Dice Score', 'IoU', 'Sensitivity', 'Specificity'],
                'Average Value': [
                    np.mean([m['dice'] for m in all_metrics]),
                    np.mean([m['iou'] for m in all_metrics]),
                    np.mean([m['sensitivity'] for m in all_metrics]),
                    np.mean([m['specificity'] for m in all_metrics])
                ],
                'Standard Deviation': [
                    np.std([m['dice'] for m in all_metrics]),
                    np.std([m['iou'] for m in all_metrics]),
                    np.std([m['sensitivity'] for m in all_metrics]),
                    np.std([m['specificity'] for m in all_metrics])
                ]
            }
            
            # Create DataFrame and display
            import pandas as pd
            df = pd.DataFrame(avg_metrics)
            
            # Format the values to 4 decimal places
            df['Average Value'] = df['Average Value'].map('{:.4f}'.format)
            df['Standard Deviation'] = df['Standard Deviation'].map('{:.4f}'.format)
            
            # Display results
            st.success("Metrics calculated successfully!")
            st.dataframe(df, use_container_width=True)
            
            # Add explanation of metrics
            st.markdown("""
            ### Understanding the Metrics:
            - **Dice Score**: Measures overlap between predicted and original masks (0-1)
              - 1.0 = perfect match
              - 0.0 = no overlap
            - **IoU (Intersection over Union)**: Similar to Dice score but more strict
              - 1.0 = perfect overlap
              - 0.0 = no overlap
            - **Sensitivity**: How well the model detects the structures
              - 1.0 = detects all structures
              - 0.0 = misses all structures
            - **Specificity**: How well the model avoids false positives
              - 1.0 = no false positives
              - 0.0 = all predictions are false positives
            """)
            
            progress_bar.progress(100)
            status_text.text("Evaluation completed!")
            
        except Exception as e:
            st.error(f"Error during evaluation: {e}")
            progress_bar.empty()
            status_text.empty() 
