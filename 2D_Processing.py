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
from ultralytics import YOLO
import torch

st.set_page_config(page_title="ZERO-YOLO", layout="centered")
st.title("ZERO-YOLO: 3D Data Splitter")

# Initialize session state variables if they don't exist
if 'img_folder' not in st.session_state:
    st.session_state.img_folder = "images/"
if 'label_folder' not in st.session_state:
    st.session_state.label_folder = "masks/"
if 'output_dir' not in st.session_state:
    st.session_state.output_dir = "Dataset/"

# Step 1: Data Splitting
st.header("Step 1: For 3D Data")

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
                label_filename = filename.replace('.jpg', '.txt')
                src_label_path = os.path.join(label_folder, label_filename)
                dst_label_path = os.path.join(output_dir, 'train', 'masks', label_filename)
                shutil.copyfile(src_label_path, dst_label_path)
            for filename in test_files:
                src_img_path = os.path.join(img_folder, filename)
                dst_img_path = os.path.join(output_dir, 'test', 'images', filename)
                shutil.copyfile(src_img_path, dst_img_path)
                label_filename = filename.replace('.jpg', '.txt')
                src_label_path = os.path.join(label_folder, label_filename)
                dst_label_path = os.path.join(output_dir, 'test', 'masks', label_filename)
                shutil.copyfile(src_label_path, dst_label_path)
            for filename in val_files:
                src_img_path = os.path.join(img_folder, filename)
                dst_img_path = os.path.join(output_dir, 'val', 'images', filename)
                shutil.copyfile(src_img_path, dst_img_path)
                label_filename = filename.replace('.jpg', '.txt')
                src_label_path = os.path.join(label_folder, label_filename)
                dst_label_path = os.path.join(output_dir, 'val', 'masks', label_filename)
                shutil.copyfile(src_label_path, dst_label_path)
        try:
            split_data(img_folder, label_folder, output_dir, train_split, test_split, val_split)
            st.success(f"Data split successfully! Check the output directory: {output_dir}")
        except Exception as e:
            st.error(f"Error: {e}")

# Step 2: Process Dataset
st.header("Step 2: Process Dataset")

with st.form("process_form"):
    st.subheader("Train Data")
    train_images = st.text_input("Train Images Directory", value="/path/to/train/images")
    train_masks = st.text_input("Train Masks Directory", value="/path/to/train/masks")
    train_output = st.text_input("Train Output Directory", value="/path/to/train/output")

    st.subheader("Validation Data")
    val_images = st.text_input("Validation Images Directory", value="/path/to/val/images")
    val_masks = st.text_input("Validation Masks Directory", value="/path/to/val/masks")
    val_output = st.text_input("Validation Output Directory", value="/path/to/val/output")

    st.subheader("Test Data")
    test_images = st.text_input("Test Images Directory", value="/path/to/test/images")
    test_masks = st.text_input("Test Masks Directory", value="/path/to/test/masks")
    test_output = st.text_input("Test Output Directory", value="/path/to/test/output")

    process_submitted = st.form_submit_button("Process Dataset")

    if process_submitted:
        try:
            # Create progress bars
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process train data
            status_text.text("Processing train data...")
            process_dataset(train_images, train_masks, train_output)
            progress_bar.progress(33)
            st.success("Train data processed successfully!")

            # Process validation data
            status_text.text("Processing validation data...")
            process_dataset(val_images, val_masks, val_output)
            progress_bar.progress(66)
            st.success("Validation data processed successfully!")

            # Process test data
            status_text.text("Processing test data...")
            process_dataset(test_images, test_masks, test_output)
            progress_bar.progress(100)
            st.success("Test data processed successfully!")

            status_text.text("All data processing completed!")
            st.success("All data processing completed successfully!")
        except Exception as e:
            st.error(f"Error during processing: {e}")
            progress_bar.empty()
            status_text.empty()

# Step 3: Convert to YOLO Format
st.header("Step 3: Convert to YOLO Format")
st.write("This step converts the processed masks to YOLO format.")

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
st.header("Step 4: Create YAML File")
st.write("This step creates a YAML file for YOLO training.")

with st.form("yaml_form"):
    st.subheader("Dataset Paths")
    dataset_dir = st.text_input("Dataset Directory", value=os.getcwd())
    
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
            
            # Set up paths - using absolute paths
            train_path = os.path.abspath(os.path.join(dataset_dir, "train", "images"))
            val_path = os.path.abspath(os.path.join(dataset_dir, "val", "images"))
            test_path = os.path.abspath(os.path.join(dataset_dir, "test", "images"))
            
            # Verify paths exist
            for path in [train_path, val_path, test_path]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Directory not found: {path}")
            
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

# Step 5: YOLO Training
st.header("Step 5: YOLO Training")
st.write("Configure and start YOLO training.")

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
            
            status_text.text("Initializing training...")
            
            # Validate paths
            if not os.path.exists(yaml_path):
                raise FileNotFoundError(f"data.yaml not found at: {yaml_path}")
            
            if not os.path.exists(model_path) and model_path != "yolov8n-seg.pt":
                raise FileNotFoundError(f"Model not found at: {model_path}")
            
            # Set up device
            if device == "auto":
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
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