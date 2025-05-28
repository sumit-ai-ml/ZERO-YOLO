import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

def ensure_dir(directory):
    """Ensure that a directory exists."""
    os.makedirs(directory, exist_ok=True)

def load_model(model_path):
    """
    Load the YOLO segmentation model.
    
    Args:
        model_path (str): Path to the YOLO model weights.
    
    Returns:
        YOLO: Loaded YOLO model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return YOLO(model_path)

def get_image_files(image_dir, extensions=('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')):
    """
    Retrieve a list of image file paths from a directory.
    
    Args:
        image_dir (str): Directory containing images.
        extensions (tuple): File extensions to include.
    
    Returns:
        list: List of image file paths.
    """
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(image_dir, ext)))
    return sorted(files)

def process_images(model, image_dir, output_dir, image_extension_replace='.tif'):
    """
    Process images to generate and save predicted masks.
    
    Args:
        model (YOLO): Loaded YOLO model.
        image_dir (str): Directory containing input images.
        output_dir (str): Directory to save predicted masks.
        image_extension_replace (str): Extension to replace in output masks.
    
    Returns:
        None
    """
    ensure_dir(output_dir)
    image_files = get_image_files(image_dir)
    
    if not image_files:
        print(f"No images found in {image_dir}.")
        return
    
    for image_path in image_files:
        # Read the image using OpenCV
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Failed to read {image_path}. Skipping.")
            continue

        # Perform segmentation prediction
        try:
            results = model.predict(image, save=False)  # Set save=False to prevent automatic saving
        except Exception as e:
            print(f"Prediction failed for {image_path}: {e}. Skipping.")
            continue

        if not results:
            print(f"No results for {image_path}. Skipping.")
            continue

        # Assuming single image per prediction; adjust if batching
        result = results[0]

        # Check if masks are available in the result
        if not hasattr(result, 'masks') or result.masks is None:
            print(f"No masks found for {image_path}. Skipping.")
            continue

        # Get the mask tensor and class labels
        masks = result.masks.data.cpu().numpy()  # Shape: [num_instances, mask_height, mask_width]
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs for each instance

        if masks.size == 0:
            print(f"No masks detected in {image_path}. Skipping.")
            continue

        # Initialize an empty mask image with zeros (background)
        height, width = image.shape[0], image.shape[1]
        combined_mask = np.zeros((height, width), dtype=np.uint8)

        for idx, mask in enumerate(masks):
            class_id = int(class_ids[idx])  # Ensure class_id is an integer
            mask_value = class_id + 1  # Add 1 to make classes start from 1

            # Resize the mask to match the original image size
            resized_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # Create a binary mask for the current instance
            binary_mask = (resized_mask > 0.5).astype(np.uint8)

            # Assign the class value to the mask where binary_mask is 1
            # Overwrite with the latest mask
            combined_mask[binary_mask == 1] = mask_value

            # If you prefer to keep the first mask's class in overlapping areas, use:
            # combined_mask = np.where((combined_mask == 0) & (binary_mask == 1), mask_value, combined_mask)

        # Optionally, apply some post-processing (e.g., dilation) to enhance mask edges
        # Example: combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

        # Debugging information
        unique_values = np.unique(combined_mask)
        print(f"Processed {os.path.basename(image_path)}: Unique mask values {unique_values}, Shape {combined_mask.shape}")
        
        # Define the output mask path
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        mask_output_path = os.path.join(output_dir, f"{base_name}{image_extension_replace}")
        
        # Save the mask as an image
        success = cv2.imwrite(mask_output_path, combined_mask)
        if not success:
            print(f"Failed to save mask for {image_path} at {mask_output_path}.")

def generate_masks(model_path, train_image_dir, val_image_dir, 
                  train_output_dir, val_output_dir):
    """
    Generate predicted masks for training and validation datasets.
    
    Args:
        model_path (str): Path to the YOLO model weights.
        train_image_dir (str): Directory containing training images.
        val_image_dir (str): Directory containing validation images.
        train_output_dir (str): Directory to save training masks.
        val_output_dir (str): Directory to save validation masks.
    
    Returns:
        None
    """
    # Load the YOLO modeln
    model = load_model(model_path)
    
    print("Processing Training Images...")
    process_images(model, train_image_dir, train_output_dir)
    
    print("Processing Validation Images...")
    process_images(model, val_image_dir, val_output_dir)
    
    print("Mask generation completed.")

'''if __name__ == "__main__":
    # Example usage
    MODEL_PATH = "runs/segment/train5/weights/best.pt"
    
    TRAIN_IMAGE_DIR = 'datasets/train/images'
    VAL_IMAGE_DIR = 'datasets/val/images'
    
    TRAIN_OUTPUT_DIR = 'datasets/train/predicted_masks'
    VAL_OUTPUT_DIR = 'datasets/val/predicted_masks'
    
    generate_masks(
        model_path=MODEL_PATH,
        train_image_dir=TRAIN_IMAGE_DIR,
        val_image_dir=VAL_IMAGE_DIR,
        train_output_dir=TRAIN_OUTPUT_DIR,
        val_output_dir=VAL_OUTPUT_DIR
    )'''
