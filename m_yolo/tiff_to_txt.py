import os
import nibabel as nib
import numpy as np
from PIL import Image
import tqdm
import cv2


def convert_mask_to_segmentation_(mask_path, output_txt_path, class_mapping):
    # Load the mask image
    mask_image = Image.open(mask_path)
    
    # Ensure the mask is in integer mode
    if mask_image.mode not in ['L', 'I']:
        mask_image = mask_image.convert('I')
    
    # Convert the mask image to a numpy array
    mask_np = np.array(mask_image).astype(np.int32)
    
    # Identify unique labels (exclude 0 if it's background)
    unique_labels = np.unique(mask_np)
    unique_labels = unique_labels[unique_labels != 0]
    
    # Image dimensions for normalization
    height, width = mask_np.shape[:2]
    
    # Open the output file
    with open(output_txt_path, 'w') as f:
        for label in unique_labels:
            # Verify the label exists in class_mapping
            if label not in class_mapping:
                print(f"Warning: Label {label} not found in class mapping.")
                continue
            
            # Create a binary mask for the current label
            mask_bin = (mask_np == label).astype(np.uint8)
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if len(contour) < 3:
                    continue  # Skip if contour has less than 3 points

                # Normalize and flatten the contour coordinates
                contour = contour.reshape(-1, 2).astype(float)
                contour[:, 0] = contour[:, 0] / width  # Normalize x
                contour[:, 1] = contour[:, 1] / height  # Normalize y
                segmentation = contour.flatten().round(6).tolist()
                
                # Format segmentation points as strings
                segmentation_str = ' '.join(f"{coord:.6f}" for coord in segmentation)
                class_id = class_mapping[label]
                
                # Write to file in format: <class_id> <points>
                f.write(f"{class_id} {segmentation_str}\n")
