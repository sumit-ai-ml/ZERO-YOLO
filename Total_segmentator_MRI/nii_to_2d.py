import os
from pathlib import Path
import m_yolo
from m_yolo import nii_rgb
import nibabel as nib
import numpy as np
from PIL import Image
import tqdm

def process_dataset(images_dir, masks_dir, output_dir, angle=0):
    """
    Processes NIfTI images and their corresponding masks to save them as RGB slices.

    Args:
        images_dir (str or Path): Directory containing the input images.
        masks_dir (str or Path): Directory containing the input masks.
        output_dir (str or Path): Directory where the output RGB images will be saved.
        angle (int, optional): Rotation angle for the slices. Defaults to 0.
    """
    # Convert to absolute paths
    images_dir = Path(images_dir).resolve()
    masks_dir = Path(masks_dir).resolve()
    output_dir = Path(output_dir).resolve()

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # List all image files
    image_files = list(images_dir.iterdir())

    # Process each file with a progress bar
    for image_file in tqdm.tqdm(image_files, desc=f"Processing {output_dir.name}"):
        file_name = image_file.name
        mask_file = masks_dir / file_name

        # Ensure both image and mask exist
        if not mask_file.exists():
            print(f"Warning: Mask file {mask_file} does not exist. Skipping.")
            continue

        # Save slices as RGB
        nii_rgb.save_slices_as_rgb(
            str(image_file),
            str(mask_file),
            str(output_dir),
            name=file_name,
            angle=angle
        )
