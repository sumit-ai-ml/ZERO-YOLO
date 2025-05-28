
import os
import tqdm
from m_yolo import tiff_to_txt



def convert_masks_to_labels(input_mask_dir: str, output_label_dir: str, class_map: dict):
    """
    Converts TIFF mask files to TXT label files using the provided class mapping.

    Args:
        input_mask_dir (str): Directory containing input mask files.
        output_label_dir (str): Directory to save the output label files.
        class_map (dict): Mapping from original class IDs to new class IDs.
    """
    # Ensure the output directory exists
    os.makedirs(output_label_dir, exist_ok=True)

    # List all files in the input mask directory
    try:
        files = os.listdir(input_mask_dir)
    except FileNotFoundError:
        print(f"Input directory '{input_mask_dir}' does not exist.")
        return

    print(f"Total number of masks in '{input_mask_dir}': {len(files)}")

    # Process each file with a progress bar
    for filename in tqdm.tqdm(files, desc=f"Processing {os.path.basename(input_mask_dir)}"):
        input_file_path = os.path.join(input_mask_dir, filename)
        
        # Replace the file extension from .tiff to .txt
        output_filename = filename.replace('.tiff', '.txt')
        output_file_path = os.path.join(output_label_dir, output_filename)

        # Perform the conversion
        try:
            tiff_to_txt.convert_mask_to_segmentation_(input_file_path, output_file_path, class_map)
        except Exception as e:
            print(f"Error processing file '{filename}': {e}")



