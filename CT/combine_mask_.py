import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd
import tqdm

def combine_mask(folder = 'MRI_data'):
    # Get the absolute path of the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script Directory: {script_dir}")

    # Define paths relative to the script's directory
    segmentation_dir = os.path.join(script_dir, folder, 's0001', 'segmentations')
    combined_scan_pattern = os.path.join(script_dir, folder, 's*', '*', 'combined*')
    folders_pattern = os.path.join(script_dir, folder, 's*', 'se*')
    label_excel_path = os.path.join(script_dir, 'label_names.xlsx')

    print(f"Segmentation Directory: {segmentation_dir}")

    # Step 1: Remove existing combined masks if any
    print('Step 1: Remove existing combined masks if any')
    
    combined_scans = glob.glob(combined_scan_pattern)
    print(f"Number of combined scans found: {len(combined_scans)}")
    
    # Remove the combined mask files
    for file in combined_scans:
        try:
            os.remove(file)
            print(f"Removed: {file}")
        except OSError as e:
            print(f"Error removing {file}: {e}")
    
    # Verify removal
    combined_scans_after = glob.glob(combined_scan_pattern)
    print(f"Number of combined scans after removal: {len(combined_scans_after)}")

    # Step 2: Create label_names.xlsx
    print('Step 2: Create label_names.xlsx')

    # Check if the segmentation directory exists
    if not os.path.exists(segmentation_dir):
        print(f"Error: The directory '{segmentation_dir}' does not exist.")
        return  # Exit the function or handle accordingly

    # List and sort the segmentation files
    file_names = os.listdir(segmentation_dir)
    if not file_names:
        print(f"No files found in '{segmentation_dir}'.")
        return

    file_names.sort()

    # Create a DataFrame with label names and multiplied labels
    df = pd.DataFrame(file_names, columns=['label_names'])
    df['multiplied_labels'] = range(1, len(file_names) + 1)

    # Save to Excel
    try:
        df.to_excel(label_excel_path, index=False)
        print(f"Saved label names to '{label_excel_path}'.")
    except Exception as e:
        print(f"Error saving '{label_excel_path}': {e}")
        return

    # Step 3: Remove combined masks again (redundant but kept as per original script)
    print('Step 3: Remove combined masks again (cleanup)')

    combined_scans = glob.glob(combined_scan_pattern)
    print(f"Number of combined scans found before second removal: {len(combined_scans)}")

    for file in combined_scans:
        try:
            os.remove(file)
            print(f"Removed: {file}")
        except OSError as e:
            print(f"Error removing {file}: {e}")

    combined_scans_after = glob.glob(combined_scan_pattern)
    print(f"Number of combined scans after second removal: {len(combined_scans_after)}")

    # Load label information from Excel
    try:
        df_labels = pd.read_excel(label_excel_path)
        label_names = df_labels['label_names'].tolist()
        multiplied_labels = df_labels['multiplied_labels'].tolist()
        print(f"Loaded {len(label_names)} label names from '{label_excel_path}'.")
    except Exception as e:
        print(f"Error reading '{label_excel_path}': {e}")
        return

    # Get list of folders containing masks
    folders = glob.glob(folders_pattern)
    folders.sort()

    if not folders:
        print(f"No folders found matching pattern '{folders_pattern}'.")
        return

    print(f"Processing {len(folders)} folders...")

    for folder in tqdm.tqdm(folders, desc="Processing folders"):
        # Initialize variables
        initial_file = os.path.join(folder, label_names[0])
        
        # Check if the initial file exists
        if not os.path.exists(initial_file):
            print(f"Initial file '{initial_file}' does not exist. Skipping folder '{folder}'.")
            continue
        
        try:
            initial_mask_nii = nib.load(initial_file)
            mask_shape = initial_mask_nii.get_fdata().shape
            affine = initial_mask_nii.affine
        except Exception as e:
            print(f"Error loading '{initial_file}': {e}. Skipping folder '{folder}'.")
            continue

        # Initialize the combined mask with zeros
        combined_mask = np.zeros(mask_shape)
        non_empty_label_count = 0
        empty_masks = []

        for label_name, label_value in zip(label_names, multiplied_labels):
            file_path = os.path.join(folder, label_name)
            
            # Check if the mask file exists
            if not os.path.exists(file_path):
                print(f"Mask file '{file_path}' does not exist. Skipping this label.")
                continue
            
            try:
                current_mask_nii = nib.load(file_path)
                current_mask = current_mask_nii.get_fdata()
            except Exception as e:
                print(f"Error loading '{file_path}': {e}. Skipping this label.")
                continue
            
            # Check if the current mask has any non-zero values
            if np.any(current_mask):
                non_empty_label_count += 1
            else:
                empty_masks.append(label_name)
                continue  # Skip processing empty masks

            # Scale the mask to assign unique label values
            current_mask_scaled = current_mask * label_value

            # Identify common regions where both masks have non-zero values
            common_regions = (current_mask > 0) & (combined_mask > 0)
            if np.any(common_regions):
                combined_mask[common_regions] = current_mask_scaled[common_regions]

            # Identify new regions to add to the combined mask where combined_mask is zero
            new_regions = (current_mask > 0) & (combined_mask == 0)
            combined_mask[new_regions] = current_mask_scaled[new_regions]

        # Exclude the background label from the count
        unique_labels = np.unique(combined_mask)
        labels_in_mask = len(unique_labels) - (1 if 0 in unique_labels else 0)

        # Save the combined mask
        combined_mask_nii = nib.Nifti1Image(combined_mask, affine)
        output_path = os.path.join(folder, 'combined_mask.nii.gz')
        try:
            nib.save(combined_mask_nii, output_path)
        except Exception as e:
            print(f"Error saving combined mask to '{output_path}': {e}")

    print('Mask combination process completed successfully.')

'''if __name__ == "__main__":
    combine_mask()'''
