import pandas as pd
import glob
import nibabel as nib
import numpy as np
import tqdm
import os 

import os
import glob 
combined_scan = glob.glob('New/s*/*/combined*')
print(len(combined_scan ))
# remove these files
for file in combined_scan:
    os.remove(file)
combined_scan = glob.glob('New/s*/*/combined*')
print(len(combined_scan ))


# Load label information from Excel
df = pd.read_excel('label_names.xlsx')
label_names = df['label_names'].tolist()
multiplied_labels = df['multiplied_labels'].tolist()
print(label_names[0], len(label_names))
print(multiplied_labels[0], len(multiplied_labels))

# Get list of folders containing masks
folders = glob.glob('New/s*/se*')
folders.sort()

for folder_num, folder in enumerate(tqdm.tqdm(folders, desc="Processing folders")):
    # Load one mask to get the shape and affine information
    initial_file = os.path.join(folder, label_names[0])
    initial_mask_nii = nib.load(initial_file)
    mask_shape = initial_mask_nii.get_fdata().shape
    affine = initial_mask_nii.affine

    # Initialize the combined mask with zeros
    combined_mask = np.zeros(mask_shape)
    non_empty_label_count = 0
    empty_masks = []

    for label_num, (label_name, label_value) in enumerate(tqdm.tqdm(zip(label_names, multiplied_labels))):
        file_path = os.path.join(folder, label_name)
        
        # Load current mask
        current_mask_nii = nib.load(file_path)
        current_mask = current_mask_nii.get_fdata()
        
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
            #print(f"Updated {np.sum(common_regions)} common regions for label '{label_name}' in folder '{folder}'")

        # Identify new regions to add to the combined mask where combined_mask is zero
        new_regions = (current_mask > 0) & (combined_mask == 0)
        combined_mask[new_regions] = current_mask_scaled[new_regions]

    # Exclude the background label from the count
    labels_in_mask = len(np.unique(combined_mask)) - 1  # Subtract 1 to exclude zero
    #print(f"{folder} Labels in mask: {labels_in_mask} Non-empty Labels: {non_empty_label_count}")

    # Save the combined mask
    combined_mask_nii = nib.Nifti1Image(combined_mask, affine)
    output_path = os.path.join(folder, 'combined_mask.nii.gz')
    nib.save(combined_mask_nii, output_path)
