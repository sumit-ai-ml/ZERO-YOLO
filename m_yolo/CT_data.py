import pandas as pd
import glob
import nibabel as nib
import numpy as np
import tqdm
import os 

df = pd.read_excel('label_names.xlsx')
label_names = df['label_names'].tolist()
multiplied_labels = df['multiplied_labels'].tolist()
print(label_names[0], len(label_names))
print(multiplied_labels[0], len(multiplied_labels))


folders = glob.glob('New/s*/se*')
folders.sort()

for folder_num in range(len(folders)):
    # Load one mask to get the shape and affine information
    file = folders[folder_num] + '/' + label_names[0]
    initial_mask_nii = nib.load(file)
    mask_shape = initial_mask_nii.get_fdata().shape
    affine = initial_mask_nii.affine

    # Initialize the combined mask with zeros
    combined_mask = np.zeros(mask_shape)
    length = 0
    empty_masks = []

    for label_num in tqdm.tqdm(range(len(label_names))):
        file = folders[folder_num] + '/' + label_names[label_num]
        current_mask = nib.load(file).get_fdata()

        # Check if the current mask has any non-zero values
        if np.any(current_mask):
            length += 1
        else:
            empty_masks.append(label_names[label_num])

        # Get the label value from 'multiplied_labels'
        label_value = multiplied_labels[label_num]
        #print(label_value, label_names[label_num])


        # Scale the mask to assign unique label values
        current_mask_scaled = current_mask * label_value

        # Identify new regions to add to the combined mask
        new_regions = (current_mask > 0) & (combined_mask == 0)
        combined_mask[new_regions] = current_mask_scaled[new_regions]

    # Exclude the background label from the count
    labels_in_mask = len(np.unique(combined_mask)) - 1  # Subtract 1 to exclude zero
    print(f"{folders[folder_num]} Labels in mask: {labels_in_mask} Length: {length}")

    # Save the combined mask
    combined_mask_nii = nib.Nifti1Image(combined_mask, affine)
    nib.save(combined_mask_nii, folders[folder_num] + '/combined_mask.nii.gz')
