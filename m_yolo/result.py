import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import tqdm 


import os
import numpy as np
import nibabel as nib
import pandas as pd

import numpy as np

def match_shapes(original_image, predicted_image):
    """
    Reorders the axes of predicted_image to match the shape of original_image,
    assuming the two shapes are permutations of each other.
    """
    original_shape = list(original_image.shape)
    predicted_shape = list(predicted_image.shape)
    
    # Ensure both images have the same number of dimensions
    if len(original_shape) != len(predicted_shape):
        raise ValueError("Images must have the same number of dimensions.")
    
    perm = []
    used_indices = []
    for dim in original_shape:
        # Find an index in predicted_shape that matches the dimension and isn't used yet.
        found = False
        for i, p_dim in enumerate(predicted_shape):
            if i not in used_indices and p_dim == dim:
                perm.append(i)
                used_indices.append(i)
                found = True
                break
        if not found:
            raise ValueError(f"Dimension {dim} not found in predicted image shape.")
    
    reordered_image = np.transpose(predicted_image, perm)
    
    if reordered_image.shape != tuple(original_shape):
        raise ValueError(f"Shape mismatch after transformation: {reordered_image.shape} vs {tuple(original_shape)}")
    
    return reordered_image


import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import glob

def create_3d_nifti_from_slices(name_file, img_file_path, pred_mask_file_path, output_path, input_path):
    """
    Creates a 3D NIfTI image from 2D predicted mask slices.

    Parameters:
    - name_file (str): Base name of the files to process (e.g., patient ID).
    - img_file_path (str): Path to the directory containing the original image slices.
    - pred_mask_file_path (str): Path to the directory containing the predicted mask slices.
    - output_path (str): Path to the directory where the output NIfTI image will be saved.

    Returns:
    - None
    """

    # Get the list of original image slices
    image_files_num = sorted(glob.glob(os.path.join(img_file_path, f"{name_file}*.tiff")))
    print('Total number of image slices:', (img_file_path))
    print('Total number of image slices:', (image_files_num))
    print('Total number of image slices:', len(image_files_num))

    # Get the list of predicted mask slices
    mask_files_num = sorted(glob.glob(os.path.join(pred_mask_file_path, f"{name_file}*.tif")))
    print('Total number of slices in predicted masks:', (mask_files_num))

    # Calculate how many slices need to be filled with blank images
    print('Need to fill images:', len(image_files_num) - len(mask_files_num))

    # Get the shape of a sample image to use for creating blank slices
    if len(mask_files_num) > 0:
        sample_img = plt.imread(mask_files_num[0])
    elif len(image_files_num) > 0:
        sample_img = plt.imread(image_files_num[0])
    else:
        print('No images found.')
        return
    print('Shape of the image:', sample_img.shape)

    img_3d = []

    # Loop over each expected slice number
    for slice_num in range(len(image_files_num)):
        # Construct the filename for the predicted mask slice
        mask_slice_filename = os.path.join(
            pred_mask_file_path,
            f"{name_file}.nii.gz_slice_{slice_num}_2_.tif"
        )
        if os.path.exists(mask_slice_filename):
            # Read the mask slice if it exists
            slice_img = plt.imread(mask_slice_filename)
        else:
            # Create a blank image if the mask slice is missing
            slice_img = np.zeros(sample_img.shape)
        img_3d.append(slice_img)

    # Convert the list of slices into a 3D NumPy array
    img_3d = np.array(img_3d)
    print('Shape of the 3D image:', img_3d.shape)

    # Create a NIfTI image from the 3D array
    # read the affine matrix from the original image
    img_file = nib.load(input_path+name_file+".nii.gz")
    affine = img_file.affine

    original_3d = img_file.get_fdata()
    print('Shape of the original 3D image:', original_3d.shape)
    print('shape of predicted image: '  , img_3d.shape)
    img_3d = match_shapes(original_3d, img_3d)
    print('shape of predicted image: '  , img_3d.shape)

    img_nifti = nib.Nifti1Image(img_3d, affine)
    print(np.unique(img_3d))


    # Save the NIfTI image to the specified output path
    output_filename = os.path.join(output_path, f"{name_file}.nii.gz")
    nib.save(img_nifti, output_filename)
    print('Saved NIfTI image to:', output_filename)

def choose_folder(folder= 'train'):
    if folder == 'train':
        img_file_path = 'dataset/'+folder+'/images/'  # Replace with the path to your image slices
        pred_mask_file_path = 'dataset/'+folder+'/masks/'  # Replace with the path to your predicted mask slices
        output_path = 'dataset/'+folder+'/nii_masks/'  # Replace with your desired output directory
        input_path = 'dataset_nii/'+folder+'/images/'  
    elif folder == 'val':
        img_file_path = 'dataset/'+folder+'/images/'
        pred_mask_file_path = 'dataset/'+folder+'/masks/'
        output_path = 'dataset/'+folder+'/nii_masks/'   
        input_path = 'dataset_nii/'+folder+'/images/'  



    print(img_file_path)
    print(pred_mask_file_path)
    print(output_path)

    os.makedirs(output_path, exist_ok=True)

    # List all files in the predicted mask file path
    file_list = os.listdir(pred_mask_file_path)
    # Sort the file names

    file = os.listdir(pred_mask_file_path)
    # sort the file names
    file.sort()
    # get all the file names using ''.join(file).split('.')[0]
    file = [''.join(file).split('.')[0] for file in file]
    file_unique = np.unique(file)
    #print(len(file_unique))
    #print(file_unique)
    name_file = file_unique[2]
    print('preparing nii files ...... ')
    for i in tqdm.tqdm(range(len(file_unique))):
        name_file = file_unique[i]
        
        create_3d_nifti_from_slices(name_file, img_file_path, pred_mask_file_path, output_path)




def compute_dice_scores(dataset_type):
    """
    Compute Dice scores for predicted and mask images in the specified dataset type.

    Parameters:
    - dataset_type (str): 'train' or 'val' to specify the dataset.

    The function saves the Dice scores to a CSV file named 'dice_scores_{dataset_type}.csv'.
    """
    # Validate dataset_type
    assert dataset_type in ['train', 'val'], "dataset_type must be 'train' or 'val'"

    # Define the paths to the folders containing the predicted and mask images
    predicted_folder = f'dataset/{dataset_type}/nii_masks/'
    mask_folder = f'dataset_nii/{dataset_type}/labels/'

    # Get sorted lists of NIfTI files in each folder
    predicted_files = sorted([
        f for f in os.listdir(predicted_folder) 
        if f.endswith('.nii') or f.endswith('.nii.gz')
    ])
    mask_files = sorted([
        f for f in os.listdir(mask_folder) 
        if f.endswith('.nii') or f.endswith('.nii.gz')
    ])

    # Ensure that the number of files in both folders is the same
    assert len(predicted_files) == len(mask_files), (
        "The number of predicted images and mask images must be the same."
    )

    # Initialize a list to store the results
    results = []

    # Iterate over each pair of predicted and mask images with a progress bar
    for pred_file, mask_file in tqdm.tqdm(zip(predicted_files, mask_files), 
                                     total=len(predicted_files), 
                                     desc=f"Processing {dataset_type} dataset"):
        pred_path = os.path.join(predicted_folder, pred_file)
        mask_path = os.path.join(mask_folder, mask_file)
        
        # Load the images using nibabel
        pred_img = nib.load(pred_path)
        mask_img = nib.load(mask_path)
        
        # Get the data arrays from the images
        pred_data = pred_img.get_fdata()
        mask_data = mask_img.get_fdata()
        
        # Get unique labels from both images, excluding the background label (assumed to be 0)
        pred_labels = np.unique(pred_data)
        mask_labels = np.unique(mask_data)
        all_labels = np.union1d(pred_labels, mask_labels)
        all_labels = all_labels[all_labels != 0]  # Exclude background label
        
        if len(all_labels) == 0:
            print(f'No labels other than background present in {pred_file} and {mask_file}.')
            continue
        
        # Compute the Dice score for each label
        for label in all_labels:
            pred_binary = (pred_data == label)
            mask_binary = (mask_data == label)
            
            intersection = np.sum(pred_binary & mask_binary)
            size_pred = np.sum(pred_binary)
            size_mask = np.sum(mask_binary)
            
            # Handle different cases to avoid division by zero
            if size_pred == 0 and size_mask == 0:
                dice = 1.0  # Perfect agreement (both images do not have the label)
            elif size_pred == 0 or size_mask == 0:
                dice = 0.0  # One image has the label while the other does not
            else:
                dice = 2.0 * intersection / (size_pred + size_mask)
            
            # Append the result to the list
            results.append({
                'Image': pred_file,
                'Label': label,
                'Dice Score': dice
            })
                
    # Convert the results to a pandas DataFrame
    results_df = pd.DataFrame(results)
    
    # Save the results to a CSV file
    output_csv = f'dice_scores_{dataset_type}.csv'
    results_df.to_csv(output_csv, index=False)
    
    # Print the mean Dice score per label across all images
    mean_dice_per_label = results_df.groupby('Label')['Dice Score'].mean()
    print('\nMean Dice Score per Label:')
    print(mean_dice_per_label)

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_dice_scores(dataset_type):
    """
    Generates combined plots for Dice scores for each class based on the computed Dice scores.

    Parameters:
    - dataset_type (str): 'train' or 'val' to specify the dataset.

    The function generates combined histograms, box plots, and line plots for all labels,
    and saves them in a directory named 'plots'.
    The plot filenames include the dataset type to distinguish between 'train' and 'val'.
    """
    # Validate dataset_type
    assert dataset_type in ['train', 'val'], "dataset_type must be 'train' or 'val'"

    # Read the Dice scores from the CSV file
    csv_file = f'dice_scores_{dataset_type}.csv'
    assert os.path.exists(csv_file), (
        f"The file {csv_file} does not exist. Please run compute_dice_scores('{dataset_type}') first."
    )

    # Load the data into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Ensure that 'Label' and 'Dice Score' columns are present
    assert 'Label' in df.columns and 'Dice Score' in df.columns, (
        "The CSV file must contain 'Label' and 'Dice Score' columns."
    )

    # Create output directory for plots (common directory for all datasets)
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)

    # Set Seaborn style for beautiful plots
    sns.set(style='whitegrid', context='talk', palette='Set2')

    # Combined box plot for all labels
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Label', y='Dice Score', data=df, palette='Set2')
    plt.title(f'Combined Box Plot of Dice Scores by Label ({dataset_type})')
    plt.xlabel('Label')
    plt.ylabel('Dice Score')
    plt.tight_layout()
    plot_filename = f'combined_boxplot_all_labels_{dataset_type}.png'
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()

    # Combined violin plot for all labels
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Label', y='Dice Score', data=df, palette='Set2', inner='quartile')
    plt.title(f'Violin Plot of Dice Scores by Label ({dataset_type})')
    plt.xlabel('Label')
    plt.ylabel('Dice Score')
    plt.tight_layout()
    plot_filename = f'combined_violinplot_all_labels_{dataset_type}.png'
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()

    # Swarm plot to show individual data points
    plt.figure(figsize=(10, 6))
    sns.swarmplot(x='Label', y='Dice Score', data=df, palette='Set2')
    plt.title(f'Swarm Plot of Dice Scores by Label ({dataset_type})')
    plt.xlabel('Label')
    plt.ylabel('Dice Score')
    plt.tight_layout()
    plot_filename = f'combined_swarmplot_all_labels_{dataset_type}.png'
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()

    # Histogram of Dice Scores for all labels
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Dice Score', hue='Label', bins=20, kde=True, palette='Set2', multiple='stack')
    plt.title(f'Histogram of Dice Scores for All Labels ({dataset_type})')
    plt.xlabel('Dice Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plot_filename = f'combined_histogram_all_labels_{dataset_type}.png'
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()

    # Line plot of Dice scores across images for all labels
    plt.figure(figsize=(12, 6))
    # Create a pivot table to have Images as x-axis, Labels as columns, Dice Scores as values
    df_pivot = df.pivot( columns='Label', values='Dice Score')
    df_pivot.plot(kind='line', marker='o', figsize=(12, 6))
    plt.title(f'Line Plot of Dice Scores Across Images for All Labels ({dataset_type})')
    plt.xlabel('Image')
    plt.ylabel('Dice Score')
    plt.xticks(rotation=90)
    plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_filename = f'combined_lineplot_all_labels_{dataset_type}.png'
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()

    print(f"Combined plots saved in '{output_dir}' directory.")

