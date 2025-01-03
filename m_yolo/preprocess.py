# preprocess CT scan 
import nibabel as nib
import os 
import numpy as np
import glob 
import tqdm


def process_CT_data(input_pattern, output_file, file_index=0):
    """
    Processes MRI data by loading a specific NIfTI file, adjusting its dimensions, 
    and saving the processed data to a new file.

    Args:
        input_pattern (str): Glob pattern to locate NIfTI files.
        output_file (str): Path to save the processed NIfTI file.
        file_index (int): Index of the file to process from the glob result. Default is 0.

    Returns:
        str: Path of the processed file.
    """
    def get_data(path):
        data = nib.load(path)
        data_ = data.get_fdata()
        data_ = np.clip(data_, 0, 120) 
        
        affine = data.affine
        return data_, affine

    # Find files matching the pattern
    files = glob.glob(input_pattern)
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {input_pattern}")

    if file_index >= len(files):
        raise IndexError(f"File index {file_index} out of range. Found {len(files)} file(s).")

    file_path = files[file_index]
    #print(f"Processing file: {file_path}")

    # Load the data
    data, affine = get_data(file_path)
    #print(f"Original data shape: {data.shape}")

    # Get the least number of dimensions
    min_dim = min(data.shape)

    # Ensure min_dim is on the 3rd axis
    if min_dim == data.shape[0]:
        data = np.moveaxis(data, 0, -1)
    elif min_dim == data.shape[1]:
        data = np.moveaxis(data, 1, -1)
    elif min_dim == data.shape[2]:
        pass  # Already on the 3rd axis

    #print(f"Adjusted data shape: {data.shape}")

    # Save the processed data as a new NIfTI file
    nib.save(nib.Nifti1Image(data, affine), output_file)
    #print(f"Processed data saved to: {output_file}")

    return output_file


def process_CT_seg_data(input_pattern, output_file, file_index=0):
    """
    Processes MRI data by loading a specific NIfTI file, adjusting its dimensions, 
    and saving the processed data to a new file.

    Args:
        input_pattern (str): Glob pattern to locate NIfTI files.
        output_file (str): Path to save the processed NIfTI file.
        file_index (int): Index of the file to process from the glob result. Default is 0.

    Returns:
        str: Path of the processed file.
    """
    def get_data(path):
        data = nib.load(path)
        data_ = data.get_fdata()
        
        affine = data.affine
        return data_, affine

    # Find files matching the pattern
    files = glob.glob(input_pattern)
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {input_pattern}")

    if file_index >= len(files):
        raise IndexError(f"File index {file_index} out of range. Found {len(files)} file(s).")

    file_path = files[file_index]
    #print(f"Processing file: {file_path}")

    # Load the data
    data, affine = get_data(file_path)
    #print(f"Original data shape: {data.shape}")

    # Get the least number of dimensions
    min_dim = min(data.shape)

    # Ensure min_dim is on the 3rd axis
    if min_dim == data.shape[0]:
        data = np.moveaxis(data, 0, -1)
    elif min_dim == data.shape[1]:
        data = np.moveaxis(data, 1, -1)
    elif min_dim == data.shape[2]:
        pass  # Already on the 3rd axis

    #print(f"Adjusted data shape: {data.shape}")

    # Save the processed data as a new NIfTI file
    nib.save(nib.Nifti1Image(data, affine), output_file)
    #print(f"Processed data saved to: {output_file}")

    return output_file

def process_mri_data(input_pattern, output_file, file_index=0):
    """
    Processes MRI data by loading a specific NIfTI file, adjusting its dimensions, 
    and saving the processed data to a new file.

    Args:
        input_pattern (str): Glob pattern to locate NIfTI files.
        output_file (str): Path to save the processed NIfTI file.
        file_index (int): Index of the file to process from the glob result. Default is 0.

    Returns:
        str: Path of the processed file.
    """
    def get_data(path):
        data = nib.load(path)
        data_ = data.get_fdata()
        
        affine = data.affine
        return data_, affine

    # Find files matching the pattern
    files = glob.glob(input_pattern)
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {input_pattern}")

    if file_index >= len(files):
        raise IndexError(f"File index {file_index} out of range. Found {len(files)} file(s).")

    file_path = files[file_index]
    #print(f"Processing file: {file_path}")

    # Load the data
    data, affine = get_data(file_path)
    #print(f"Original data shape: {data.shape}")

    # Get the least number of dimensions
    min_dim = min(data.shape)

    # Ensure min_dim is on the 3rd axis
    if min_dim == data.shape[0]:
        data = np.moveaxis(data, 0, -1)
    elif min_dim == data.shape[1]:
        data = np.moveaxis(data, 1, -1)
    elif min_dim == data.shape[2]:
        pass  # Already on the 3rd axis

    #print(f"Adjusted data shape: {data.shape}")

    # Save the processed data as a new NIfTI file
    nib.save(nib.Nifti1Image(data, affine), output_file)
    #print(f"Processed data saved to: {output_file}")

    return output_file
